# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The exclusive save policy: the pool holds what the GPU cache let go of.

Under `write_through` every block is written to the pool as soon as it is
computed, so the pool also holds a copy of every page still on a GPU and its
distinct capacity is roughly its own size. Under `exclusive` a page reaches the
pool only when `KVCacheManagerV2` drops it from the GPU, and a page loaded back
onto a GPU leaves the pool, so GPU and pool together hold about the sum of the
two.

A page about to be dropped is reported through the manager's victim hook, from
inside the eviction that frees it. Its slot goes to a new owner as soon as the
hook returns, so the hook copies the page into a pinned staging slot right
away and a background thread writes the slot to the pool later. When no slot
is free the hook waits for one, up to a bound, and then drops the page, which
only costs a future miss.

Removal after a load is deferred. Mooncake refuses to remove an object while a
lease from a recent lookup or read is outstanding, and the load that just read
the page holds one, so removal waits out the lease; see
`config.resolve_exclusive_remove_delay`. If the page is dropped from the GPU
again before then, the pending removal is cancelled instead of writing the page
back. A removal Mooncake still refuses, because the lease is longer than
assumed or someone else read the page since, is tried again a bounded number of
times, each after twice the previous wait. A page that is already gone counts
as removed.

Pool keys under this policy are radix-tree block keys rather than the
connector's own hash chain, since those are the only identity the victim hook
reports. Pools written under the two policies therefore never share pages.
"""

import heapq
import threading
import time
import traceback
from dataclasses import dataclass, fields
from queue import Empty, Queue
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from tensorrt_llm.logger import logger

from .addressing import PageAddressing
from .config import EXCLUSIVE_REMOVE_RETRIES
from .keys import KeyNamespace
from .staging import HostStagingPool, sync_stream

__all__ = ["ExclusiveStats", "VictimWriter"]

#: Cache level of GPU memory in `KVCacheManagerV2`.
_GPU_LEVEL = 0

#: What `MooncakeDistributedStore.remove` returns for a key the pool does not
#: hold (Mooncake's `ErrorCode::OBJECT_NOT_FOUND`).
_OBJECT_NOT_FOUND = -704

#: Seconds between two statistics lines.
_STATS_INTERVAL = 60.0


@dataclass
class ExclusiveStats:
    """Counters since the last statistics line."""

    #: Full pages the victim hook reported.
    victims: int = 0
    #: Victims whose pending removal was cancelled, keeping the pool copy.
    kept: int = 0
    #: Victims dropped because no staging slot freed up in time.
    dropped: int = 0
    #: Seconds the executor waited for a staging slot.
    wait_s: float = 0.0
    #: Pages written to the pool.
    put: int = 0
    #: Pages already present in the pool, so not written.
    present: int = 0
    #: Writes the pool refused.
    put_failed: int = 0
    #: Pages removed from the pool after a load.
    removed: int = 0
    #: Removals that found the page already gone, evicted by the master or
    #: removed by another rank that loaded it too.
    remove_absent: int = 0
    #: Removals the pool refused, usually for an unexpired lease, and that were
    #: scheduled again.
    remove_retried: int = 0
    #: Removals given up on after the pool refused every try, each leaving a
    #: duplicate of a GPU page in the pool.
    remove_failed: int = 0

    def line(self) -> str:
        """The counters as `name=value` pairs."""
        return " ".join(
            f"{field.name}={getattr(self, field.name):.3f}"
            if isinstance(getattr(self, field.name), float)
            else f"{field.name}={getattr(self, field.name)}"
            for field in fields(self)
        )


class VictimWriter:
    """Writes pages the GPU cache drops to the pool, and removes loaded ones.

    One per rank. `on_victims` runs on whichever thread evicts, inside the KV
    cache manager; everything that talks to the pool runs on this object's
    thread, which is what orders a removal before a later write of the same key.
    """

    def __init__(
        self,
        *,
        store,
        addressing: PageAddressing,
        namespaces: Mapping[int, KeyNamespace],
        staging: HostStagingPool,
        batch_size: int,
        remove_delay_s: float,
        max_wait_s: float,
        rank: int,
        device_index: Optional[int],
        remove_retries: int = EXCLUSIVE_REMOVE_RETRIES,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if remove_delay_s < 0:
            raise ValueError(f"remove_delay_s must be >= 0, got {remove_delay_s}")
        if max_wait_s < 0:
            raise ValueError(f"max_wait_s must be >= 0, got {max_wait_s}")
        if remove_retries < 0:
            raise ValueError(f"remove_retries must be >= 0, got {remove_retries}")
        self._store = store
        self._addressing = addressing
        self._namespaces = dict(namespaces)
        self._staging = staging
        self._batch_size = int(batch_size)
        self._remove_delay_s = float(remove_delay_s)
        self._remove_retries = int(remove_retries)
        self._max_wait_s = float(max_wait_s)
        self._rank = rank
        self._tokens_per_block = addressing.tokens_per_block

        self._lock = threading.Condition()
        self._free_slots: List[int] = list(range(staging.num_slots))
        #: Keys staged and not yet written, so a second report of one is ignored.
        self._queued: Set[str] = set()
        #: Key -> (due time, tries the pool already refused) of its removal. A
        #: heap entry whose due time no longer matches was cancelled or
        #: rescheduled.
        self._pending_removals: Dict[str, Tuple[float, int]] = {}
        self._removal_heap: List[Tuple[float, str]] = []
        self._stats = ExclusiveStats()
        self._stats_since = time.monotonic()

        # Created on the thread that builds the writer, which is the executor's,
        # so it belongs to the rank's device. The hook runs there too.
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )
        self._device_index = device_index
        self._queue: "Queue[Optional[List[Tuple[str, int, int]]]]" = Queue()
        self._thread = threading.Thread(
            target=self._run, name=f"mooncake-store-victims-{rank}", daemon=True
        )
        self._thread.start()
        logger.info(
            f"mooncake-store rank {rank}: exclusive save policy, "
            f"{staging.num_slots} victim staging slots of {staging.slot_bytes} B, "
            f"removal after {remove_delay_s:g}s with up to {remove_retries} retries, "
            f"wait for a slot up to {max_wait_s:g}s"
        )

    # ---- executor side ----

    def on_victims(
        self, group_offset: int, cache_level: int, pages: Sequence[Tuple[bytes, int, int, int]]
    ) -> None:
        """Copy the full pages a KV cache manager is about to drop into staging.

        Args:
            group_offset: Connector layer-group id of the manager's life cycle 0.
            cache_level: Level the pages are dropped from. Only GPU drops are
                kept: a connector run has no lower tier.
            pages: `(block_key, life_cycle, page_index, num_tokens)` per page,
                valid until this returns.
        """
        if cache_level != _GPU_LEVEL:
            return
        handle = int(self._stream.cuda_stream) if self._stream is not None else 0
        staged: List[Tuple[str, int, int]] = []
        for block_key, life_cycle, page_index, num_tokens in pages:
            if num_tokens != self._tokens_per_block:
                # A partial block is not a cache entry anywhere in the connector.
                continue
            layer_group_id = group_offset + int(life_cycle)
            namespace = self._namespaces.get(layer_group_id)
            if namespace is None:
                continue
            key = namespace.key(bytes(block_key))
            slot = self._claim_slot(key)
            if slot is None:
                continue
            addresses, sizes = self._addressing.buffers(layer_group_id, int(page_index))
            _, total = self._staging.gather(slot, addresses, sizes, handle)
            staged.append((key, slot, total))
        if not staged:
            return
        if self._stream is not None:
            # The slot is the page's only copy once this returns.
            sync_stream(handle)
        self._queue.put(staged)

    def _claim_slot(self, key: str) -> Optional[int]:
        """A free staging slot for `key`, or None when it needs none or got none."""
        with self._lock:
            self._stats.victims += 1
            if self._pending_removals.pop(key, None) is not None:
                # Loaded from the pool and dropped again before the removal
                # ran: the pool copy is the victim copy.
                self._stats.kept += 1
                return None
            if key in self._queued:
                return None
            if not self._free_slots and self._max_wait_s > 0:
                started = time.monotonic()
                self._lock.wait_for(lambda: bool(self._free_slots), timeout=self._max_wait_s)
                self._stats.wait_s += time.monotonic() - started
            if not self._free_slots:
                self._stats.dropped += 1
                return None
            self._queued.add(key)
            return self._free_slots.pop()

    def schedule_removal(self, keys: Sequence[str]) -> None:
        """Remove `keys` from the pool once the lease of the load that read them ends."""
        if not keys:
            return
        due = time.monotonic() + self._remove_delay_s
        with self._lock:
            for key in keys:
                if key in self._queued:
                    continue
                self._pending_removals[key] = (due, 0)
                heapq.heappush(self._removal_heap, (due, key))
        # Wake the writer, which may be sleeping past this due time.
        self._queue.put([])

    def shutdown(self, timeout: float = 30.0) -> None:
        """Stop the writer thread after it drains what is already staged. Idempotent."""
        thread = self._thread
        if thread is None:
            return
        self._queue.put(None)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.error(
                f"mooncake-store rank {self._rank}: the victim writer did not stop "
                f"within {timeout:g}s"
            )
            return
        self._thread = None

    @property
    def stats(self) -> ExclusiveStats:
        """A copy of the counters since the last statistics line."""
        with self._lock:
            return ExclusiveStats(
                **{f.name: getattr(self._stats, f.name) for f in fields(self._stats)}
            )

    # ---- writer thread ----

    def _run(self) -> None:
        if self._device_index is not None:
            torch.cuda.set_device(self._device_index)
        while True:
            try:
                item = self._queue.get(timeout=self._next_wakeup())
            except Empty:
                item = []
            try:
                self._remove_due()
                if item:
                    self._write(item)
            except Exception as exc:
                # Thread boundary. A lost write or removal costs a miss or a
                # duplicate, never a wrong page, so the thread carries on.
                logger.error(
                    f"mooncake-store rank {self._rank} victim writer: "
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                )
            self._maybe_log_stats()
            if item is None:
                return

    def _next_wakeup(self) -> float:
        with self._lock:
            if not self._removal_heap:
                return _STATS_INTERVAL
            return max(0.0, self._removal_heap[0][0] - time.monotonic())

    def _write(self, staged: List[Tuple[str, int, int]]) -> None:
        try:
            for start in range(0, len(staged), self._batch_size):
                batch = staged[start : start + self._batch_size]
                keys = [key for key, _, _ in batch]
                present = self._store.batch_is_exist(keys)
                pending = [i for i, status in enumerate(present) if status != 1]
                put = failed = 0
                if pending:
                    results = self._store.batch_put_from_multi_buffers(
                        [keys[i] for i in pending],
                        [[self._staging.slot_address(batch[i][1])] for i in pending],
                        [[batch[i][2]] for i in pending],
                    )
                    failed = sum(
                        1 for result in results if not isinstance(result, int) or result < 0
                    )
                    put = len(pending) - failed
                with self._lock:
                    self._stats.put += put
                    self._stats.put_failed += failed
                    self._stats.present += len(batch) - len(pending)
        finally:
            with self._lock:
                for key, slot, _ in staged:
                    self._queued.discard(key)
                    self._free_slots.append(slot)
                self._lock.notify_all()

    def _remove_due(self) -> None:
        now = time.monotonic()
        due: List[Tuple[str, int]] = []
        with self._lock:
            while self._removal_heap and self._removal_heap[0][0] <= now:
                when, key = heapq.heappop(self._removal_heap)
                pending = self._pending_removals.get(key)
                if pending is not None and pending[0] == when:
                    del self._pending_removals[key]
                    due.append((key, pending[1]))
        if not due:
            return
        removed = absent = 0
        refused: List[Tuple[str, int]] = []
        for key, refusals in due:
            result = self._store.remove(key)
            if result == 0:
                removed += 1
            elif result == _OBJECT_NOT_FOUND:
                absent += 1
            else:
                refused.append((key, refusals + 1))
        retried = failed = 0
        with self._lock:
            now = time.monotonic()
            for key, refusals in refused:
                if key in self._pending_removals or key in self._queued:
                    # Loaded again, or dropped again, while the removal was in
                    # flight: whatever that scheduled supersedes this try.
                    continue
                if refusals > self._remove_retries:
                    failed += 1
                    continue
                when = now + self._remove_delay_s * (1 << refusals)
                self._pending_removals[key] = (when, refusals)
                heapq.heappush(self._removal_heap, (when, key))
                retried += 1
            self._stats.removed += removed
            self._stats.remove_absent += absent
            self._stats.remove_retried += retried
            self._stats.remove_failed += failed

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._stats_since < _STATS_INTERVAL:
                return
            line = self._stats.line()
            pending = len(self._pending_removals)
            free = len(self._free_slots)
            self._stats = ExclusiveStats()
            self._stats_since = now
        logger.info(
            f"mooncake-store rank {self._rank} exclusive stats: {line} "
            f"pending_removals={pending} free_slots={free}"
        )
