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
"""Unit tests for the mooncake-store exclusive save policy.

Runs without Mooncake and without a GPU: the store is an in-process fake, the
staging slots record what was copied into them, and the KV cache manager is a
stand-in that only names blocks.
"""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    RequestData,
    SchedulerOutput,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    MooncakeStoreConnectorConfig,
    SavePolicy,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.exclusive import VictimWriter
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import KeyNamespace
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.scheduler import (
    MooncakeStoreConnectorScheduler,
)

pytestmark = pytest.mark.cpu_only

TOKENS_PER_BLOCK = 4
PAGE_BYTES = 32


class FakeStore:
    """Remembers which keys exist and how each call went."""

    def __init__(self):
        self.objects = set()
        self.put_keys = []
        self.removed = []
        self.refuse_removal = set()
        self.put_gate = threading.Event()
        self.put_gate.set()

    def batch_is_exist(self, keys):
        return [1 if key in self.objects else 0 for key in keys]

    def batch_put_from_multi_buffers(self, keys, addresses, sizes):
        self.put_gate.wait(timeout=10)
        self.put_keys.extend(keys)
        self.objects.update(keys)
        return [sum(size) for size in sizes]

    def remove(self, key):
        if key in self.refuse_removal:
            return -706  # OBJECT_HAS_LEASE
        self.removed.append(key)
        self.objects.discard(key)
        return 0


class FakeStaging:
    """Slots that record which page was gathered into them."""

    def __init__(self, num_slots):
        self.num_slots = num_slots
        self.slot_bytes = PAGE_BYTES
        self.gathered = {}

    def gather(self, index, addresses, sizes, stream):
        self.gathered[index] = (tuple(addresses), tuple(sizes))
        return self.slot_address(index), sum(sizes)

    def slot_address(self, index):
        return 0x10000 + index * PAGE_BYTES


class FakeAddressing:
    tokens_per_block = TOKENS_PER_BLOCK

    def buffers(self, layer_group_id, page_index):
        return [0x1000 * (layer_group_id + 1) + page_index * PAGE_BYTES], [PAGE_BYTES]


def namespace(layer_group_id):
    return KeyNamespace(
        namespace="ns",
        model_key="m",
        rank=0,
        world_size=1,
        layer_group_id=layer_group_id,
        tokens_per_block=TOKENS_PER_BLOCK,
        bytes_per_page=PAGE_BYTES,
    )


def make_writer(store, *, num_slots=8, remove_delay_s=0.0, max_wait_s=0.0, groups=(0,)):
    return VictimWriter(
        store=store,
        addressing=FakeAddressing(),
        namespaces={group: namespace(group) for group in groups},
        staging=FakeStaging(num_slots),
        batch_size=4,
        remove_delay_s=remove_delay_s,
        max_wait_s=max_wait_s,
        rank=0,
        device_index=None,
    )


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def key(block_key, group=0):
    return namespace(group).key(block_key)


@pytest.fixture
def writer_factory():
    writers = []

    def factory(store, **kwargs):
        writer = make_writer(store, **kwargs)
        writers.append(writer)
        return writer

    yield factory
    for writer in writers:
        writer.shutdown()


# ---- VictimWriter ----


def test_victim_writer_writes_only_full_gpu_pages_of_known_groups(writer_factory):
    store = FakeStore()
    writer = writer_factory(store, groups=(0, 2))
    writer.on_victims(
        1,
        0,
        [
            (b"\x01" * 32, 1, 5, TOKENS_PER_BLOCK),  # group 1 + offset 1 = 2
            (b"\x02" * 32, 0, 6, TOKENS_PER_BLOCK - 1),  # partial block
            (b"\x03" * 32, 5, 7, TOKENS_PER_BLOCK),  # group 6 is not registered
        ],
    )
    writer.on_victims(0, 1, [(b"\x04" * 32, 0, 8, TOKENS_PER_BLOCK)])  # host level
    assert wait_until(lambda: store.put_keys == [key(b"\x01" * 32, group=2)])
    writer.shutdown()
    assert store.put_keys == [key(b"\x01" * 32, group=2)]


def test_victim_writer_copies_the_page_the_hook_names(writer_factory):
    store = FakeStore()
    writer = writer_factory(store, num_slots=1)
    writer.on_victims(0, 0, [(b"\x01" * 32, 0, 3, TOKENS_PER_BLOCK)])
    assert writer._staging.gathered[0] == ((0x1000 + 3 * PAGE_BYTES,), (PAGE_BYTES,))


def test_victim_writer_skips_pages_the_pool_already_has(writer_factory):
    store = FakeStore()
    store.objects.add(key(b"\x01" * 32))
    writer = writer_factory(store)
    writer.on_victims(0, 0, [(b"\x01" * 32, 0, 3, TOKENS_PER_BLOCK)])
    writer.shutdown()
    assert store.put_keys == []


def test_victim_writer_drops_a_page_when_no_slot_frees_up(writer_factory):
    store = FakeStore()
    store.put_gate.clear()
    writer = writer_factory(store, num_slots=1, max_wait_s=0.0)
    writer.on_victims(
        0,
        0,
        [(b"\x01" * 32, 0, 1, TOKENS_PER_BLOCK), (b"\x02" * 32, 0, 2, TOKENS_PER_BLOCK)],
    )
    assert writer.stats.dropped == 1
    store.put_gate.set()
    writer.shutdown()
    assert store.put_keys == [key(b"\x01" * 32)]


def test_victim_writer_waits_for_a_slot_when_allowed(writer_factory):
    store = FakeStore()
    writer = writer_factory(store, num_slots=1, max_wait_s=5.0)
    writer.on_victims(0, 0, [(b"\x01" * 32, 0, 1, TOKENS_PER_BLOCK)])
    writer.on_victims(0, 0, [(b"\x02" * 32, 0, 2, TOKENS_PER_BLOCK)])
    writer.shutdown()
    assert store.put_keys == [key(b"\x01" * 32), key(b"\x02" * 32)]
    assert writer.stats.dropped == 0


def test_loaded_page_is_removed_after_the_delay(writer_factory):
    store = FakeStore()
    loaded = key(b"\x01" * 32)
    store.objects.add(loaded)
    writer = writer_factory(store, remove_delay_s=0.05)
    writer.schedule_removal([loaded])
    assert store.removed == []
    assert wait_until(lambda: store.removed == [loaded])
    assert loaded not in store.objects


def test_dropping_a_loaded_page_again_keeps_the_pool_copy(writer_factory):
    store = FakeStore()
    loaded = key(b"\x01" * 32)
    store.objects.add(loaded)
    writer = writer_factory(store, remove_delay_s=60.0)
    writer.schedule_removal([loaded])
    writer.on_victims(0, 0, [(b"\x01" * 32, 0, 1, TOKENS_PER_BLOCK)])
    writer.shutdown()
    assert store.removed == []
    assert store.put_keys == []
    assert loaded in store.objects


def test_a_refused_removal_is_not_retried(writer_factory):
    store = FakeStore()
    leased = key(b"\x01" * 32)
    store.objects.add(leased)
    store.refuse_removal.add(leased)
    writer = writer_factory(store, remove_delay_s=0.0)
    writer.schedule_removal([leased])
    assert wait_until(lambda: not writer._pending_removals and not writer._removal_heap)
    writer.shutdown()
    assert leased in store.objects
    assert store.removed == []


def test_a_page_dropped_after_its_removal_is_written_back(writer_factory):
    store = FakeStore()
    loaded = key(b"\x01" * 32)
    store.objects.add(loaded)
    writer = writer_factory(store, remove_delay_s=0.0)
    writer.schedule_removal([loaded])
    assert wait_until(lambda: store.removed == [loaded])
    writer.on_victims(0, 0, [(b"\x01" * 32, 0, 1, TOKENS_PER_BLOCK)])
    writer.shutdown()
    assert store.put_keys == [loaded]


# ---- config ----


def write_config(tmp_path, **settings):
    raw = {
        "master_server_address": "127.0.0.1:50051",
        "protocol": "tcp",
        "model_key": "test-model",
        **settings,
    }
    path = tmp_path / "mooncake.json"
    path.write_text(json.dumps(raw))
    return path


def test_save_policy_defaults_to_write_through(tmp_path):
    config = MooncakeStoreConnectorConfig.from_file(str(write_config(tmp_path)))
    assert config.save_policy is SavePolicy.WRITE_THROUGH
    assert not config.exclusive


def test_save_policy_reads_exclusive(tmp_path):
    config = MooncakeStoreConnectorConfig.from_file(
        str(write_config(tmp_path, save_policy="Exclusive", victim_staging_bytes="64MiB"))
    )
    assert config.exclusive
    assert config.victim_staging_bytes == 64 << 20


def test_save_policy_rejects_an_unknown_value(tmp_path):
    with pytest.raises(ValueError, match="save_policy"):
        MooncakeStoreConnectorConfig.from_file(str(write_config(tmp_path, save_policy="lazy")))


# ---- scheduler ----


class FakeWorker:
    def __init__(self, hit_blocks):
        self.hit_blocks = hit_blocks
        self.lookups = []

    def count_prefix_hit(self, block_hashes):
        self.lookups.append(list(block_hashes))
        return min(self.hit_blocks, len(block_hashes))


class FakeManager:
    """Names block `i` of any request by a key derived from `i` alone."""

    def reuse_block_keys(self, request):
        tokens = request.get_tokens(0)
        return [bytes([i]) * 32 for i in range((len(tokens) - 1) // TOKENS_PER_BLOCK)]


def make_request(request_id, tokens):
    return SimpleNamespace(
        request_id=request_id,
        cache_salt=None,
        lora_task_id=None,
        multimodal_hashes=None,
        multimodal_positions=None,
        multimodal_lengths=None,
        get_tokens=lambda _beam=0, _tokens=tuple(tokens): list(_tokens),
    )


def request_data(request_id, new_tokens, page_indices):
    return RequestData(
        request_id=request_id,
        new_tokens=list(new_tokens),
        new_block_ids=list(page_indices),
        computed_position=0,
        num_scheduled_tokens=len(new_tokens),
        new_block_ids_by_layer_group=[list(page_indices)],
    )


@pytest.fixture
def exclusive_scheduler(tmp_path, monkeypatch):
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(write_config(tmp_path, save_policy="exclusive")))
    llm_args = SimpleNamespace(
        model="/models/test-model",
        kv_cache_config=SimpleNamespace(tokens_per_block=TOKENS_PER_BLOCK),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        sparse_attention_config=None,
        kv_connector_config=SimpleNamespace(
            connector="mooncake-store", mooncake_store=SimpleNamespace(run_dir=None)
        ),
    )
    scheduler = MooncakeStoreConnectorScheduler(llm_args)
    scheduler._worker = FakeWorker(hit_blocks=2)
    scheduler.attach_kv_cache_managers([(FakeManager(), 0)])
    return scheduler


def test_exclusive_lookup_uses_the_managers_block_keys(exclusive_scheduler):
    tokens = list(range(4 * TOKENS_PER_BLOCK + 1))
    matched, is_async = exclusive_scheduler.get_num_new_matched_tokens(
        make_request(1, tokens), TOKENS_PER_BLOCK
    )
    assert (matched, is_async) == (2 * TOKENS_PER_BLOCK, False)
    assert exclusive_scheduler._worker.lookups == [[bytes([i]) * 32 for i in (1, 2, 3)]]


def test_exclusive_loads_by_block_key_and_saves_nothing(exclusive_scheduler):
    tokens = list(range(4 * TOKENS_PER_BLOCK + 1))
    request = make_request(1, tokens)
    exclusive_scheduler.get_num_new_matched_tokens(request, 0)
    metadata = exclusive_scheduler.build_connector_meta(
        SchedulerOutput(new_requests=[request_data(1, tokens, [10, 11, 12, 13, 14])])
    )
    assert [(page.block_hash, page.page_index) for page in metadata.loads[0].pages] == [
        (bytes([0]) * 32, 10),
        (bytes([1]) * 32, 11),
    ]
    assert metadata.saves == []
    assert exclusive_scheduler.request_finished(request, []) is False


def test_exclusive_lookup_needs_an_attached_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(write_config(tmp_path, save_policy="exclusive")))
    llm_args = SimpleNamespace(
        model="/models/test-model",
        kv_cache_config=SimpleNamespace(tokens_per_block=TOKENS_PER_BLOCK),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        sparse_attention_config=None,
        kv_connector_config=SimpleNamespace(
            connector="mooncake-store", mooncake_store=SimpleNamespace(run_dir=None)
        ),
    )
    scheduler = MooncakeStoreConnectorScheduler(llm_args)
    scheduler._worker = FakeWorker(hit_blocks=1)
    with pytest.raises(RuntimeError, match="attach_kv_cache_managers"):
        scheduler.get_num_new_matched_tokens(make_request(1, list(range(9))), 0)
