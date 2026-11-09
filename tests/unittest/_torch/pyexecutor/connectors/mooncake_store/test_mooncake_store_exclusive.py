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
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import config as config_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    EXCLUSIVE_REMOVE_DELAY_ENV,
    MooncakeStoreConnectorConfig,
    SavePolicy,
    resolve_exclusive_remove_delay,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.exclusive import VictimWriter
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import KeyNamespace
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    PoolManifest,
    _client_config,
    _lease_flag,
)
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
        self.remove_calls = []
        #: Key -> how many more removals of it to refuse; -1 refuses forever.
        self.refuse_removal = {}
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
        self.remove_calls.append(key)
        refusals = self.refuse_removal.get(key, 0)
        if refusals:
            self.refuse_removal[key] = refusals - 1 if refusals > 0 else refusals
            return -706  # OBJECT_HAS_LEASE
        if key not in self.objects:
            return -704  # OBJECT_NOT_FOUND
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


def make_writer(
    store, *, num_slots=8, remove_delay_s=0.0, max_wait_s=0.0, groups=(0,), remove_retries=3
):
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
        remove_retries=remove_retries,
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
    writer.shutdown()
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


def test_one_report_of_more_pages_than_slots_waits_for_the_writer_not_the_timeout(
    writer_factory,
):
    store = FakeStore()
    writer = writer_factory(store, num_slots=2, max_wait_s=30.0)
    block_keys = [bytes([i]) * 32 for i in range(1, 8)]
    started = time.monotonic()
    writer.on_victims(
        0, 0, [(block_key, 0, i, TOKENS_PER_BLOCK) for i, block_key in enumerate(block_keys)]
    )
    assert time.monotonic() - started < 10.0
    writer.shutdown()
    assert store.put_keys == [key(block_key) for block_key in block_keys]
    stats = writer.stats
    assert stats.victims == len(block_keys)
    assert stats.dropped == 0


def test_loaded_page_is_removed_after_the_delay(writer_factory):
    store = FakeStore()
    loaded = key(b"\x01" * 32)
    store.objects.add(loaded)
    writer = writer_factory(store, remove_delay_s=0.05)
    writer.schedule_removal([loaded])
    assert store.removed == []
    assert wait_until(lambda: store.removed == [loaded])
    writer.shutdown()
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


def test_a_refused_removal_is_retried_until_it_succeeds(writer_factory):
    store = FakeStore()
    leased = key(b"\x01" * 32)
    store.objects.add(leased)
    store.refuse_removal[leased] = 2
    writer = writer_factory(store, remove_delay_s=0.01)
    writer.schedule_removal([leased])
    assert wait_until(lambda: store.removed == [leased])
    writer.shutdown()
    assert store.remove_calls == [leased] * 3
    stats = writer.stats
    assert (stats.removed, stats.remove_retried, stats.remove_failed) == (1, 2, 0)
    assert "remove_retried=2" in stats.line()
    assert not writer._pending_removals


def test_a_removal_refused_every_time_is_given_up(writer_factory):
    store = FakeStore()
    leased = key(b"\x01" * 32)
    store.objects.add(leased)
    store.refuse_removal[leased] = -1
    writer = writer_factory(store, remove_delay_s=0.0, remove_retries=3)
    writer.schedule_removal([leased])
    assert wait_until(lambda: writer.stats.remove_failed == 1)
    writer.shutdown()
    assert store.remove_calls == [leased] * 4
    assert writer.stats.remove_retried == 3
    assert not writer._pending_removals
    assert leased in store.objects


def test_retries_wait_twice_as_long_each_time(writer_factory):
    store = FakeStore()
    leased = key(b"\x01" * 32)
    store.objects.add(leased)
    store.refuse_removal[leased] = -1
    writer = writer_factory(store, remove_delay_s=0.2)
    scheduled = time.monotonic()
    writer.schedule_removal([leased])
    assert wait_until(lambda: len(store.remove_calls) == 1)
    with writer._lock:
        due, refusals = writer._pending_removals[leased]
    writer.shutdown()
    assert refusals == 1
    # The first try ran after 0.2s; the retry waits 2 x 0.2s more.
    assert due - scheduled >= 0.2 + 0.4 - 0.05


def test_a_page_already_gone_counts_as_removed(writer_factory):
    store = FakeStore()
    gone = key(b"\x01" * 32)
    writer = writer_factory(store, remove_delay_s=0.0)
    writer.schedule_removal([gone])
    assert wait_until(lambda: writer.stats.remove_absent == 1)
    writer.shutdown()
    assert store.remove_calls == [gone]
    assert (writer.stats.remove_retried, writer.stats.remove_failed) == (0, 0)


def test_dropping_a_page_while_its_removal_awaits_a_retry_keeps_the_pool_copy(writer_factory):
    store = FakeStore()
    leased = key(b"\x01" * 32)
    store.objects.add(leased)
    store.refuse_removal[leased] = 1
    writer = writer_factory(store, remove_delay_s=0.05)
    writer.schedule_removal([leased])
    assert wait_until(lambda: writer.stats.remove_retried == 1)
    writer.on_victims(0, 0, [(b"\x01" * 32, 0, 1, TOKENS_PER_BLOCK)])
    time.sleep(0.3)
    writer.shutdown()
    assert store.remove_calls == [leased]
    assert store.put_keys == []
    assert leased in store.objects
    assert writer.stats.kept == 1


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


# ---- removal delay ----


def read_config(tmp_path, **settings):
    return MooncakeStoreConnectorConfig.from_file(str(write_config(tmp_path, **settings)))


def test_removal_delay_defaults_to_the_manifest_lease_plus_a_margin(tmp_path):
    delay = resolve_exclusive_remove_delay(read_config(tmp_path, kv_lease_ttl_s=10.0), environ={})
    assert delay.seconds == 11.0
    assert (delay.lease_s, delay.lease_known, delay.explicit) == (10.0, True, False)
    assert "pool manifest" in delay.source
    assert delay.warning is None


def test_removal_delay_assumes_mooncakes_default_lease_for_an_external_master(tmp_path):
    delay = resolve_exclusive_remove_delay(read_config(tmp_path), environ={})
    assert delay.seconds == 6.0
    assert (delay.lease_s, delay.lease_known) == (5.0, False)
    assert "Mooncake's default" in delay.source


def test_removal_delay_set_in_the_config_wins_over_the_lease(tmp_path):
    config = read_config(tmp_path, kv_lease_ttl_s=10.0, exclusive_remove_delay_s=20)
    delay = resolve_exclusive_remove_delay(config, environ={})
    assert (delay.seconds, delay.source, delay.explicit) == (20.0, "exclusive_remove_delay_s", True)
    assert delay.warning is None


def test_removal_delay_environment_override_wins_over_the_config(tmp_path):
    config = read_config(tmp_path, kv_lease_ttl_s=10.0, exclusive_remove_delay_s=20)
    delay = resolve_exclusive_remove_delay(config, environ={EXCLUSIVE_REMOVE_DELAY_ENV: " 11 "})
    assert (delay.seconds, delay.source) == (11.0, EXCLUSIVE_REMOVE_DELAY_ENV)


def test_removal_delay_reads_the_process_environment_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv(EXCLUSIVE_REMOVE_DELAY_ENV, "12.5")
    assert resolve_exclusive_remove_delay(read_config(tmp_path)).seconds == 12.5


@pytest.mark.parametrize("value", ["soon", "-1", "nan", "inf"])
def test_removal_delay_environment_override_must_be_seconds(tmp_path, value):
    with pytest.raises(ValueError, match=EXCLUSIVE_REMOVE_DELAY_ENV):
        resolve_exclusive_remove_delay(
            read_config(tmp_path), environ={EXCLUSIVE_REMOVE_DELAY_ENV: value}
        )


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"exclusive_remove_delay_s": -1}, "exclusive_remove_delay_s"),
        ({"kv_lease_ttl_s": 0}, "kv_lease_ttl_s"),
    ],
)
def test_removal_settings_are_validated(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        read_config(tmp_path, **settings)


class RecordingLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(message)

    def warning(self, message):
        self.warnings.append(message)


@pytest.mark.parametrize("seconds", ["6", "10"])
def test_a_delay_that_does_not_outlast_the_known_lease_is_warned_about(
    tmp_path, monkeypatch, seconds
):
    recorder = RecordingLogger()
    monkeypatch.setattr(config_module, "logger", recorder)
    delay = resolve_exclusive_remove_delay(
        read_config(tmp_path, kv_lease_ttl_s=10.0),
        environ={EXCLUSIVE_REMOVE_DELAY_ENV: seconds},
    )
    delay.log(rank=3)
    assert len(recorder.infos) == 1 and "rank 3" in recorder.infos[0]
    assert len(recorder.warnings) == 1
    assert "pool's 10s object lease" in recorder.warnings[0]
    assert EXCLUSIVE_REMOVE_DELAY_ENV in recorder.warnings[0]


def test_a_delay_under_the_assumed_default_lease_is_warned_about(tmp_path):
    delay = resolve_exclusive_remove_delay(
        read_config(tmp_path, exclusive_remove_delay_s=3), environ={}
    )
    assert "Mooncake's default 5s object lease" in delay.warning


def test_a_derived_delay_logs_its_source_without_a_warning(tmp_path, monkeypatch):
    recorder = RecordingLogger()
    monkeypatch.setattr(config_module, "logger", recorder)
    resolve_exclusive_remove_delay(read_config(tmp_path), environ={}).log(rank=0)
    assert recorder.warnings == []
    assert "Mooncake's default" in recorder.infos[0]


# ---- lease in the pool manifest ----


def server_block(**settings):
    """The fields of `MooncakeStoreConfig` that `_client_config` reads."""
    block = {
        "namespace": None,
        "model_key": "test-model",
        "local_hostname": None,
        "segment_size": "1GiB",
        "role": "both",
        "transfer_batch_size": 64,
        "stage_through_host": False,
        "save_policy": "exclusive",
        "exclusive_remove_delay_s": None,
    }
    block.update(settings)
    return SimpleNamespace(**block)


def test_the_manifest_lease_reaches_the_ranks_and_sets_their_delay(tmp_path):
    published = PoolManifest(master_server_address="10.0.0.1:50051", kv_lease_ttl_s=10.0)
    manifest = PoolManifest.from_json(json.loads(json.dumps(published.to_json())))
    assert manifest.kv_lease_ttl_s == 10.0
    assert "lease=10s" in manifest.describe()
    rendered = _client_config(server_block(), manifest, device_name="")
    assert rendered["kv_lease_ttl_s"] == 10.0
    assert "exclusive_remove_delay_s" not in rendered
    path = tmp_path / "mooncake.json"
    path.write_text(json.dumps(rendered))
    config = MooncakeStoreConnectorConfig.from_file(str(path))
    assert resolve_exclusive_remove_delay(config, environ={}).seconds == 11.0


def test_a_manifest_without_a_lease_leaves_the_ranks_to_assume_the_default():
    manifest = PoolManifest.from_json({"master_server_address": "10.0.0.1:50051"})
    assert manifest.kv_lease_ttl_s is None
    assert "kv_lease_ttl_s" not in manifest.to_json()
    assert "kv_lease_ttl_s" not in _client_config(server_block(), manifest, device_name="")


def test_a_server_set_delay_is_rendered_for_its_ranks():
    manifest = PoolManifest(master_server_address="10.0.0.1:50051", kv_lease_ttl_s=10.0)
    rendered = _client_config(server_block(exclusive_remove_delay_s=15.0), manifest, "")
    assert rendered["exclusive_remove_delay_s"] == 15.0


@pytest.mark.parametrize("value", [0, -5, "ten"])
def test_a_manifest_lease_must_be_positive_seconds(value):
    with pytest.raises(ValueError, match="kv_lease_ttl_s"):
        PoolManifest.from_json(
            {"master_server_address": "10.0.0.1:50051", "kv_lease_ttl_s": value}, "pool.json"
        )


def test_the_master_is_given_its_lease_in_milliseconds():
    assert _lease_flag(10.0) == "--default_kv_lease_ttl=10000"
    assert _lease_flag(5) == "--default_kv_lease_ttl=5000"
    with pytest.raises(ValueError, match="kv_lease_ttl_s"):
        _lease_flag(0.0001)


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
