# SPDX-License-Identifier: Apache-2.0
"""Tests for the mamba store-skip / retrieve-window logic in
``lmcache_driven_transfer``.

- ``all_null_chunk_masks`` (store side): mark chunks whose block ids are all the
  null block so ``store`` never commits them.
- ``retrieve`` (read side): read/transfer only each object group's in-window
  suffix, None-padding the skipped prefix so the transfer path is unchanged.
"""

# Standard
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
    all_null_chunk_masks,
)

# ------------------------------------------------------------------ #
#  all_null_chunk_masks (store-side skip)                              #
# ------------------------------------------------------------------ #


def _og(kernel_group_indices):
    return ObjectGroupInfo(kernel_group_indices=list(kernel_group_indices))


def test_full_attention_group_never_null():
    # One real block per chunk -> nothing skipped.
    masks = all_null_chunk_masks(
        block_ids=[[1, 2, 3]],
        object_groups=[_og([0])],
        blocks_per_chunk=[1],
        num_chunks=3,
    )
    assert masks == [[False, False, False]]


def test_mamba_group_one_block_per_chunk_marks_null_prefix():
    # Align-mamba: only the last block is real; earlier chunks are the null
    # block (id 0) and must be marked skippable.
    masks = all_null_chunk_masks(
        block_ids=[[0, 0, 0, 7]],
        object_groups=[_og([0])],
        blocks_per_chunk=[1],
        num_chunks=4,
    )
    assert masks == [[True, True, True, False]]


def test_multi_block_per_chunk_null_only_when_all_blocks_zero():
    # chunk size = 2 blocks. Chunk 0 = [0, 0] (null), chunk 1 = [0, 9] (has a
    # real block in its second slot) -> not null.
    masks = all_null_chunk_masks(
        block_ids=[[0, 0, 0, 9]],
        object_groups=[_og([0])],
        blocks_per_chunk=[2],
        num_chunks=2,
    )
    assert masks == [[True, False]]


def test_two_object_groups_independent():
    # Group 0 = full attention (kernel group 0, all real); group 1 = mamba
    # (kernel group 1, null prefix). Masks are per object group.
    masks = all_null_chunk_masks(
        block_ids=[[1, 2, 3], [0, 0, 5]],
        object_groups=[_og([0]), _og([1])],
        blocks_per_chunk=[1, 1],
        num_chunks=3,
    )
    assert masks == [[False, False, False], [True, True, False]]


def test_object_group_null_only_when_all_its_kernel_groups_null():
    # An object group spanning two kernel groups: a chunk is null only if every
    # kernel group's blocks for that chunk are null.
    masks = all_null_chunk_masks(
        block_ids=[[0, 0], [0, 4]],
        object_groups=[_og([0, 1])],
        blocks_per_chunk=[1, 1],
        num_chunks=2,
    )
    # chunk 0: kg0=0 and kg1=0 -> null; chunk 1: kg0=0 but kg1=4 -> not null.
    assert masks == [[True, False]]


# ------------------------------------------------------------------ #
#  retrieve (read-side window)                                         #
# ------------------------------------------------------------------ #


def _make_module(monkeypatch, num_chunks, num_chunks_in_sw):
    """Build an LMCacheDrivenTransferModule with its collaborators mocked, and
    return (module, read_calls, transfer_calls) capturing what retrieve reads
    and transfers per object group."""
    num_object_groups = len(num_chunks_in_sw)

    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)

    kvlgm = SimpleNamespace(
        num_object_groups=num_object_groups,
        num_kernel_groups=num_object_groups,
        object_groups=[_og([g]) for g in range(num_object_groups)],
        get_attn_desc=lambda: SimpleNamespace(num_chunks_in_sw=num_chunks_in_sw),
        get_subchunk_sw_size_tokens=lambda kg: 256,
    )
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager = kvlgm
    cache_context.calculate_num_blocks.return_value = 1  # 1 block per chunk
    cache_context.lmcache_tokens_per_chunk = 256
    cache_context.max_batch_size = 8

    event_backend = MagicMock()
    entry = SimpleNamespace(
        cache_context=cache_context, model_name="m", event_backend=event_backend
    )
    module.get_and_touch_context_entry = MagicMock(return_value=entry)

    # Object keys: one distinct key per (group, chunk).
    obj_keys = [
        [f"g{g}c{c}" for c in range(num_chunks)] for g in range(num_object_groups)
    ]
    ctx = MagicMock()
    ctx.chunk_size = 256
    ctx.retrieve_window_chunks = 2
    ctx.resolve_obj_keys.return_value = obj_keys

    read_calls: list[list[str]] = []

    @contextmanager
    def fake_read(keys):
        read_calls.append(list(keys))
        yield [MagicMock(get_size=MagicMock(return_value=10)) for _ in keys]

    ctx.storage_manager.read_prefetched_results = MagicMock(side_effect=fake_read)
    module._ctx = ctx

    transfer_calls: list[tuple[int, list, list, int]] = []

    def fake_transfer(
        cache_context,
        block_ids,
        memory_objs,
        object_group_id,
        batch_size,
        skip_first_n_tokens,
        direction,
    ):
        transfer_calls.append(
            (object_group_id, list(memory_objs), list(block_ids), skip_first_n_tokens)
        )

    stream_releases: list[list[str]] = []

    def fake_stream_release(stream, kind, keys):
        stream_releases.append((kind, list(keys)))

    monkeypatch.setattr(mod, "transfer_kv_per_object_group", fake_transfer)
    monkeypatch.setattr(mod, "downsample_and_stage_block_ids", lambda cc, b, num_chunks=0: b)
    monkeypatch.setattr(mod, "submit_callback_to_stream", fake_stream_release)
    monkeypatch.setattr(mod, "torch_dev", MagicMock())
    monkeypatch.setattr(mod, "Event", MagicMock())

    return module, read_calls, transfer_calls, stream_releases


def test_retrieve_reads_and_transfers_only_in_window(monkeypatch):
    # Group 0 = full attention (-1): whole prefix; group 1 = mamba window 1:
    # only the last chunk. retrieve_window_chunks=2 (set in _make_module).
    num_chunks = 5
    module, read_calls, transfer_calls, stream_releases = _make_module(
        monkeypatch, num_chunks, num_chunks_in_sw=[-1, 1]
    )
    # 1 block per chunk -> block-id lists of length num_chunks (avoid underflow).
    gpu_block_ids = [[1, 2, 3, 4, 5], [0, 0, 0, 0, 9]]

    _handle, ok = module.retrieve(
        key=SimpleNamespace(request_id="req", cache_salt="salt"),
        instance_id=1,
        gpu_block_ids=gpu_block_ids,
        event_ipc_handle=b"x",
    )
    assert ok is True

    # Reads stream in windows of 2 chunks per group; the mamba group reads
    # only its in-window tail.
    assert read_calls == [
        ["g0c0", "g0c1"],
        ["g0c2", "g0c3"],
        ["g0c4"],
        ["g1c4"],
    ]

    # memory_objs are window-local (no None padding); block ids are sliced
    # to the same chunk window (1 block per chunk here).
    # block ids arrive per kernel group; each object group here maps 1:1.
    assert [(g, blocks[g]) for g, _mem, blocks, _skip in transfer_calls] == [
        (0, [1, 2]),
        (0, [3, 4]),
        (0, [5]),
        (1, [9]),
    ]
    assert all(
        all(o is not None for o in mem) for _g, mem, _b, _s in transfer_calls
    )

    # Each window's L1 read locks are released in stream order right after
    # its copy is enqueued.
    assert stream_releases == [
        ("finish_read_prefetched", ["g0c0", "g0c1"]),
        ("finish_read_prefetched", ["g0c2", "g0c3"]),
        ("finish_read_prefetched", ["g0c4"]),
        ("finish_read_prefetched", ["g1c4"]),
    ]


def test_retrieve_full_attention_only_reads_everything(monkeypatch):
    # Single full-attention group: every chunk is read and transferred,
    # streamed in windows.
    num_chunks = 3
    module, read_calls, transfer_calls, stream_releases = _make_module(
        monkeypatch, num_chunks, num_chunks_in_sw=[-1]
    )
    _handle, ok = module.retrieve(
        key=SimpleNamespace(request_id="req", cache_salt="salt"),
        instance_id=1,
        gpu_block_ids=[[1, 2, 3]],
        event_ipc_handle=b"x",
    )
    assert ok is True
    assert read_calls == [["g0c0", "g0c1"], ["g0c2"]]
    for _g, mem, _b, _s in transfer_calls:
        assert all(o is not None for o in mem)
    assert [blocks[0] for _g, _m, blocks, _s in transfer_calls] == [[1, 2], [3]]
    assert stream_releases == [
        ("finish_read_prefetched", ["g0c0", "g0c1"]),
        ("finish_read_prefetched", ["g0c2"]),
    ]


def test_retrieve_window_skip_tokens_shift_per_window(monkeypatch):
    # skip_first_n_tokens shifts into window-local coordinates: only the
    # first windows keep a nonzero skip.
    num_chunks = 5
    module, _read_calls, transfer_calls, _releases = _make_module(
        monkeypatch, num_chunks, num_chunks_in_sw=[-1]
    )
    _handle, ok = module.retrieve(
        key=SimpleNamespace(request_id="req", cache_salt="salt"),
        instance_id=1,
        gpu_block_ids=[[1, 2, 3, 4, 5]],
        event_ipc_handle=b"x",
        skip_first_n_tokens=300,
    )
    assert ok is True
    skips = [skip for _g, _m, _b, skip in transfer_calls]
    assert skips == [300, 0, 0]


def test_slice_block_ids_for_window_contiguous():
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager = SimpleNamespace(
        num_kernel_groups=2,
        get_subchunk_sw_size_tokens=lambda kg: 256 if kg == 0 else 128,
    )
    cache_context.lmcache_tokens_per_chunk = 256
    # keep = 1 block/chunk for kg0 (min(256,256)=256), 2 for kg1 (min(256,128)=128)
    cache_context.calculate_num_blocks = MagicMock(
        side_effect=lambda tokens, kg: 1 if tokens == 256 else 2
    )
    staged = [list(range(10)), list(range(20))]

    sliced = mod._slice_block_ids_for_window(
        cache_context, staged, start_chunk=2, num_chunks=3
    )
    assert sliced[0] == [2, 3, 4]
    assert sliced[1] == [4, 5, 6, 7, 8, 9]
