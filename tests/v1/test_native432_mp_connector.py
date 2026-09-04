# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the native432 LMCache MP connector gate."""

# Standard
from dataclasses import dataclass, field

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
from lmcache.integration.vllm.native432_mp_connector import (
    NATIVE432_C4_LAYER_IDS,
    NATIVE432_C128_LAYER_IDS,
    NATIVE432_RECORD_BYTES,
    NATIVE432_ROLE_GEOMETRY,
    NATIVE432_SWA_LAYER_IDS,
    Native432LMCacheMPConnector,
    Native432RegistrationError,
    validate_native432_runtime_registration,
)


class _FakeStorage:
    def __init__(self, key: int):
        self._key = key

    def data_ptr(self) -> int:
        return self._key


class _FakeTensor:
    def __init__(
        self,
        shape,
        strides,
        *,
        offset: int = 0,
        storage_key: int | None = None,
        dtype: str = "torch.uint8",
    ):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self._offset = offset
        self._storage = _FakeStorage(storage_key if storage_key is not None else id(self))
        self.dtype = dtype

    def stride(self):
        return self._strides

    def storage_offset(self) -> int:
        return self._offset

    def untyped_storage(self) -> _FakeStorage:
        return self._storage

    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@dataclass
class _CfgTensor:
    size: int
    shared_by: list[str]
    offset: int = 0
    block_stride: int = 0


@dataclass
class _Group:
    layer_names: list[str]
    is_eagle_group: bool = False


@dataclass
class _Config:
    kv_cache_tensors: list[_CfgTensor] = field(default_factory=list)
    kv_cache_groups: list[_Group] = field(default_factory=list)


def _mla_name(layer_id: int) -> str:
    return f"model.layers.{layer_id}.attn"


def _swa_name(layer_id: int) -> str:
    return f"model.layers.{layer_id}.attn.swa_cache"


def _indexer_name(layer_id: int) -> str:
    return f"model.layers.{layer_id}.attn.indexer.k_cache"


def _native_tensor(role: str, *, offset: int, storage_key: int, blocks: int = 4):
    _, _, slots = NATIVE432_ROLE_GEOMETRY[role]
    dim0 = slots * NATIVE432_RECORD_BYTES * 2
    return _FakeTensor(
        (blocks, slots, NATIVE432_RECORD_BYTES),
        (dim0, NATIVE432_RECORD_BYTES, 1),
        offset=offset,
        storage_key=storage_key,
    )


def _valid_runtime():
    """Full mapped inventory with packed, non-overlapping layer views."""
    config = _Config()
    kv_caches: dict[str, _FakeTensor] = {}
    for role, layer_ids, name_fn in (
        ("c4", sorted(NATIVE432_C4_LAYER_IDS), _mla_name),
        ("c128", sorted(NATIVE432_C128_LAYER_IDS), _mla_name),
        ("swa", sorted(NATIVE432_SWA_LAYER_IDS), _swa_name),
    ):
        _, _, slots = NATIVE432_ROLE_GEOMETRY[role]
        page_bytes = slots * NATIVE432_RECORD_BYTES
        storage_key = hash(role) & 0xFFFF
        names = [name_fn(layer_id) for layer_id in layer_ids]
        for index, name in enumerate(names):
            offset = index * page_bytes
            kv_caches[name] = _native_tensor(role, offset=offset, storage_key=storage_key)
            block_stride = len(names) * page_bytes
            config.kv_cache_tensors.append(
                _CfgTensor(
                    size=kv_caches[name].shape[0] * block_stride,
                    shared_by=[name],
                    offset=offset,
                    block_stride=block_stride,
                )
            )
        config.kv_cache_groups.append(_Group(layer_names=names))
    # Excluded-role indexer caches share a group with SWA layers.
    indexer_names = []
    for layer_id in sorted(NATIVE432_SWA_LAYER_IDS):
        name = _indexer_name(layer_id)
        indexer_names.append(name)
        kv_caches[name] = _FakeTensor(
            (11084, 64, 132), (64 * 132, 132, 1), offset=0, dtype="torch.uint8"
        )
        config.kv_cache_tensors.append(
            _CfgTensor(size=kv_caches[name].numel(), shared_by=[name])
        )
    config.kv_cache_groups[2].layer_names.extend(indexer_names)
    # DSv4 dspark draft attention reads the target SWA caches, so the runtime
    # marks the SWA group as the eagle group. SWA tensors must still gate.
    config.kv_cache_groups[2].is_eagle_group = True
    # Compressor state caches ride in the MLA groups and must be excluded.
    for group, layer_ids in (
        (config.kv_cache_groups[0], sorted(NATIVE432_C4_LAYER_IDS)),
        (config.kv_cache_groups[1], sorted(NATIVE432_C128_LAYER_IDS)),
    ):
        for layer_id in layer_ids:
            name = f"model.layers.{layer_id}.attn.compressor.state_cache"
            group.layer_names.append(name)
            kv_caches[name] = _FakeTensor(
                (11074, 4, 2048), (4 * 2048, 2048, 1), offset=0, dtype="torch.uint8"
            )
            config.kv_cache_tensors.append(
                _CfgTensor(size=kv_caches[name].numel(), shared_by=[name])
            )
    return config, kv_caches


def test_valid_mapped_inventory_passes():
    config, kv_caches = _valid_runtime()
    validate_native432_runtime_registration(config, kv_caches)


def test_missing_config_fails():
    _, kv_caches = _valid_runtime()
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(None, kv_caches)


def test_missing_layer_tensor_fails():
    config, kv_caches = _valid_runtime()
    del kv_caches[_swa_name(sorted(NATIVE432_SWA_LAYER_IDS)[0])]
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_interleaved_group_views_on_shared_slab_pass():
    """Runtime layout: SWA and C4 views interleave in one packed slab."""
    config, kv_caches = _valid_runtime()
    swa_name = _swa_name(0)
    mla_name = _mla_name(2)
    shared_storage = 0xDEADBEEF
    block_stride = 860160
    swa_tensor = _FakeTensor(
        (11074, 64, NATIVE432_RECORD_BYTES),
        (block_stride, NATIVE432_RECORD_BYTES, 1),
        offset=0,
        storage_key=shared_storage,
    )
    mla_tensor = _FakeTensor(
        (11047, 64, NATIVE432_RECORD_BYTES),
        (block_stride, NATIVE432_RECORD_BYTES, 1),
        offset=8704,
        storage_key=shared_storage,
    )
    kv_caches[swa_name] = swa_tensor
    kv_caches[mla_name] = mla_tensor
    for name, tensor in ((swa_name, swa_tensor), (mla_name, mla_tensor)):
        cfg = next(cfg for cfg in config.kv_cache_tensors if cfg.shared_by == [name])
        cfg.offset = tensor.storage_offset()
        cfg.block_stride = block_stride
        cfg.size = tensor.shape[0] * block_stride
    validate_native432_runtime_registration(config, kv_caches)




def test_missing_layer_group_fails():
    config, kv_caches = _valid_runtime()
    swa_group = next(
        group
        for group in config.kv_cache_groups
        if _swa_name(0) in group.layer_names
    )
    dropped = _swa_name(0)
    swa_group.layer_names.remove(dropped)
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)
    swa_group.layer_names.append(dropped)


def test_wrong_dtype_fails():
    config, kv_caches = _valid_runtime()
    name = _mla_name(sorted(NATIVE432_C4_LAYER_IDS)[0])
    original = kv_caches[name]
    kv_caches[name] = _FakeTensor(
        original.shape,
        original.stride(),
        offset=original.storage_offset(),
        dtype="torch.float8_e4m3fn",
    )
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_wrong_slot_count_fails():
    config, kv_caches = _valid_runtime()
    name = _mla_name(sorted(NATIVE432_C128_LAYER_IDS)[0])
    kv_caches[name] = _FakeTensor(
        (4, 64, NATIVE432_RECORD_BYTES),
        (64 * NATIVE432_RECORD_BYTES, NATIVE432_RECORD_BYTES, 1),
        offset=0,
    )
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_inner_stride_mismatch_fails():
    config, kv_caches = _valid_runtime()
    name = _mla_name(sorted(NATIVE432_C4_LAYER_IDS)[0])
    kv_caches[name] = _FakeTensor(
        (4, 64, NATIVE432_RECORD_BYTES),
        (64 * NATIVE432_RECORD_BYTES, NATIVE432_RECORD_BYTES + 16, 1),
        offset=0,
    )
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_offset_disagreement_fails():
    config, kv_caches = _valid_runtime()
    name = _mla_name(sorted(NATIVE432_C4_LAYER_IDS)[0])
    cfg = next(cfg for cfg in config.kv_cache_tensors if cfg.shared_by == [name])
    cfg.offset += NATIVE432_RECORD_BYTES
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_packed_overlap_fails():
    config, kv_caches = _valid_runtime()
    names = [_mla_name(2), _mla_name(4)]
    victim = kv_caches[names[1]]
    kv_caches[names[1]] = _FakeTensor(
        victim.shape,
        victim.stride(),
        offset=0,
        storage_key=victim.untyped_storage().data_ptr(),
    )
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_shared_alias_view_is_not_overlap():
    config, kv_caches = _valid_runtime()
    owner = _mla_name(2)
    alias = "model.layers.0.attn.alias_view"
    owner_cfg = next(cfg for cfg in config.kv_cache_tensors if cfg.shared_by == [owner])
    owner_cfg.shared_by.append(alias)
    kv_caches[alias] = kv_caches[owner]
    config.kv_cache_groups.append(_Group(layer_names=[alias]))
    with pytest.raises(Native432RegistrationError):
        # Alias layer id 0 is not a C4 owner, so the name itself must fail.
        validate_native432_runtime_registration(config, kv_caches)


def test_native_shaped_compressor_tensor_fails():
    config, kv_caches = _valid_runtime()
    name = "model.layers.2.attn.compressor.state_cache"
    kv_caches[name] = _native_tensor("c4", offset=0, storage_key=777)
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_unmapped_native_shaped_tensor_fails():
    config, kv_caches = _valid_runtime()
    draft_name = "model.draft_model.layers.0.attn"
    kv_caches[draft_name] = _native_tensor("c4", offset=0, storage_key=424242)
    config.kv_cache_tensors.append(
        _CfgTensor(
            size=kv_caches[draft_name].numel(),
            shared_by=[draft_name],
            offset=0,
            block_stride=64 * NATIVE432_RECORD_BYTES,
        )
    )
    config.kv_cache_groups.append(_Group(layer_names=[draft_name], is_eagle_group=True))
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_native_shaped_indexer_tensor_fails():
    config, kv_caches = _valid_runtime()
    name = _indexer_name(2)
    kv_caches[name] = _native_tensor("c4", offset=0, storage_key=31337)
    with pytest.raises(Native432RegistrationError):
        validate_native432_runtime_registration(config, kv_caches)


def test_non_native_draft_group_is_ignored():
    config, kv_caches = _valid_runtime()
    draft_name = "model.draft_model.layers.0.attn"
    kv_caches[draft_name] = _FakeTensor(
        (4, 64, 656), (64 * 656, 656, 1), offset=0, dtype="torch.uint8"
    )
    config.kv_cache_tensors.append(
        _CfgTensor(size=kv_caches[draft_name].numel(), shared_by=[draft_name])
    )
    config.kv_cache_groups.append(_Group(layer_names=[draft_name], is_eagle_group=True))
    validate_native432_runtime_registration(config, kv_caches)


def test_connector_gate_runs_before_parent_registration(monkeypatch):
    calls: list[str] = []

    def _fake_parent_register(self, kv_caches):
        calls.append("parent")

    monkeypatch.setattr(LMCacheMPConnector, "register_kv_caches", _fake_parent_register)
    connector = object.__new__(Native432LMCacheMPConnector)
    connector._kv_cache_config = None
    with pytest.raises(Native432RegistrationError):
        connector.register_kv_caches({})
    assert calls == []


def test_connector_delegates_after_gate(monkeypatch):
    calls: list[str] = []

    def _fake_parent_register(self, kv_caches):
        calls.append("parent")

    monkeypatch.setattr(LMCacheMPConnector, "register_kv_caches", _fake_parent_register)
    config, kv_caches = _valid_runtime()
    connector = object.__new__(Native432LMCacheMPConnector)
    connector._kv_cache_config = config
    connector.register_kv_caches(kv_caches)
    assert calls == ["parent"]
