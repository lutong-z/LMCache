# SPDX-License-Identifier: Apache-2.0
"""Fail-closed native NVFP4 DS-MLA (native432) LMCache MP connector.

The DeepSeek V4 native runtime stores compressed MLA records as 432-byte
rows (256 B e2m1 nibbles + 32 B e4m3 group scales + 16 B zero padding +
128 B bf16 RoPE) in packed ``uint8`` tensors. Shape alone cannot distinguish
a native432 tensor from any other byte buffer, so this connector refuses
registration unless vLLM's ``KVCacheConfig`` and the registered tensors
together prove the pinned runtime mapping:

- C4 layers 2..42 (even), 4:1 compression, 64 storage slots per 256-token page
- C128 layers 3..41 (odd), 128:1 compression, 2 storage slots per 256-token page
- SWA layers 0..42 (``.swa_cache`` views), 64 slots per 64-token window page

Classification is per tensor name, never per group: the runtime mixes MLA,
SWA, indexer, and draft caches inside kv cache groups. Indexer, draft, and
MTP caches are excluded roles; they are ignored unless their tensors look
like unmapped native432 records. Anything missing, duplicated, ambiguous,
stale, or overlapping raises ``Native432RegistrationError`` before the
LMCache server handshake.
"""

# Standard
import re
from typing import Any

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

NATIVE432_ABI_IDENTITY = "nvfp4_ds_mla:bf16-rope-432:identity-v1"
NATIVE432_RECORD_BYTES = 432

NATIVE432_C4_LAYER_IDS = frozenset(range(2, 43, 2))
NATIVE432_C128_LAYER_IDS = frozenset(range(3, 43, 2))
NATIVE432_SWA_LAYER_IDS = frozenset(range(43))

# role -> (compression_ratio, logical block size, storage slots per page)
NATIVE432_ROLE_GEOMETRY = {
    "c4": (4, 256, 64),
    "c128": (128, 256, 2),
    "swa": (1, 64, 64),
}

_LAYER_ID_RE = re.compile(r"layers\.(\d+)\.")


class Native432RegistrationError(ValueError):
    """Raised when a runtime cache cannot be proven to satisfy native432."""


def _fail(reason: str) -> None:
    raise Native432RegistrationError(f"native432 registration refused: {reason}")


def _layer_id(name: str) -> int | None:
    match = _LAYER_ID_RE.search(name)
    return int(match.group(1)) if match else None


def _is_excluded_name(name: str) -> bool:
    """Indexer, draft, and MTP caches are never native432 MLA pages."""
    lowered = name.lower()
    return ".indexer." in lowered or "draft" in lowered or "mtp" in lowered


def _role_for_name(name: str) -> str | None:
    """Map one tensor name to its native432 role, or None when unmappable."""
    if name.endswith(".swa_cache"):
        layer_id = _layer_id(name)
        if layer_id in NATIVE432_SWA_LAYER_IDS:
            return "swa"
        return None
    layer_id = _layer_id(name)
    if layer_id in NATIVE432_C4_LAYER_IDS:
        return "c4"
    if layer_id in NATIVE432_C128_LAYER_IDS:
        return "c128"
    return None


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        _fail("registered tensor has no shape")
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        _fail(f"registered tensor shape is not integral: {shape!r} ({exc})")


def _tensor_strides(tensor: Any) -> tuple[int, ...]:
    stride_fn = getattr(tensor, "stride", None)
    if not callable(stride_fn):
        _fail("registered tensor has no stride()")
    try:
        return tuple(int(dim) for dim in stride_fn())
    except (TypeError, ValueError) as exc:
        _fail(f"registered tensor strides are not integral ({exc})")


def _tensor_storage_offset(tensor: Any) -> int:
    offset_fn = getattr(tensor, "storage_offset", None)
    if not callable(offset_fn):
        _fail("registered tensor has no storage_offset()")
    return int(offset_fn())


def _tensor_storage_key(tensor: Any) -> Any:
    storage_fn = getattr(tensor, "untyped_storage", None)
    if not callable(storage_fn):
        _fail("registered tensor has no untyped_storage()")
    storage = storage_fn()
    data_ptr = getattr(storage, "data_ptr", None)
    return data_ptr() if callable(data_ptr) else id(storage)


def _tensor_is_uint8(tensor: Any) -> bool:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return False
    return "uint8" in str(dtype).lower()


def _looks_native432(tensor: Any) -> bool:
    """Heuristic used only to reject unmapped native-shaped tensors."""
    try:
        shape = _tensor_shape(tensor)
    except Native432RegistrationError:
        return False
    return (
        len(shape) == 3
        and shape[-1] == NATIVE432_RECORD_BYTES
        and _tensor_is_uint8(tensor)
    )


def _tensor_config_by_alias(kv_cache_config: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for tensor_cfg in getattr(kv_cache_config, "kv_cache_tensors", ()) or ():
        shared_by = getattr(tensor_cfg, "shared_by", None)
        if not shared_by:
            _fail("kv_cache_tensor entry has no shared_by aliases")
        for alias in shared_by:
            if alias in result:
                _fail(f"kv_cache_tensor alias {alias!r} is mapped twice")
            result[alias] = tensor_cfg
    return result


def _validate_native_tensor(
    name: str,
    tensor: Any,
    tensor_cfg: Any,
    *,
    role: str,
) -> tuple[Any, int, int]:
    """Validate one native432 tensor view; return (storage, offset, end)."""
    _, _, slots = NATIVE432_ROLE_GEOMETRY[role]
    shape = _tensor_shape(tensor)
    if len(shape) != 3:
        _fail(f"{role} tensor {name!r} must be 3-D [N,slots,432], got {shape!r}")
    if shape[1] != slots or shape[2] != NATIVE432_RECORD_BYTES:
        _fail(
            f"{role} tensor {name!r} requires [N,{slots},432], got {shape!r}"
        )
    if not _tensor_is_uint8(tensor):
        _fail(f"{role} tensor {name!r} must be uint8, got {tensor.dtype!r}")
    strides = _tensor_strides(tensor)
    if len(strides) != 3 or strides[2] != 1 or strides[1] != NATIVE432_RECORD_BYTES:
        _fail(
            f"{role} tensor {name!r} inner element strides must be "
            f"[...,432,1], got {strides!r}"
        )
    if strides[0] <= 0:
        _fail(f"{role} tensor {name!r} dim-0 stride must be positive")
    if shape[0] <= 0:
        _fail(f"{role} tensor {name!r} must own at least one block")

    offset = _tensor_storage_offset(tensor)
    cfg_offset = getattr(tensor_cfg, "offset", None)
    if cfg_offset is None:
        _fail(f"{role} tensor {name!r} is missing a stamped offset")
    if int(cfg_offset) != offset:
        _fail(
            f"{role} tensor {name!r} storage_offset {offset} disagrees with "
            f"stamped offset {cfg_offset}"
        )
    block_stride = getattr(tensor_cfg, "block_stride", 0) or 0
    page_bytes = slots * NATIVE432_RECORD_BYTES
    if block_stride > 0 and offset + page_bytes > block_stride:
        _fail(
            f"{role} tensor {name!r} page [{offset},{offset + page_bytes}) "
            f"exceeds packed block_stride {block_stride}"
        )
    size = getattr(tensor_cfg, "size", None)
    if size is not None:
        numel_fn = getattr(tensor, "numel", None)
        if callable(numel_fn) and int(numel_fn()) != int(size):
            _fail(
                f"{role} tensor {name!r} numel {int(numel_fn())} disagrees "
                f"with stamped size {size}"
            )
    return _tensor_storage_key(tensor), offset, offset + page_bytes


def validate_native432_runtime_registration(
    kv_cache_config: Any,
    kv_caches: dict[str, Any],
) -> None:
    """Prove the registered caches satisfy the pinned native432 mapping.

    Fail closed on any missing, duplicated, ambiguous, or overlapping
    native432 view. Indexer, draft, and MTP caches are ignored unless their
    tensors look like unmapped native432 records.
    """
    if kv_cache_config is None:
        _fail("kv_cache_config is unavailable")
    if not isinstance(kv_caches, dict) or not kv_caches:
        _fail("no kv caches were registered")

    groups = list(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    if not groups:
        _fail("kv_cache_config has no kv_cache_groups")

    tensor_cfgs = _tensor_config_by_alias(kv_cache_config)
    seen_aliases: set[str] = set()
    validated_cfgs: set[int] = set()
    role_layers: dict[str, set[int]] = {"c4": set(), "c128": set(), "swa": set()}
    # (storage key, dim0 stride) -> sorted list of [offset, end) intervals
    packed_intervals: dict[tuple[Any, int], list[tuple[int, int]]] = {}

    for group in groups:
        is_eagle_group = bool(getattr(group, "is_eagle_group", False))
        for name in list(getattr(group, "layer_names", ()) or ()):
            if not isinstance(name, str):
                _fail(f"kv cache group layer name is not a string: {name!r}")
            if is_eagle_group or _is_excluded_name(name):
                tensor = kv_caches.get(name)
                if tensor is not None and _looks_native432(tensor):
                    _fail(
                        f"excluded-role tensor {name!r} matches native432 "
                        f"geometry"
                    )
                continue

            role = _role_for_name(name)
            if role is None:
                tensor = kv_caches.get(name)
                if tensor is not None and _looks_native432(tensor):
                    _fail(
                        f"tensor {name!r} matches native432 geometry but is "
                        f"outside the pinned mapping"
                    )
                continue
            layer_id = _layer_id(name)
            assert layer_id is not None  # guaranteed by _role_for_name

            if name in seen_aliases:
                _fail(f"layer {name!r} appears in more than one group")
            seen_aliases.add(name)
            tensor = kv_caches.get(name)
            if tensor is None:
                _fail(f"{role} layer {name!r} has no registered tensor")
            tensor_cfg = tensor_cfgs.get(name)
            if tensor_cfg is None:
                _fail(f"{role} layer {name!r} has no kv_cache_tensor entry")
            role_layers[role].add(layer_id)

            # Aliases of one kv_cache_tensor share the same view; validate
            # the underlying view once so sharing is not read as overlap.
            cfg_key = id(tensor_cfg)
            if cfg_key in validated_cfgs:
                continue
            validated_cfgs.add(cfg_key)
            storage_key, offset, end = _validate_native_tensor(
                name, tensor, tensor_cfg, role=role
            )
            strides = _tensor_strides(tensor)
            packed_intervals.setdefault((storage_key, strides[0]), []).append(
                (offset, end)
            )

    for role, expected in (
        ("c4", NATIVE432_C4_LAYER_IDS),
        ("c128", NATIVE432_C128_LAYER_IDS),
        ("swa", NATIVE432_SWA_LAYER_IDS),
    ):
        actual = role_layers[role]
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            _fail(
                f"{role} layer inventory mismatch; missing={missing!r} "
                f"extra={extra!r}"
            )

    for (storage_key, _), intervals in packed_intervals.items():
        intervals.sort()
        previous_end: int | None = None
        for start, end in intervals:
            if previous_end is not None and start < previous_end:
                _fail(
                    f"native432 packed views overlap on storage {storage_key!r}: "
                    f"[{start},{end}) inside [{intervals[0][0]},{previous_end})"
                )
            previous_end = end


class Native432LMCacheMPConnector(LMCacheMPConnector):
    """LMCache MP connector gated on the native432 runtime mapping."""

    def register_kv_caches(self, kv_caches: dict[str, Any]):
        validate_native432_runtime_registration(
            getattr(self, "_kv_cache_config", None), kv_caches
        )
        return super().register_kv_caches(kv_caches)
