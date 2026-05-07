from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .protocol import (
    DEFAULT_CONF2_BYTE,
    HPF_HZ_TO_BITS,
    INPUT_BLOCK_INDEX,
    INPUT_BLOCK_NAMES,
    LPF_HZ_TO_BITS,
    MODE_NAME_TO_BITS,
    SIDE_NAME_TO_BITS,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_lookup_or_bits(
    value: Any, *, field_name: str, name_to_bits: Mapping[str, int], allow_bit_3: bool = True
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a string or integer")

    if isinstance(value, int):
        upper = 3 if allow_bit_3 else 2
        if 0 <= value <= upper:
            return value

    token = _normalize_token(value)
    if token in name_to_bits:
        return name_to_bits[token]

    raise ValueError(f"Unsupported {field_name} value: {value!r}")


def _parse_filter_bits(value: Any, *, field_name: str, hz_to_bits: Mapping[float, int]) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a frequency or bit value")

    if isinstance(value, int) and 0 <= value <= 3:
        return value

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {field_name} value: {value!r}") from exc

    for hz, bits in hz_to_bits.items():
        if abs(numeric - hz) < 1e-9:
            return bits
    raise ValueError(f"Unsupported {field_name} value: {value!r}")


def _parse_conf2_block(
    raw_block: Mapping[str, Any] | None, *, default_byte: int, block_name: str
) -> int:
    if raw_block is None:
        return default_byte
    if not isinstance(raw_block, Mapping):
        raise ValueError(f"{block_name} must be a TOML table")

    side_bits = (default_byte >> 6) & 0b11
    hpf_bits = (default_byte >> 4) & 0b11
    lpf_bits = (default_byte >> 2) & 0b11
    mode_bits = default_byte & 0b11

    allowed = {"side", "hpf", "lpf", "mode"}
    unknown_fields = set(raw_block.keys()) - allowed
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"{block_name} has unknown field(s): {unknown}")

    if "side" in raw_block:
        side_bits = _parse_lookup_or_bits(
            raw_block["side"],
            field_name=f"{block_name}.side",
            name_to_bits=SIDE_NAME_TO_BITS,
        )
    if "hpf" in raw_block:
        hpf_bits = _parse_filter_bits(
            raw_block["hpf"],
            field_name=f"{block_name}.hpf",
            hz_to_bits=HPF_HZ_TO_BITS,
        )
    if "lpf" in raw_block:
        lpf_bits = _parse_filter_bits(
            raw_block["lpf"],
            field_name=f"{block_name}.lpf",
            hz_to_bits=LPF_HZ_TO_BITS,
        )
    if "mode" in raw_block:
        mode_bits = _parse_lookup_or_bits(
            raw_block["mode"],
            field_name=f"{block_name}.mode",
            name_to_bits=MODE_NAME_TO_BITS,
        )

    return (side_bits << 6) | (hpf_bits << 4) | (lpf_bits << 2) | mode_bits


def _parse_conf2_overrides(
    raw_overrides: Mapping[str, Any] | None, *, default_byte: int
) -> tuple[int, ...]:
    conf2_bytes = [default_byte for _ in range(len(INPUT_BLOCK_NAMES))]
    if raw_overrides is None:
        return tuple(conf2_bytes)
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("conf2_overrides must be a TOML table")

    for raw_name, raw_block in raw_overrides.items():
        name_token = str(raw_name).upper().replace(" ", "_")
        block_idx = INPUT_BLOCK_INDEX.get(name_token)
        if block_idx is None:
            raise ValueError(
                f"Unknown conf2_overrides key {raw_name!r}. "
                f"Expected one of: {', '.join(INPUT_BLOCK_NAMES)}"
            )
        conf2_bytes[block_idx] = _parse_conf2_block(
            raw_block,
            default_byte=default_byte,
            block_name=f"conf2_overrides.{raw_name}",
        )
    return tuple(conf2_bytes)


def load_input_conf2_bytes(path: str | Path) -> tuple[int, ...]:
    """Load per-input-block conf2 bytes from a TOML file.

    The file may contain `[conf2_defaults]` (side/hpf/lpf/mode applied to all
    blocks) and `[conf2_overrides.<BLOCK_NAME>]` for per-block overrides.
    Unknown top-level keys are rejected.
    """
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    allowed = {"conf2_defaults", "conf2_overrides"}
    unknown_fields = set(payload.keys()) - allowed
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown conf2 config field(s): {unknown}")

    default_byte = _parse_conf2_block(
        payload.get("conf2_defaults"),
        default_byte=DEFAULT_CONF2_BYTE,
        block_name="conf2_defaults",
    )
    return _parse_conf2_overrides(
        payload.get("conf2_overrides"),
        default_byte=default_byte,
    )


