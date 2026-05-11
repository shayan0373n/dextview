from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

SUPPORTED_SAMPLE_RATES = (512, 2048, 5120, 10240)
NCH_BITS_TO_NUM_CHANNELS = {0: 120, 1: 216, 2: 312, 3: 408}

_FSAMP_BITS = {512: 0, 2048: 8, 5120: 16, 10240: 24}
_NCH_TO_BITS = {0: 0, 1: 2, 2: 4, 3: 6}
COMMAND_LENGTH = 40

INPUT_BLOCK_NAMES = (
    "IN1",
    "IN2",
    "IN3",
    "IN4",
    "IN5",
    "IN6",
    "IN7",
    "IN8",
    "MULTIPLE_IN1",
    "MULTIPLE_IN2",
    "MULTIPLE_IN3",
    "MULTIPLE_IN4",
)
INPUT_BLOCK_INDEX = {
    name.upper().replace(" ", "_"): idx for idx, name in enumerate(INPUT_BLOCK_NAMES)
}

DEFAULT_FORCE_CHANNELS = tuple(range(10))
DEFAULT_CONF2_BYTE = 0b00010100
DEFAULT_INPUT_CONF2_BYTES = tuple(
    DEFAULT_CONF2_BYTE for _ in range(len(INPUT_BLOCK_NAMES))
)

SIDE_NAME_TO_BITS = {
    "not_defined": 0,
    "undefined": 0,
    "left": 1,
    "right": 2,
    "none": 3,
}
MODE_NAME_TO_BITS = {
    "monopolar": 0,
    "differential": 1,
    "bipolar": 2,
}

HPF_HZ_TO_BITS = {
    0.7: 0,
    10.0: 1,
    100.0: 2,
    200.0: 3,
}
LPF_HZ_TO_BITS = {
    130.0: 0,
    500.0: 1,
    900.0: 2,
    4400.0: 3,
}


def smallest_nch_for_channel_count(min_channels: int) -> int:
    """Return the smallest nch bit-code whose channel count >= min_channels."""
    if min_channels <= 0:
        raise ValueError(f"min_channels must be positive, got {min_channels}")
    for nch_code in sorted(NCH_BITS_TO_NUM_CHANNELS):
        if NCH_BITS_TO_NUM_CHANNELS[nch_code] >= min_channels:
            return nch_code
    raise ValueError(
        f"No nch configuration supports {min_channels} channels "
        f"(max is {max(NCH_BITS_TO_NUM_CHANNELS.values())})"
    )


def _crc8(values: Sequence[int], length: int) -> int:
    crc = 0
    index = 0
    remaining = length

    while remaining > 0:
        extract = values[index]
        for _ in range(8, 0, -1):
            xor_sum = (crc % 2) ^ (extract % 2)
            crc //= 2

            if xor_sum > 0:
                crc ^= 140

            extract //= 2

        remaining -= 1
        index += 1

    return crc


def build_start_command(
    *,
    decimation_enabled: bool,
    rec_on: bool,
    fsamp: int,
    nch: int,
    input_conf2_bytes: tuple[int, ...],
) -> bytes:
    """Encode a 40-byte Quattrocento start-acquisition command frame."""
    command = [0] * COMMAND_LENGTH

    acq_sett = (
        0b10000000
        + (0b01000000 if decimation_enabled else 0)
        + (0b00100000 if rec_on else 0)
        + _FSAMP_BITS[fsamp]
        + _NCH_TO_BITS[nch]
        + 1
    )
    command[0] = acq_sett
    command[1] = 9
    command[2] = 0

    for input_idx, base in enumerate(range(3, COMMAND_LENGTH - 1, 3)):
        command[base] = 0
        command[base + 1] = 0
        command[base + 2] = input_conf2_bytes[input_idx]

    command[-1] = _crc8(command, COMMAND_LENGTH - 1)
    return bytes(command)


@dataclass(frozen=True, slots=True)
class StartCommand:
    """Decoded fields from a 40-byte Quattrocento start-acquisition frame."""

    fsamp_hz: int
    nch_code: int
    decimation_enabled: bool
    rec_on: bool
    input_conf2_bytes: tuple[int, ...]


_FSAMP_BITS_REV: dict[int, int] = {v: k for k, v in _FSAMP_BITS.items()}
_NCH_BITS_REV: dict[int, int] = {v: k for k, v in _NCH_TO_BITS.items()}


def parse_start_command(frame: bytes) -> StartCommand:
    """Decode a 40-byte Quattrocento start-acquisition frame.

    Raises ValueError if the frame length is wrong or the CRC does not match.
    """
    if len(frame) != COMMAND_LENGTH:
        raise ValueError(f"Expected {COMMAND_LENGTH} bytes, got {len(frame)}")
    expected_crc = _crc8(frame, COMMAND_LENGTH - 1)
    if frame[-1] != expected_crc:
        raise ValueError(
            f"CRC mismatch: expected {expected_crc:#04x}, got {frame[-1]:#04x}"
        )

    byte0 = frame[0]
    decimation_enabled = bool(byte0 & 0b01000000)
    rec_on = bool(byte0 & 0b00100000)
    fsamp_raw = byte0 & 0b00011000
    nch_raw = byte0 & 0b00000110

    if fsamp_raw not in _FSAMP_BITS_REV:
        raise ValueError(f"Unknown fsamp bits: {fsamp_raw:#010b}")
    if nch_raw not in _NCH_BITS_REV:
        raise ValueError(f"Unknown nch bits: {nch_raw:#010b}")

    input_conf2_bytes = tuple(
        frame[base + 2] for base in range(3, COMMAND_LENGTH - 1, 3)
    )
    return StartCommand(
        fsamp_hz=_FSAMP_BITS_REV[fsamp_raw],
        nch_code=_NCH_BITS_REV[nch_raw],
        decimation_enabled=decimation_enabled,
        rec_on=rec_on,
        input_conf2_bytes=input_conf2_bytes,
    )


def build_stop_command() -> bytes:
    """Encode a 40-byte Quattrocento stop-acquisition command frame."""
    command = [0] * COMMAND_LENGTH
    command[0] = 0b10000000
    command[-1] = _crc8(command, COMMAND_LENGTH - 1)
    return bytes(command)

