import logging
import socket
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .protocol import SUPPORTED_SAMPLE_RATES

logger = logging.getLogger("dextview.rebroadcast_detect")

POSSIBLE_CHANNEL_COUNTS = tuple(range(8, 409))
DEFAULT_N_CHANNELS = 408
DEFAULT_SAMPLING_RATE = 2048
BYTES_PER_SAMPLE = 2
RAMP_CONSISTENCY_THRESHOLD = 0.95
RAMP_COLUMN = -8


@dataclass(frozen=True, slots=True)
class DetectedParams:
    """Result of probing a rebroadcast server for stream parameters."""

    n_channels: int
    sampling_rate_hz: int


def detect_stream_params(
    host: str,
    port: int,
    *,
    detect_nch: bool,
    detect_rate: bool,
    known_nch: int | None = None,
    known_rate: int | None = None,
) -> DetectedParams:
    """Probe the rebroadcast server for n_channels and/or sampling_rate.

    When detection is skipped, the caller-supplied known value is used.
    Detection runs a single sacrificial connection and analyzes the ramp signal.
    """
    if not detect_nch and not detect_rate:
        return DetectedParams(
            n_channels=known_nch or DEFAULT_N_CHANNELS,
            sampling_rate_hz=known_rate or DEFAULT_SAMPLING_RATE,
        )

    logger.info(f"Probing {host}:{port} for stream parameters...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    try:
        sock.sendall(b"startTX")
        sock.recv(8)  # Burn the 8-byte header

        # 1. Detect n_channels if needed.
        if detect_nch:
            # Read enough to reasonably detect the channel count (200KB).
            raw_bytes = b""
            while len(raw_bytes) < 200000:
                packet = sock.recv(4096)
                if not packet:
                    break
                raw_bytes += packet
            
            raw_data = np.frombuffer(raw_bytes, dtype="<i2")
            n_channels = _detect_n_channels_from_data(raw_data)
        else:
            assert known_nch is not None
            n_channels = known_nch

        # 2. Detect sampling rate if needed.
        if detect_rate:
            # We already have the socket open and streaming.
            # Read a fixed number of samples and measure the time.
            target_samples = 2048
            target_bytes = target_samples * n_channels * BYTES_PER_SAMPLE
            received_bytes = 0

            start_time = time.perf_counter()
            while received_bytes < target_bytes:
                packet = sock.recv(min(4096, target_bytes - received_bytes))
                if not packet:
                    break
                received_bytes += len(packet)

            actual_duration = time.perf_counter() - start_time
            sampling_rate_hz = _resolve_sampling_rate(target_samples, actual_duration)
        else:
            assert known_rate is not None
            sampling_rate_hz = known_rate

    finally:
        sock.sendall(b"stopTX")
        sock.close()

    return DetectedParams(n_channels=n_channels, sampling_rate_hz=sampling_rate_hz)


def _detect_n_channels_from_data(raw_data: NDArray) -> int:
    """Detect n_channels from a raw data buffer by analyzing the ramp signal."""
    for n_channels in POSSIBLE_CHANNEL_COUNTS:
        usable_samples = (len(raw_data) // n_channels) * n_channels
        if usable_samples == 0:
            continue

        reshaped_data = raw_data[:usable_samples].reshape(-1, n_channels)
        ramp_column = reshaped_data[:, RAMP_COLUMN]

        consistency_ratio = _ramp_consistency_ratio(ramp_column)
        ramp_step = _get_ramp_step(ramp_column)

        if consistency_ratio > RAMP_CONSISTENCY_THRESHOLD and ramp_step is not None:
            logger.info(f"Auto-detected n_channels: {n_channels}")
            return n_channels

    logger.warning(
        f"Could not detect channels, falling back to {DEFAULT_N_CHANNELS}"
    )
    return DEFAULT_N_CHANNELS


def _resolve_sampling_rate(target_samples: int, actual_duration: float) -> int:
    """Matches measured rate against the nearest supported Quattrocento rate."""
    if actual_duration <= 0:
        return DEFAULT_SAMPLING_RATE

    measured_rate = target_samples / actual_duration
    logger.info(f"Measured sampling rate: {measured_rate:.1f} Hz")

    best_rate = min(SUPPORTED_SAMPLE_RATES, key=lambda r: abs(r - measured_rate))
    distance = abs(best_rate - measured_rate)
    logger.info(f"Closest match: {best_rate} Hz (distance: {distance:.1f} Hz)")

    return best_rate


def _ramp_consistency_ratio(signal_column: NDArray) -> float:
    """Calculate how consistent the ramp steps are."""
    if len(signal_column) < 10:
        return 0.0
    diffs = np.diff(signal_column)
    _, counts = np.unique(diffs, return_counts=True)
    if len(counts) == 0:
        return 0.0
    return float(np.max(counts) / len(diffs))


def _get_ramp_step(signal_column: NDArray) -> int | None:
    """Extract the most frequent step size from the ramp column."""
    if len(signal_column) < 10:
        return None
    diffs = np.diff(signal_column)
    values, counts = np.unique(diffs, return_counts=True)
    if len(counts) == 0:
        return None
    most_common_idx = int(np.argmax(counts))
    step = int(values[most_common_idx])
    return step if step > 0 else None
