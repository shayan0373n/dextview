from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .config import QuattrocentoConfig
from .models import CapturedWindow, DataBatch

logger = logging.getLogger("quattrocento.processing")


def aggregate_finger_forces(
    sensor_forces: NDArray[np.float64], finger_sensor_map: Mapping[str, int]
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Map sensor forces into ordered finger-force series."""
    finger_labels = tuple(finger_sensor_map.keys())
    sensor_indices = [finger_sensor_map[name] for name in finger_labels]
    return sensor_forces[:, sensor_indices], finger_labels


class TriggerWindowProcessor:
    """Detect rising AUX-in edges and collect fixed post-trigger windows."""

    def __init__(self, config: QuattrocentoConfig) -> None:
        """Configure trigger threshold and capture window length."""
        self._post_samples = config.post_trigger_samples
        self._pre_samples = config.pre_trigger_samples
        self._total_samples = config.total_window_samples
        self._trigger_threshold = config.trigger_threshold
        self._sample_rate_hz = config.sample_rate_hz
        self._finger_sensor_map = config.finger_sensor_map
        self._sensor_count = config.sensor_count

        # Adaptive trigger model on AUX signal:
        # - baseline: slow EMA to follow DC drift (e.g., ~8000 counts offset)
        # - noise: EMA of |aux-baseline| to estimate noise floor
        # Trigger becomes HIGH when (aux-baseline) exceeds max(static_threshold, k*noise).
        baseline_tau_seconds = 2.0
        noise_tau_seconds = 1.0
        self._baseline_alpha = 1.0 - np.exp(
            -1.0 / (self._sample_rate_hz * baseline_tau_seconds)
        )
        self._noise_alpha = 1.0 - np.exp(
            -1.0 / (self._sample_rate_hz * noise_tau_seconds)
        )
        self._noise_scale = 4.0
        self._baseline_estimate: float | None = None
        self._noise_estimate = 0.0

        self._capturing = False
        self._previous_trigger_high = False
        self._write_pos = 0
        self._capture_target = self._total_samples
        self._trigger_index_in_buffer = 0
        self._time_buffer = np.empty(self._total_samples, dtype=np.float64)
        self._force_buffer = np.empty(
            (self._total_samples, self._sensor_count), dtype=np.float64
        )

        # Pre-trigger ring buffer: holds the most recent _pre_samples samples
        # at all times so a rising edge can prepend pre-roll to the capture.
        # Updated unconditionally on every batch, regardless of capture state.
        self._ring_time = np.empty(self._pre_samples, dtype=np.float64)
        self._ring_force = np.empty(
            (self._pre_samples, self._sensor_count), dtype=np.float64
        )
        self._ring_pos = 0
        self._ring_filled = 0  # how many slots are populated (caps at _pre_samples)

    @property
    def is_capturing(self) -> bool:
        """Whether a post-trigger capture is currently in progress."""
        return self._capturing

    @property
    def is_trigger_high(self) -> bool:
        """Whether the adaptive trigger model currently considers the signal high."""
        return self._previous_trigger_high

    def reset(self) -> None:
        """Clear internal state and drop partially collected data."""
        self._capturing = False
        self._previous_trigger_high = False
        self._baseline_estimate = None
        self._noise_estimate = 0.0
        self._write_pos = 0
        self._ring_pos = 0
        self._ring_filled = 0

    def process_batch(self, batch: DataBatch) -> list[CapturedWindow]:
        """Update triggers and return windows for any completed captures.

        Scans for rising edges on the AUX channel. If not already capturing,
        a new window begins at the first edge. If a capture completes mid-batch,
        we continue scanning the remainder of the batch for new triggers.
        """
        windows: list[CapturedWindow] = []
        batch_size = batch.timestamps.shape[0]
        if batch_size == 0:
            return windows

        self._advance_baseline(batch.aux_in)

        cursor = 0
        while cursor < batch_size:
            if not self._capturing:
                edges, last_high = self._find_rising_edges(
                    batch.aux_in[cursor:], self._previous_trigger_high
                )
                if edges.size > 0:
                    trigger_idx_in_batch = cursor + int(edges[0])
                    self._previous_trigger_high = True
                    self._begin_capture_with_preroll(batch, trigger_idx_in_batch)
                    cursor = trigger_idx_in_batch
                else:
                    self._previous_trigger_high = last_high
                    cursor = batch_size

            if self._capturing:
                remaining_in_capture = self._capture_target - self._write_pos
                remaining_in_batch = batch_size - cursor
                count = min(remaining_in_batch, remaining_in_capture)

                window = self._collect_range(batch, cursor, cursor + count)
                if window is not None:
                    windows.append(window)

                cursor += count

        # Update the ring buffer for the entire batch.
        self._update_ring(batch.timestamps, batch.forces, 0, batch_size)
        return windows

    def _collect_range(
        self, batch: DataBatch, start: int, end: int
    ) -> CapturedWindow | None:
        """Copy samples from batch[start:end] into the capture buffer."""
        count = end - start
        wp = self._write_pos
        self._time_buffer[wp : wp + count] = batch.timestamps[start:end]
        self._force_buffer[wp : wp + count, :] = batch.forces[start:end, :]
        self._write_pos += count

        if self._write_pos >= self._capture_target:
            return self._complete_capture()
        return None

    def _update_ring(
        self,
        timestamps: NDArray[np.float64],
        forces: NDArray[np.float64],
        start: int,
        end: int,
    ) -> None:
        """Append samples [start:end] to the pre-trigger ring buffer."""
        if self._pre_samples == 0:
            return
        n = end - start
        if n <= 0:
            return
        # If the slice is larger than the ring, only the last _pre_samples matter.
        if n >= self._pre_samples:
            src_start = end - self._pre_samples
            self._ring_time[:] = timestamps[src_start:end]
            self._ring_force[:, :] = forces[src_start:end, :]
            self._ring_pos = 0
            self._ring_filled = self._pre_samples
            return
        # Otherwise wrap-write into the ring at _ring_pos.
        first = min(n, self._pre_samples - self._ring_pos)
        self._ring_time[self._ring_pos : self._ring_pos + first] = timestamps[
            start : start + first
        ]
        self._ring_force[self._ring_pos : self._ring_pos + first, :] = forces[
            start : start + first, :
        ]
        rest = n - first
        if rest > 0:
            self._ring_time[:rest] = timestamps[start + first : end]
            self._ring_force[:rest, :] = forces[start + first : end, :]
        self._ring_pos = (self._ring_pos + n) % self._pre_samples
        self._ring_filled = min(self._pre_samples, self._ring_filled + n)

    def _read_ring_chronological(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ring contents in oldest-to-newest order."""
        if self._ring_filled == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, self._sensor_count), dtype=np.float64),
            )
        if self._ring_filled < self._pre_samples:
            # Ring not yet full: data lives in [0:_ring_pos] in chronological order.
            return (
                self._ring_time[: self._ring_pos].copy(),
                self._ring_force[: self._ring_pos, :].copy(),
            )
        # Full ring: oldest is at _ring_pos, wrapping back around.
        rp = self._ring_pos
        time_out = np.concatenate((self._ring_time[rp:], self._ring_time[:rp]))
        force_out = np.concatenate((self._ring_force[rp:, :], self._ring_force[:rp, :]))
        return time_out, force_out

    def _begin_capture_with_preroll(self, batch: DataBatch, edge_idx: int) -> None:
        """Start a capture and seed the buffer with up to _pre_samples of pre-roll.

        If the ring is not yet full at startup, the pre-roll is shorter than
        _pre_samples — the capture target shrinks accordingly so completion
        still fires and the emitted window is just truncated on the left.
        """
        self._capturing = True
        self._write_pos = 0

        if self._pre_samples == 0:
            self._trigger_index_in_buffer = 0
            self._capture_target = self._total_samples
            return

        ring_time, ring_force = self._read_ring_chronological()
        # Pre-roll source: ring (samples before this batch) + batch[0:edge_idx].
        # Take the last _pre_samples of that combined sequence.
        pre_time = np.concatenate((ring_time, batch.timestamps[:edge_idx]))
        pre_force = np.concatenate((ring_force, batch.forces[:edge_idx, :]))
        if pre_time.shape[0] > self._pre_samples:
            pre_time = pre_time[-self._pre_samples :]
            pre_force = pre_force[-self._pre_samples :, :]
        n = pre_time.shape[0]
        self._time_buffer[:n] = pre_time
        self._force_buffer[:n, :] = pre_force
        self._write_pos = n
        self._trigger_index_in_buffer = n
        self._capture_target = n + self._post_samples

    def _advance_baseline(self, aux_in: NDArray[np.float64]) -> None:
        """Update baseline/noise EMA, skipping samples above the rejection threshold.

        Samples where abs(residual) >= threshold are not folded into the EMA — baseline
        and noise are frozen for the duration of a trigger-high period. This prevents the
        pulse from inflating the baseline but means DC drift is not tracked while the
        signal is high.
        """
        baseline = self._baseline_estimate
        noise = self._noise_estimate

        for sample in aux_in:
            sample_value = float(sample)
            if baseline is None:
                baseline = sample_value
                continue
            residual = sample_value - baseline
            threshold = max(self._trigger_threshold, self._noise_scale * noise)
            if abs(residual) < threshold:
                noise += self._noise_alpha * (abs(residual) - noise)
                baseline += self._baseline_alpha * float(
                    np.clip(residual, -threshold, threshold)
                )

        self._baseline_estimate = baseline
        self._noise_estimate = noise

    def _find_rising_edges(
        self, aux_in: NDArray[np.float64], prev_high: bool
    ) -> tuple[NDArray[np.intp], bool]:
        """Threshold aux against current baseline and return (rising_edge_indices, last_trigger_state)."""
        if self._baseline_estimate is None:
            return np.array([], dtype=np.intp), prev_high
        threshold = max(self._trigger_threshold, self._noise_scale * self._noise_estimate)
        trigger_high = np.abs(aux_in - self._baseline_estimate) >= threshold
        rising_edges = trigger_high & ~np.concatenate(([prev_high], trigger_high[:-1]))
        return np.flatnonzero(rising_edges), bool(trigger_high[-1])

    def _complete_capture(self) -> CapturedWindow:
        timestamps = self._time_buffer[: self._write_pos].copy()
        sensor_forces = self._force_buffer[: self._write_pos, :].copy()
        finger_forces, finger_labels = aggregate_finger_forces(
            sensor_forces, self._finger_sensor_map
        )
        finger_ranges = np.ptp(finger_forces, axis=0)
        trigger_index = self._trigger_index_in_buffer

        self._capturing = False
        self._write_pos = 0

        return CapturedWindow(
            timestamps=timestamps,
            finger_forces=finger_forces,
            finger_ranges=finger_ranges,
            finger_labels=finger_labels,
            trigger_index=trigger_index,
        )
