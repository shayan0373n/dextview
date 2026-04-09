from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .config import QuattrocentoConfig
from .models import CapturedWindow, DataBatch


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
        self._window_samples = config.window_samples
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
        self._time_buffer = np.empty(self._window_samples, dtype=np.float64)
        self._force_buffer = np.empty(
            (self._window_samples, self._sensor_count), dtype=np.float64
        )

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

    def process_batch(self, batch: DataBatch) -> CapturedWindow | None:
        """Consume one batch and return a completed captured window when available."""
        batch_size = batch.timestamps.shape[0]
        if batch_size == 0:
            return None

        # Identify rising edges seamlessly across batch boundaries
        trigger_high = self._compute_trigger_high(batch.aux_in)
        prev_high = np.concatenate(([self._previous_trigger_high], trigger_high[:-1]))
        rising_edges = trigger_high & ~prev_high

        self._previous_trigger_high = bool(trigger_high[-1])

        # Fast path: skip processing if inactive and no new triggers fired
        if not (self._capturing or np.any(rising_edges)):
            return None

        captured_window: CapturedWindow | None = None

        if not self._capturing:
            # Find first rising edge and start capture after it.
            edge_idx = int(np.argmax(rising_edges))
            start = edge_idx + 1
            if start < batch_size:
                captured_window = self._collect_range(
                    batch, start, batch_size
                )
        else:
            # NOTE: samples after capture completion within this batch are
            # discarded, and any rising edge in that tail is lost. At current
            # operating parameters (512 Hz, 50 ms batches, 8 s trigger interval)
            # this cannot occur. Handling it would require splitting the batch
            # at the completion boundary and scanning the remainder for edges.
            captured_window = self._collect_range(batch, 0, batch_size)

        return captured_window

    def _collect_range(
        self, batch: DataBatch, start: int, end: int
    ) -> CapturedWindow | None:
        """Copy samples from batch[start:end] into the capture buffer."""
        if not self._capturing:
            self._capturing = True
            self._write_pos = 0

        remaining = self._window_samples - self._write_pos
        count = min(end - start, remaining)

        wp = self._write_pos
        self._time_buffer[wp : wp + count] = batch.timestamps[start : start + count]
        self._force_buffer[wp : wp + count, :] = batch.forces[start : start + count, :]
        self._write_pos += count

        if self._write_pos >= self._window_samples:
            return self._complete_capture()
        return None

    def _compute_trigger_high(self, aux_in: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Threshold AUX after adaptive baseline/noise normalization."""
        trigger_high = np.zeros(aux_in.shape[0], dtype=np.bool_)

        baseline = self._baseline_estimate
        noise = self._noise_estimate

        for idx, sample in enumerate(aux_in):
            sample_value = float(sample)
            if baseline is None:
                baseline = sample_value

            residual = sample_value - baseline

            dynamic_threshold = max(self._trigger_threshold, self._noise_scale * noise)
            trigger_high[idx] = abs(residual) >= dynamic_threshold

            # Update noise and baseline after the trigger check so the pulse
            # does not inflate the threshold used to detect itself.
            noise += self._noise_alpha * (abs(residual) - noise)
            baseline_step = float(np.clip(residual, -dynamic_threshold, dynamic_threshold))
            baseline += self._baseline_alpha * baseline_step

        self._baseline_estimate = baseline
        self._noise_estimate = noise
        return trigger_high

    def _complete_capture(self) -> CapturedWindow:
        timestamps = self._time_buffer[: self._write_pos].copy()
        sensor_forces = self._force_buffer[: self._write_pos, :].copy()
        finger_forces, finger_labels = aggregate_finger_forces(
            sensor_forces, self._finger_sensor_map
        )
        finger_ranges = np.ptp(finger_forces, axis=0)

        self._capturing = False
        self._write_pos = 0

        return CapturedWindow(
            timestamps=timestamps,
            finger_forces=finger_forces,
            finger_ranges=finger_ranges,
            finger_labels=finger_labels,
        )
