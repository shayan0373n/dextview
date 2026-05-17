from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .config import QuattrocentoConfig
from .models import CapturedWindow, DataBatch, StreamMeta

logger = logging.getLogger("quattrocento.processing")

# These thresholds need empirical validation against real EMG/force data.
_ONSET_SD_MULTIPLIER: float = 5.0   # SD multiples above/below pre-trigger mean
_ONSET_MIN_CONSECUTIVE: int = 5     # consecutive threshold crossings required


def detect_onset(
    signal: np.ndarray,
    trigger_idx: int,
    sample_rate_hz: int,
    post_skip_samples: int = 0,
) -> float | None:
    """Return onset time in ms relative to trigger, or None if not detected.

    Uses pre-trigger mean ± _ONSET_SD_MULTIPLIER * SD as threshold.
    Onset is confirmed after _ONSET_MIN_CONSECUTIVE consecutive threshold
    crossings in either direction. ``post_skip_samples`` excludes the first
    N samples after the trigger from the search (e.g. to ignore a stimulator
    artifact); the returned time is still measured relative to the trigger.
    """
    pre = signal[:trigger_idx]
    if len(pre) < 10:  # minimum pre-trigger samples needed for a stable baseline estimate
        return None
    mean_pre = float(np.mean(pre))
    sd_pre = float(np.std(pre))
    if sd_pre == 0.0:
        return None
    upper = mean_pre + _ONSET_SD_MULTIPLIER * sd_pre
    lower = mean_pre - _ONSET_SD_MULTIPLIER * sd_pre
    skip = max(0, int(post_skip_samples))
    post = signal[trigger_idx + skip:]
    consecutive = 0
    for i, val in enumerate(post):
        if val > upper or val < lower:
            consecutive += 1
            if consecutive >= _ONSET_MIN_CONSECUTIVE:
                onset_sample = i + skip - (_ONSET_MIN_CONSECUTIVE - 1)
                return onset_sample * 1000.0 / sample_rate_hz
        else:
            consecutive = 0
    return None


class TriggerWindowProcessor:
    """Detect rising AUX-in edges and collect fixed post-trigger windows."""

    def __init__(self, config: QuattrocentoConfig) -> None:
        """Configure trigger threshold, capture window length, and channel count."""
        self._post_samples = config.post_trigger_samples
        self._pre_samples = config.pre_trigger_samples
        self._total_samples = config.total_window_samples
        self._trigger_threshold = config.trigger_threshold
        self._trigger_channel = config.trigger_channel
        self._sample_rate_hz = config.sample_rate_hz
        self._n_channels = config.n_channels

        # Adaptive trigger model on the designated trigger channel:
        # - _trigger_dc: slow EMA to follow DC drift
        # - _trigger_noise: EMA of |signal - dc| to estimate noise floor
        # Trigger becomes HIGH when (signal - dc) exceeds
        # max(static_threshold, k * noise).
        baseline_tau_seconds = 2.0
        noise_tau_seconds = 1.0
        self._dc_alpha = 1.0 - np.exp(
            -1.0 / (self._sample_rate_hz * baseline_tau_seconds)
        )
        self._noise_alpha = 1.0 - np.exp(
            -1.0 / (self._sample_rate_hz * noise_tau_seconds)
        )
        self._noise_scale = 4.0
        self._trigger_dc: float | None = None
        self._trigger_noise = 0.0
        # Suppress edge detection for 4 s (2τ) while the DC EMA settles to ~86% of
        # its steady-state value. Prevents false triggers from startup transients.
        self._warmup_samples = int(self._sample_rate_hz * baseline_tau_seconds * 2)
        self._samples_seen = 0

        self._capturing = False
        self._previous_trigger_high = False
        self._write_pos = 0
        self._capture_target = self._total_samples
        self._trigger_sample_in_buffer = 0
        self._time_buffer = np.empty(self._total_samples, dtype=np.float64)
        self._signal_buffer = np.empty(
            (self._total_samples, self._n_channels), dtype=np.float64
        )

        # Pre-trigger ring buffer
        self._ring_time = np.empty(self._pre_samples, dtype=np.float64)
        self._ring_signal = np.empty(
            (self._pre_samples, self._n_channels), dtype=np.float64
        )
        self._ring_pos = 0
        self._ring_filled = 0

    @property
    def is_capturing(self) -> bool:
        return self._capturing

    @property
    def is_trigger_high(self) -> bool:
        return self._previous_trigger_high

    @property
    def trigger_dc(self) -> float | None:
        return self._trigger_dc

    @property
    def trigger_noise(self) -> float:
        return self._trigger_noise

    @property
    def effective_threshold(self) -> float:
        return max(self._trigger_threshold, self._noise_scale * self._trigger_noise)

    @property
    def warmup_remaining_samples(self) -> int:
        return max(0, self._warmup_samples - self._samples_seen)

    def reset(self) -> None:
        """Clear internal state and drop partially collected data."""
        self._capturing = False
        self._previous_trigger_high = False
        self._trigger_dc = None
        self._trigger_noise = 0.0
        self._samples_seen = 0
        self._write_pos = 0
        self._ring_pos = 0
        self._ring_filled = 0

    def process_batch(
        self, batch: DataBatch, meta: StreamMeta
    ) -> list[CapturedWindow]:
        """Update triggers and return windows for any completed captures.

        Scans for rising edges on the trigger channel. If not already capturing,
        a new window begins at the first edge. If a capture completes mid-batch,
        scanning continues for new triggers in the remainder.
        """
        windows: list[CapturedWindow] = []
        batch_size = batch.timestamps.shape[0]
        if batch_size == 0:
            return windows

        trigger_col = batch.signals[:, self._trigger_channel]
        self._advance_trigger_dc(trigger_col)
        self._samples_seen += batch_size

        cursor = 0
        while cursor < batch_size:
            if not self._capturing:
                edges, last_high = self._find_rising_edges(
                    trigger_col[cursor:], self._previous_trigger_high
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

                window = self._collect_range(batch, cursor, cursor + count, meta)
                if window is not None:
                    windows.append(window)

                cursor += count

        self._update_ring(batch.timestamps, batch.signals, 0, batch_size)
        return windows

    def _collect_range(
        self, batch: DataBatch, start: int, end: int, meta: StreamMeta
    ) -> CapturedWindow | None:
        count = end - start
        wp = self._write_pos
        self._time_buffer[wp : wp + count] = batch.timestamps[start:end]
        self._signal_buffer[wp : wp + count, :] = batch.signals[start:end, :]
        self._write_pos += count

        if self._write_pos >= self._capture_target:
            return self._complete_capture(meta)
        return None

    def _update_ring(
        self,
        timestamps: NDArray[np.float64],
        signals: NDArray[np.float64],
        start: int,
        end: int,
    ) -> None:
        if self._pre_samples == 0:
            return
        n = end - start
        if n <= 0:
            return
        if n >= self._pre_samples:
            src_start = end - self._pre_samples
            self._ring_time[:] = timestamps[src_start:end]
            self._ring_signal[:, :] = signals[src_start:end, :]
            self._ring_pos = 0
            self._ring_filled = self._pre_samples
            return
        first = min(n, self._pre_samples - self._ring_pos)
        self._ring_time[self._ring_pos : self._ring_pos + first] = timestamps[
            start : start + first
        ]
        self._ring_signal[self._ring_pos : self._ring_pos + first, :] = signals[
            start : start + first, :
        ]
        rest = n - first
        if rest > 0:
            self._ring_time[:rest] = timestamps[start + first : end]
            self._ring_signal[:rest, :] = signals[start + first : end, :]
        self._ring_pos = (self._ring_pos + n) % self._pre_samples
        self._ring_filled = min(self._pre_samples, self._ring_filled + n)

    def _read_ring_chronological(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self._ring_filled == 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, self._n_channels), dtype=np.float64),
            )
        if self._ring_filled < self._pre_samples:
            return (
                self._ring_time[: self._ring_pos].copy(),
                self._ring_signal[: self._ring_pos, :].copy(),
            )
        rp = self._ring_pos
        time_out = np.concatenate((self._ring_time[rp:], self._ring_time[:rp]))
        signal_out = np.concatenate(
            (self._ring_signal[rp:, :], self._ring_signal[:rp, :])
        )
        return time_out, signal_out

    def _begin_capture_with_preroll(self, batch: DataBatch, edge_idx: int) -> None:
        self._capturing = True
        self._write_pos = 0

        if self._pre_samples == 0:
            self._trigger_sample_in_buffer = 0
            self._capture_target = self._total_samples
            return

        ring_time, ring_signal = self._read_ring_chronological()
        pre_time = np.concatenate((ring_time, batch.timestamps[:edge_idx]))
        pre_signal = np.concatenate((ring_signal, batch.signals[:edge_idx, :]))
        if pre_time.shape[0] > self._pre_samples:
            pre_time = pre_time[-self._pre_samples :]
            pre_signal = pre_signal[-self._pre_samples :, :]
        n = pre_time.shape[0]
        self._time_buffer[:n] = pre_time
        self._signal_buffer[:n, :] = pre_signal
        self._write_pos = n
        self._trigger_sample_in_buffer = n
        self._capture_target = n + self._post_samples

    def _advance_trigger_dc(self, trigger_col: NDArray[np.float64]) -> None:
        """Update DC/noise EMA, freezing during trigger-high periods."""
        dc = self._trigger_dc
        noise = self._trigger_noise

        for sample in trigger_col:
            sample_value = float(sample)
            if dc is None:
                dc = sample_value
                continue
            residual = sample_value - dc
            threshold = max(self._trigger_threshold, self._noise_scale * noise)
            if abs(residual) < threshold:
                noise += self._noise_alpha * (abs(residual) - noise)
                dc += self._dc_alpha * float(
                    np.clip(residual, -threshold, threshold)
                )

        self._trigger_dc = dc
        self._trigger_noise = noise

    def _find_rising_edges(
        self, trigger_col: NDArray[np.float64], prev_high: bool
    ) -> tuple[NDArray[np.intp], bool]:
        if self._trigger_dc is None or self._samples_seen < self._warmup_samples:
            return np.array([], dtype=np.intp), prev_high
        threshold = max(
            self._trigger_threshold, self._noise_scale * self._trigger_noise
        )
        trigger_high = np.abs(trigger_col - self._trigger_dc) >= threshold
        rising_edges = trigger_high & ~np.concatenate(
            ([prev_high], trigger_high[:-1])
        )
        edge_indices = np.flatnonzero(rising_edges)
        if edge_indices.size > 0:
            i = int(edge_indices[0])
            logger.debug(
                "trigger edge: signal=%.1f dc=%.1f threshold=%.1f samples_seen=%d",
                float(trigger_col[i]),
                self._trigger_dc,
                threshold,
                self._samples_seen,
            )
        return edge_indices, bool(trigger_high[-1])

    def _complete_capture(self, meta: StreamMeta) -> CapturedWindow:
        timestamps = self._time_buffer[: self._write_pos].copy()
        signals = self._signal_buffer[: self._write_pos, :].copy()
        trigger_sample = self._trigger_sample_in_buffer

        self._capturing = False
        self._write_pos = 0

        return CapturedWindow(
            batch=DataBatch(timestamps=timestamps, signals=signals),
            meta=meta,
            trigger_sample=trigger_sample,
        )
