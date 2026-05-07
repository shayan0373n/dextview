from __future__ import annotations

import logging
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .models import DataBatch, StreamMeta

logger = logging.getLogger("quattrocento.hooks")


class _ForceMeterDialog(QtWidgets.QDialog):
    """Small live readout for the R Index hook: a vertical progress bar
    that tracks the latest % MVC, plus a numeric readout and a status
    line. Coloured threshold zones make it obvious when the signal has
    crossed the firing line."""

    def __init__(
        self,
        threshold_pct: float,
        max_pct: float = 25.0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("R Index — % MVC")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setFixedSize(220, 460)
        self._threshold_pct = threshold_pct
        self._max_pct = max_pct

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self._value_label = QtWidgets.QLabel("0.0%")
        self._value_label.setAlignment(QtCore.Qt.AlignCenter)
        self._value_label.setStyleSheet(
            "font-size: 28px; font-weight: 700; color: #1F3242;"
        )
        layout.addWidget(self._value_label)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setOrientation(QtCore.Qt.Vertical)
        # 0.1% resolution: scale by 10.
        self._bar.setRange(0, int(round(max_pct * 10)))
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar, stretch=1, alignment=QtCore.Qt.AlignHCenter)

        self._status_label = QtWidgets.QLabel("Waiting…")
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
        self._status_label.setStyleSheet(
            "font-size: 12px; color: #64748B; font-weight: 600;"
        )
        layout.addWidget(self._status_label)

        self._threshold_label = QtWidgets.QLabel(
            f"Fires above {threshold_pct:.0f}% (after onset + ≥2 s)"
        )
        self._threshold_label.setAlignment(QtCore.Qt.AlignCenter)
        self._threshold_label.setStyleSheet("font-size: 11px; color: #94A3B8;")
        layout.addWidget(self._threshold_label)

        self._set_bar_color("#94A3B8")

    def _set_bar_color(self, color: str) -> None:
        self._bar.setStyleSheet(
            "QProgressBar { border: 1px solid #BFD0E1; border-radius: 6px; "
            "background: #F4F7FB; } "
            f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}"
        )

    def set_pct(self, pct: float) -> None:
        self._value_label.setText(f"{pct:.1f}%")
        clamped = max(0.0, min(self._max_pct, pct))
        self._bar.setValue(int(round(clamped * 10)))
        if pct >= self._threshold_pct:
            self._set_bar_color("#2A9D8F")
        elif pct >= self._threshold_pct * 0.5:
            self._set_bar_color("#F6B73C")
        else:
            self._set_bar_color("#6FA8DC")

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)


class _RampOnsetDetector:
    """Pure onset/release/threshold-crossing state machine. No I/O or Qt."""

    def __init__(
        self,
        threshold_pct: float,
        onset_floor_pct: float,
        onset_dwell_s: float,
        min_elapsed_s: float,
        release_dwell_s: float,
    ) -> None:
        self._threshold_pct = threshold_pct
        self._onset_floor_pct = onset_floor_pct
        self._onset_dwell_s = onset_dwell_s
        self._min_elapsed_s = min_elapsed_s
        self._release_dwell_s = release_dwell_s
        self._streak_samples = 0
        self._streak_start_t: float | None = None
        self._below_streak = 0
        self._ramp_onset_t: float | None = None
        self._fired = False

    @property
    def onset_t(self) -> float | None:
        return self._ramp_onset_t

    def reset(self) -> None:
        self._streak_samples = 0
        self._streak_start_t = None
        self._below_streak = 0
        self._ramp_onset_t = None
        self._fired = False

    def update(
        self,
        pct: np.ndarray,
        timestamps: np.ndarray,
        sample_rate_hz: int,
    ) -> float | None:
        """Advance by one batch. Returns the crossing timestamp the first time
        the threshold is crossed with sufficient elapsed time since onset, else None.
        onset_t remains set after firing so callers can compute elapsed."""
        if self._fired:
            return None

        self._update_onset_state(pct, timestamps, sample_rate_hz)

        if self._ramp_onset_t is None:
            return None

        crossings = np.flatnonzero(pct >= self._threshold_pct)
        if crossings.size == 0:
            return None
        crossing_t = float(timestamps[crossings[0]])
        elapsed = crossing_t - self._ramp_onset_t
        if elapsed < self._min_elapsed_s:
            logger.debug(
                f"{self._threshold_pct:.0f}% crossing at t={crossing_t:.3f}s "
                f"rejected — only {elapsed:.3f}s since onset "
                f"(min {self._min_elapsed_s:.1f}s)"
            )
            return None

        self._fired = True
        return crossing_t

    def _update_onset_state(
        self,
        pct: np.ndarray,
        timestamps: np.ndarray,
        sample_rate_hz: int,
    ) -> None:
        onset_dwell = max(1, int(round(self._onset_dwell_s * sample_rate_hz)))
        release_dwell = max(1, int(round(self._release_dwell_s * sample_rate_hz)))
        floor = self._onset_floor_pct

        for k in range(pct.shape[0]):
            if pct[k] >= floor:
                self._below_streak = 0
                if self._streak_samples == 0:
                    self._streak_start_t = float(timestamps[k])
                self._streak_samples += 1
                if (
                    self._ramp_onset_t is None
                    and self._streak_samples >= onset_dwell
                ):
                    self._ramp_onset_t = self._streak_start_t
                    return
            else:
                self._streak_samples = 0
                self._streak_start_t = None
                if self._ramp_onset_t is not None:
                    self._below_streak += 1
                    if self._below_streak >= release_dwell:
                        self._ramp_onset_t = None
                        self._below_streak = 0


class _LabJackPulse:
    """Manages a LabJack T4 connection and fires a 5 ms TTL pulse on FIO4.

    The labjack driver is imported on open() so hosts without the driver
    installed are unaffected until the hook is activated."""

    def __init__(self) -> None:
        self._ljm: object | None = None
        self._handle: object | None = None

    def open(self) -> None:
        from labjack import ljm
        self._ljm = ljm
        self._handle = ljm.openS("T4", "ANY", "ANY")

    def close(self) -> None:
        if self._handle is not None:
            self._ljm.close(self._handle)  # type: ignore[union-attr]
            self._handle = None
            self._ljm = None

    def fire(self) -> None:
        self._ljm.eWriteName(self._handle, "FIO4", 1)  # type: ignore[union-attr]
        time.sleep(0.005)
        self._ljm.eWriteName(self._handle, "FIO4", 0)  # type: ignore[union-attr]


class PassedTenPercentRightIndex:
    """Compositor: wires _RampOnsetDetector → _ForceMeterDialog → _LabJackPulse.

    Stays inert until `set_active(True)` (e.g. via the UI button). On
    activation a small always-on-top dialog opens showing R Index force as
    % MVC; it closes on deactivation. Designed for a 0%→20% ramp ~5 s long.
    """

    name = "R Index 10% MVC"
    ui_controls = True

    def __init__(
        self,
        threshold_pct: float = 10.0,
        onset_floor_pct: float = 3.0,
        onset_dwell_s: float = 0.2,
        min_elapsed_s: float = 2.0,
        release_dwell_s: float = 0.5,
    ) -> None:
        self._threshold_pct = threshold_pct
        self._onset_floor_pct = onset_floor_pct
        self._onset_dwell_s = onset_dwell_s
        self._min_elapsed_s = min_elapsed_s
        self._release_dwell_s = release_dwell_s
        self._detector = _RampOnsetDetector(
            threshold_pct=threshold_pct,
            onset_floor_pct=onset_floor_pct,
            onset_dwell_s=onset_dwell_s,
            min_elapsed_s=min_elapsed_s,
            release_dwell_s=release_dwell_s,
        )
        self._hw = _LabJackPulse()
        self._active = False
        self._meter: _ForceMeterDialog | None = None

    def set_active(self, active: bool) -> None:
        """Arm or disarm. Opens/closes the LabJack handle and shows/hides the meter."""
        self._detector.reset()
        if active:
            self._hw.open()
            self._active = True
            if self._meter is None:
                self._meter = _ForceMeterDialog(threshold_pct=self._threshold_pct)
            self._meter.set_pct(0.0)
            self._meter.set_status("Armed — waiting for onset")
            self._meter.show()
            self._meter.raise_()
        else:
            self._active = False
            self._hw.close()
            if self._meter is not None:
                self._meter.hide()
        logger.info(f"{'ACTIVATED' if active else 'DEACTIVATED'} — state cleared")

    def __call__(self, batch: DataBatch, meta: StreamMeta) -> None:
        if not self._active:
            return
        if meta.rest_means is None or meta.mvc_maxs is None:
            if self._meter is not None:
                self._meter.set_status("Calibration missing")
            return
        if batch.timestamps.shape[0] == 0:
            return

        finger = "R Index"
        i = meta.finger_labels.index(finger)
        sensor = meta.finger_sensor_map[finger]
        rest = meta.rest_means[i]
        span = meta.mvc_maxs[i] - rest
        if span == 0:
            if self._meter is not None:
                self._meter.set_status("Zero MVC span — recalibrate")
            return

        pct = (batch.forces[:, sensor] - rest) / span * 100.0
        if self._meter is not None:
            self._meter.set_pct(float(pct[-1]))

        prev_onset = self._detector.onset_t
        crossing_t = self._detector.update(pct, batch.timestamps, meta.sample_rate_hz)
        new_onset = self._detector.onset_t

        if prev_onset is None and new_onset is not None:
            logger.debug(
                f"onset detected at t={new_onset:.3f}s "
                f"(floor={self._onset_floor_pct:.1f}% MVC, "
                f"dwell={self._onset_dwell_s:.3f}s)"
            )
            if self._meter is not None:
                self._meter.set_status(
                    f"Onset @ t={new_onset:.2f}s — waiting ≥{self._min_elapsed_s:.1f}s"
                )
        elif prev_onset is not None and new_onset is None:
            logger.debug(
                f"release detected — force below {self._onset_floor_pct:.1f}% "
                f"for ≥{self._release_dwell_s:.2f}s, timer reset"
            )
            if self._meter is not None:
                self._meter.set_status("Released — waiting for next onset")

        if crossing_t is not None:
            elapsed = crossing_t - self._detector.onset_t  # onset_t persists after fire
            logger.info(
                f"PASSED  (crossing at t={crossing_t:.3f}s, "
                f"elapsed since onset {elapsed:.3f}s)"
            )
            self._hw.fire()
            if self._meter is not None:
                self._meter.set_status(f"PASSED ✓  Δt={elapsed:.2f}s")
