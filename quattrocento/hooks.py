import logging
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from .models import DataBatch, StreamMeta

logger = logging.getLogger("quattrocento.hooks")

_HOLD_TRACE_ROLLING_SECONDS = 10.0


class _ForceMeterDialog(QtWidgets.QDialog):
    """Small live readout for the R Index hook: a vertical progress bar
    that tracks the latest % MVC, plus a numeric readout, a threshold
    spinbox, and a status line. Coloured threshold zones make it obvious
    when the signal has crossed the firing line."""

    threshold_changed = QtCore.pyqtSignal(float)

    def __init__(
        self,
        threshold_pct: float,
        min_elapsed_s: float = 2.0,
        max_pct: float = 25.0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Any Finger — % MVC")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.setFixedSize(220, 500)
        self._threshold_pct = threshold_pct
        self._min_elapsed_s = min_elapsed_s
        self._max_pct = max(max_pct, threshold_pct * 1.5)

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
        self._bar.setRange(0, int(round(self._max_pct * 10)))
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar, stretch=1, alignment=QtCore.Qt.AlignHCenter)

        threshold_row = QtWidgets.QHBoxLayout()
        threshold_row.setSpacing(6)
        prefix_label = QtWidgets.QLabel("Threshold:")
        prefix_label.setStyleSheet("font-size: 12px; color: #1F3242; font-weight: 600;")
        threshold_row.addWidget(prefix_label)
        self._threshold_spin = QtWidgets.QDoubleSpinBox()
        self._threshold_spin.setRange(1.0, 100.0)
        self._threshold_spin.setSingleStep(1.0)
        self._threshold_spin.setDecimals(1)
        self._threshold_spin.setSuffix(" %")
        self._threshold_spin.setValue(threshold_pct)
        self._threshold_spin.valueChanged.connect(self.threshold_changed.emit)
        threshold_row.addWidget(self._threshold_spin, stretch=1)
        layout.addLayout(threshold_row)

        self._status_label = QtWidgets.QLabel("Waiting…")
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
        self._status_label.setStyleSheet(
            "font-size: 12px; color: #64748B; font-weight: 600;"
        )
        layout.addWidget(self._status_label)

        self._threshold_label = QtWidgets.QLabel()
        self._threshold_label.setAlignment(QtCore.Qt.AlignCenter)
        self._threshold_label.setStyleSheet("font-size: 11px; color: #94A3B8;")
        layout.addWidget(self._threshold_label)
        self._refresh_threshold_label()

        self._set_bar_color("#94A3B8")

    def _refresh_threshold_label(self) -> None:
        self._threshold_label.setText(
            f"Fires above {self._threshold_pct:.1f}% (after onset + ≥{self._min_elapsed_s:.1f} s)"
        )

    def set_threshold(self, pct: float) -> None:
        """Update the displayed threshold. Does not touch the detector — the
        compositor is responsible for that. Spinbox edits are blocked while
        the new value is written back to avoid signal re-entry."""
        self._threshold_pct = pct
        if pct * 1.5 > self._max_pct:
            self._max_pct = pct * 1.5
            self._bar.setRange(0, int(round(self._max_pct * 10)))
        if self._threshold_spin.value() != pct:
            self._threshold_spin.blockSignals(True)
            self._threshold_spin.setValue(pct)
            self._threshold_spin.blockSignals(False)
        self._refresh_threshold_label()

    def _set_bar_color(self, color: str) -> None:
        """Sets the CSS color of the progress bar chunk."""
        self._bar.setStyleSheet(
            "QProgressBar { border: 1px solid #BFD0E1; border-radius: 6px; "
            "background: #F4F7FB; } "
            f"QProgressBar::chunk {{ background: {color}; border-radius: 4px; }}"
        )

    def set_pct(self, pct: float) -> None:
        """Updates the progress bar and label with the given MVC percentage."""
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
        """Updates the status line text."""
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
        """Clears all streak and onset state, resetting the detector to idle."""
        self._streak_samples = 0
        self._streak_start_t = None
        self._below_streak = 0
        self._ramp_onset_t = None
        self._fired = False

    def set_threshold(self, pct: float) -> None:
        """Update the crossing threshold and reset all streak/onset state.

        Resetting prevents the new threshold from firing on a streak that
        built up under the old one.
        """
        self._threshold_pct = pct
        self.reset()

    def update(
        self,
        pct: np.ndarray,
        timestamps: np.ndarray,
        sample_rate_hz: int,
    ) -> float | None:
        """Advance by one batch. Returns the crossing timestamp the first time
        the threshold is crossed with sufficient elapsed time since onset, else None.
        onset_t remains set after firing so callers can compute elapsed.

        Auto-re-arms when force returns below `onset_floor_pct` for
        `release_dwell_s` after a fire (release branch clears `_fired`).
        At most one crossing is returned per call: if a batch contains both
        a release and a new crossing, the second fire is deferred to the
        next batch."""
        self._update_onset_state(pct, timestamps, sample_rate_hz)

        if self._fired:
            return None

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
        """Advances the onset/release streak counters by one batch of samples."""
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
                    return  # remaining samples in batch still checked for threshold crossing in update()
            else:
                self._streak_samples = 0
                self._streak_start_t = None
                if self._ramp_onset_t is not None:
                    self._below_streak += 1
                    if self._below_streak >= release_dwell:
                        self._ramp_onset_t = None
                        self._below_streak = 0
                        self._fired = False


class _LabJackPulse:
    """Manages a LabJack T4 connection and fires a 5 ms TTL pulse on FIO4."""

    def __init__(self) -> None:
        self._ljm = None
        self._handle = None

    def open(self) -> None:
        """Established connection to a LabJack T4 device."""
        from labjack import ljm
        self._ljm = ljm
        self._handle = ljm.openS("T4", "ANY", "ANY")

    def close(self) -> None:
        """Close the LabJack connection."""
        if self._handle is not None:
            self._ljm.close(self._handle)  # type: ignore[union-attr]
            self._handle = None
            self._ljm = None

    def fire(self) -> None:
        """Fire a 5 ms TTL pulse on FIO4."""
        self._ljm.eWriteName(self._handle, "FIO4", 1)  # type: ignore[union-attr]
        time.sleep(0.005)
        self._ljm.eWriteName(self._handle, "FIO4", 0)  # type: ignore[union-attr]


class PassedThresholdAnyFinger:
    """Compositor: wires _RampOnsetDetector → _ForceMeterDialog → _LabJackPulse.

    Stays inert until `set_active(True)`. On activation a small always-on-top
    dialog opens showing the max % MVC across all finger channels and a
    spinbox to edit the firing threshold live; the dialog closes on
    deactivation. Fires when any finger first crosses the threshold after
    sufficient time since onset. After firing, auto-re-arms once force drops
    below `onset_floor_pct` for `release_dwell_s`; a new ramp then triggers
    a fresh fire.
    """

    name = "Any Finger Threshold"
    ui_controls = True
    group = "rtms"

    def __init__(
        self,
        finger_indices: list[int],
        threshold_pct: float = 10.0,
        onset_floor_pct: float = 3.0,
        onset_dwell_s: float = 0.2,
        min_elapsed_s: float = 2.0,
        release_dwell_s: float = 0.5,
    ) -> None:
        self._finger_indices = finger_indices
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

    def reset(self) -> None:
        """Reset internal hook state."""
        self._detector.reset()
        if self._meter is not None:
            self._meter.set_pct(0.0)
            self._meter.set_status("Armed — waiting for onset")
        logger.info("PassedThresholdAnyFinger RESET — detector streak cleared")

    def set_threshold(self, pct: float) -> None:
        """Update the firing threshold across detector and meter, and reset
        the detector so the new threshold cannot fire on a streak built up
        under the old threshold."""
        self._threshold_pct = pct
        self._detector.set_threshold(pct)
        if self._meter is not None:
            self._meter.set_threshold(pct)
            self._meter.set_pct(0.0)
            self._meter.set_status("Armed — waiting for onset")
        logger.info(f"threshold set to {pct:.1f}% — detector reset")

    def set_active(self, active: bool) -> None:
        """Set whether the hook is active."""
        self._detector.reset()
        if active:
            self._hw.open()
            self._active = True
            if self._meter is None:
                self._meter = _ForceMeterDialog(
                    threshold_pct=self._threshold_pct,
                    min_elapsed_s=self._min_elapsed_s,
                )
                self._meter.threshold_changed.connect(self.set_threshold)
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
        """Process a data batch and update the state machine."""
        if not self._active:
            return
        if meta.baseline is None or meta.peak is None:
            if self._meter is not None:
                self._meter.set_status("Calibration missing")
            return
        if batch.timestamps.shape[0] == 0:
            return

        pct_cols = []
        for idx in self._finger_indices:
            span = meta.peak[idx] - meta.baseline[idx]
            if span == 0:
                continue
            pct_cols.append((batch.signals[:, idx] - meta.baseline[idx]) / span * 100.0)

        if not pct_cols:
            if self._meter is not None:
                self._meter.set_status("Zero MVC span — recalibrate")
            return

        pct = np.max(np.stack(pct_cols, axis=1), axis=1)
        if self._meter is not None:
            self._meter.set_pct(float(pct[-1]))

        prev_onset = self._detector.onset_t
        crossing_t = self._detector.update(pct, batch.timestamps, meta.config.sample_rate_hz)
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


class _HoldTargetDialog(QtWidgets.QDialog):
    """Live readout for the Hold-In-Target hook: a rolling time-series of
    % MVC with horizontal target/low/high band lines, a numeric readout,
    a horizontal dwell-progress bar, and spinboxes for target and T."""

    target_changed = QtCore.pyqtSignal(float)
    dwell_changed = QtCore.pyqtSignal(float)

    def __init__(
        self,
        target_pct: float,
        dwell_s: float,
        tolerance_rel: float,
        rolling_seconds: float = _HOLD_TRACE_ROLLING_SECONDS,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hold In Target — % MVC")
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        self.resize(540, 460)
        self._target_pct = target_pct
        self._dwell_s = dwell_s
        self._tolerance_rel = tolerance_rel
        self._rolling_seconds = rolling_seconds
        self._times = np.empty(0, dtype=np.float64)
        self._pcts = np.empty(0, dtype=np.float64)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._value_label = QtWidgets.QLabel("0.0%")
        self._value_label.setAlignment(QtCore.Qt.AlignCenter)
        self._value_label.setStyleSheet(
            "font-size: 24px; font-weight: 700; color: #1F3242;"
        )
        layout.addWidget(self._value_label)

        self._plot = pg.PlotWidget()
        self._plot.setTitle(f"% MVC — rolling {self._rolling_seconds:.0f} s")
        self._plot.setLabel("left", "% MVC")
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setMenuEnabled(False)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        layout.addWidget(self._plot, stretch=1)

        self._band_region = pg.LinearRegionItem(
            values=[
                target_pct * (1.0 - tolerance_rel),
                target_pct * (1.0 + tolerance_rel),
            ],
            orientation="horizontal",
            brush=pg.mkBrush(42, 157, 143, 50),
            pen=pg.mkPen(None),
            movable=False,
        )
        self._band_region.setZValue(-10)
        self._plot.addItem(self._band_region)

        self._target_line = pg.InfiniteLine(
            angle=0,
            pos=target_pct,
            pen=pg.mkPen("#2A9D8F", width=1.5, style=QtCore.Qt.DashLine),
            label="target",
            labelOpts={"position": 0.04, "color": "#2A9D8F"},
        )
        self._plot.addItem(self._target_line)

        self._curve = self._plot.plot([], [], pen=pg.mkPen("#0077B6", width=1.8))

        self._band_label = QtWidgets.QLabel()
        self._band_label.setAlignment(QtCore.Qt.AlignCenter)
        self._band_label.setStyleSheet("font-size: 12px; color: #1F3242;")
        layout.addWidget(self._band_label)

        self._dwell_bar = QtWidgets.QProgressBar()
        self._dwell_bar.setOrientation(QtCore.Qt.Horizontal)
        self._dwell_bar.setRange(0, int(round(self._dwell_s * 1000)))
        self._dwell_bar.setFormat("%v / %m ms")
        layout.addWidget(self._dwell_bar)

        spin_row = QtWidgets.QHBoxLayout()
        spin_row.setSpacing(12)

        target_prefix = QtWidgets.QLabel("Target:")
        target_prefix.setStyleSheet("font-size: 12px; color: #1F3242; font-weight: 600;")
        spin_row.addWidget(target_prefix)
        self._target_spin = QtWidgets.QDoubleSpinBox()
        self._target_spin.setRange(1.0, 100.0)
        self._target_spin.setSingleStep(1.0)
        self._target_spin.setDecimals(1)
        self._target_spin.setSuffix(" %")
        self._target_spin.setValue(target_pct)
        self._target_spin.valueChanged.connect(self.target_changed.emit)
        spin_row.addWidget(self._target_spin)

        dwell_prefix = QtWidgets.QLabel("Hold T:")
        dwell_prefix.setStyleSheet("font-size: 12px; color: #1F3242; font-weight: 600;")
        spin_row.addWidget(dwell_prefix)
        self._dwell_spin = QtWidgets.QDoubleSpinBox()
        self._dwell_spin.setRange(0.1, 30.0)
        self._dwell_spin.setSingleStep(0.1)
        self._dwell_spin.setDecimals(2)
        self._dwell_spin.setSuffix(" s")
        self._dwell_spin.setValue(dwell_s)
        self._dwell_spin.valueChanged.connect(self.dwell_changed.emit)
        spin_row.addWidget(self._dwell_spin)
        spin_row.addStretch(1)
        layout.addLayout(spin_row)

        self._status_label = QtWidgets.QLabel("Waiting…")
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
        self._status_label.setStyleSheet(
            "font-size: 12px; color: #64748B; font-weight: 600;"
        )
        layout.addWidget(self._status_label)

        self._refresh_band_label()
        self._refresh_y_range()

    def _refresh_band_label(self) -> None:
        low = self._target_pct * (1.0 - self._tolerance_rel)
        high = self._target_pct * (1.0 + self._tolerance_rel)
        self._band_label.setText(
            f"Band: {low:.1f}–{high:.1f}% MVC  (±{self._tolerance_rel * 100:.0f}%)"
        )

    def _refresh_y_range(self) -> None:
        high = self._target_pct * (1.0 + self._tolerance_rel)
        y_max = max(high * 1.5, 10.0)
        self._plot.setYRange(0.0, y_max, padding=0.0)

    def set_target(self, pct: float) -> None:
        """Update the displayed target. Spinbox edits are blocked while the
        new value is written back to avoid signal re-entry."""
        self._target_pct = pct
        low = pct * (1.0 - self._tolerance_rel)
        high = pct * (1.0 + self._tolerance_rel)
        self._band_region.setRegion((low, high))
        self._target_line.setPos(pct)
        if self._target_spin.value() != pct:
            self._target_spin.blockSignals(True)
            self._target_spin.setValue(pct)
            self._target_spin.blockSignals(False)
        self._refresh_band_label()
        self._refresh_y_range()

    def set_dwell(self, s: float) -> None:
        """Update the displayed dwell target."""
        self._dwell_s = s
        self._dwell_bar.setRange(0, int(round(s * 1000)))
        if self._dwell_spin.value() != s:
            self._dwell_spin.blockSignals(True)
            self._dwell_spin.setValue(s)
            self._dwell_spin.blockSignals(False)

    def push_batch(self, timestamps: np.ndarray, pct: np.ndarray) -> None:
        """Append samples to the rolling trace, scroll the X axis to follow
        the latest sample, and refresh the numeric readout."""
        if timestamps.size == 0:
            return
        self._times = np.concatenate((self._times, timestamps))
        self._pcts = np.concatenate((self._pcts, pct))
        t_now = float(self._times[-1])
        cutoff = t_now - self._rolling_seconds
        keep = self._times >= cutoff
        self._times = self._times[keep]
        self._pcts = self._pcts[keep]
        self._curve.setData(self._times, self._pcts)
        self._plot.setXRange(t_now - self._rolling_seconds, t_now, padding=0.0)
        self._value_label.setText(f"{float(self._pcts[-1]):.1f}%")

    def clear_trace(self) -> None:
        """Empties the rolling trace buffer and resets the numeric readout."""
        self._times = np.empty(0, dtype=np.float64)
        self._pcts = np.empty(0, dtype=np.float64)
        self._curve.setData([], [])
        self._value_label.setText("0.0%")

    def set_time_in_band(self, seconds: float) -> None:
        """Updates the dwell-progress bar."""
        clamped = max(0.0, min(self._dwell_s, seconds))
        self._dwell_bar.setValue(int(round(clamped * 1000)))

    def set_status(self, text: str) -> None:
        """Updates the status line text."""
        self._status_label.setText(text)


class _HoldInBandDetector:
    """Pure hold-in-band state machine. No I/O or Qt.

    Tracks how long the input has been inside [target × (1-rel), target × (1+rel)].
    Fires whenever the in-band streak reaches `dwell_s`; on each fire the
    timer restarts from the fire instant, so a continuous hold produces
    periodic fires every `dwell_s`. Any out-of-band sample resets the timer."""

    def __init__(
        self,
        target_pct: float,
        tolerance_rel: float,
        dwell_s: float,
    ) -> None:
        self._target_pct = target_pct
        self._tolerance_rel = tolerance_rel
        self._dwell_s = dwell_s
        self._in_band_start_t: float | None = None
        self._time_in_band_s: float = 0.0

    @property
    def time_in_band_s(self) -> float:
        return self._time_in_band_s

    @property
    def in_band(self) -> bool:
        """True when the input is currently inside the target band."""
        return self._in_band_start_t is not None

    @property
    def low_pct(self) -> float:
        return self._target_pct * (1.0 - self._tolerance_rel)

    @property
    def high_pct(self) -> float:
        return self._target_pct * (1.0 + self._tolerance_rel)

    def reset(self) -> None:
        """Clears the in-band streak."""
        self._in_band_start_t = None
        self._time_in_band_s = 0.0

    def set_target(self, pct: float) -> None:
        """Update the target % and reset the streak. Resetting prevents the
        new band from firing on a streak built up under the old one."""
        self._target_pct = pct
        self.reset()

    def set_dwell(self, s: float) -> None:
        """Update the required dwell duration and reset the streak. Resetting
        avoids an immediate fire if dwell is shortened below the current
        time-in-band."""
        self._dwell_s = s
        self.reset()

    def update(
        self,
        pct: np.ndarray,
        timestamps: np.ndarray,
    ) -> float | None:
        """Advance by one batch. Returns the timestamp at which the in-band
        streak first reaches `dwell_s`, else None. On fire the timer
        restarts from that instant so a sustained hold re-fires every
        `dwell_s`. Any out-of-band sample resets the timer.

        At most one fire is returned per call; if a sustained hold spans
        multiple `dwell_s` intervals within a single batch (only possible
        when the batch is wider than `dwell_s`), the second fire is deferred
        to the next batch."""
        if pct.shape[0] == 0:
            return None

        low = self.low_pct
        high = self.high_pct
        fired_t: float | None = None

        for k in range(pct.shape[0]):
            t = float(timestamps[k])
            in_band = low <= pct[k] <= high
            if in_band:
                if self._in_band_start_t is None:
                    self._in_band_start_t = t
                    self._time_in_band_s = 0.0
                else:
                    self._time_in_band_s = t - self._in_band_start_t
                if self._time_in_band_s >= self._dwell_s and fired_t is None:
                    fired_t = t
                    self._in_band_start_t = t
                    self._time_in_band_s = 0.0
            else:
                self._in_band_start_t = None
                self._time_in_band_s = 0.0

        return fired_t


class HoldInTargetAnyFinger:
    """Compositor: wires _HoldInBandDetector → _HoldTargetDialog → _LabJackPulse.

    Stays inert until `set_active(True)`. On activation a small always-on-top
    dialog opens showing the max % MVC across all finger channels, the
    configured target band, and a horizontal dwell-progress bar. Fires when
    the % MVC has been continuously inside the target band for `dwell_s`.
    During a sustained hold, re-fires every `dwell_s`; any out-of-band sample
    resets the timer. (Contrast: PassedThresholdAnyFinger requires a release
    dip before re-arming.)
    """

    name = "Hold In Target"
    ui_controls = True
    group = "rtms"

    def __init__(
        self,
        finger_indices: list[int],
        target_pct: float = 30.0,
        tolerance_rel: float = 0.20,
        dwell_s: float = 2.0,
    ) -> None:
        self._finger_indices = finger_indices
        self._target_pct = target_pct
        self._tolerance_rel = tolerance_rel
        self._dwell_s = dwell_s
        self._detector = _HoldInBandDetector(
            target_pct=target_pct,
            tolerance_rel=tolerance_rel,
            dwell_s=dwell_s,
        )
        self._hw = _LabJackPulse()
        self._active = False
        self._meter: _HoldTargetDialog | None = None

    def reset(self) -> None:
        """Reset internal hook state."""
        self._detector.reset()
        if self._meter is not None:
            self._meter.clear_trace()
            self._meter.set_time_in_band(0.0)
            self._meter.set_status("Armed — waiting to enter band")
        logger.info("HoldInTargetAnyFinger RESET — detector streak cleared")

    def set_target(self, pct: float) -> None:
        """Update the target % MVC across detector and meter; resets streak."""
        self._target_pct = pct
        self._detector.set_target(pct)
        if self._meter is not None:
            self._meter.set_target(pct)
            self._meter.clear_trace()
            self._meter.set_time_in_band(0.0)
            self._meter.set_status("Armed — waiting to enter band")
        logger.info(f"target set to {pct:.1f}% — detector reset")

    def set_dwell(self, s: float) -> None:
        """Update the dwell T across detector and meter; resets streak."""
        self._dwell_s = s
        self._detector.set_dwell(s)
        if self._meter is not None:
            self._meter.set_dwell(s)
            self._meter.clear_trace()
            self._meter.set_time_in_band(0.0)
            self._meter.set_status("Armed — waiting to enter band")
        logger.info(f"dwell set to {s:.2f}s — detector reset")

    def set_active(self, active: bool) -> None:
        """Set whether the hook is active."""
        self._detector.reset()
        if active:
            self._hw.open()
            self._active = True
            if self._meter is None:
                self._meter = _HoldTargetDialog(
                    target_pct=self._target_pct,
                    dwell_s=self._dwell_s,
                    tolerance_rel=self._tolerance_rel,
                )
                self._meter.target_changed.connect(self.set_target)
                self._meter.dwell_changed.connect(self.set_dwell)
            self._meter.clear_trace()
            self._meter.set_time_in_band(0.0)
            self._meter.set_status("Armed — waiting to enter band")
            self._meter.show()
            self._meter.raise_()
        else:
            self._active = False
            self._hw.close()
            if self._meter is not None:
                self._meter.hide()
        logger.info(f"{'ACTIVATED' if active else 'DEACTIVATED'} — state cleared")

    def __call__(self, batch: DataBatch, meta: StreamMeta) -> None:
        """Process a data batch and update the state machine."""
        if not self._active:
            return
        if meta.baseline is None or meta.peak is None:
            if self._meter is not None:
                self._meter.set_status("Calibration missing")
            return
        if batch.timestamps.shape[0] == 0:
            return

        pct_cols = []
        for idx in self._finger_indices:
            span = meta.peak[idx] - meta.baseline[idx]
            if span == 0:
                continue
            pct_cols.append((batch.signals[:, idx] - meta.baseline[idx]) / span * 100.0)

        if not pct_cols:
            if self._meter is not None:
                self._meter.set_status("Zero MVC span — recalibrate")
            return

        pct = np.max(np.stack(pct_cols, axis=1), axis=1)
        if self._meter is not None:
            self._meter.push_batch(batch.timestamps, pct)

        prev_in_band = self._detector.in_band
        fire_t = self._detector.update(pct, batch.timestamps)
        now_in_band = self._detector.in_band

        if self._meter is not None:
            self._meter.set_time_in_band(self._detector.time_in_band_s)
            if not prev_in_band and now_in_band:
                self._meter.set_status(
                    f"In band — holding ≥{self._dwell_s:.2f}s"
                )
            elif prev_in_band and not now_in_band and fire_t is None:
                self._meter.set_status("Left band — waiting to re-enter")

        if fire_t is not None:
            logger.info(
                f"HELD  (fire at t={fire_t:.3f}s, "
                f"target={self._target_pct:.1f}% ±{self._tolerance_rel * 100:.0f}%, "
                f"dwell={self._dwell_s:.2f}s)"
            )
            self._hw.fire()
            if self._meter is not None:
                self._meter.set_status(f"HELD ✓  T={self._dwell_s:.2f}s")
