import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

_HOLD_TRACE_ROLLING_SECONDS = 10.0


class _ForceMeterDialog(QtWidgets.QDialog):
    """Small live readout for the R Index hook: a vertical progress bar
    that tracks the latest % MVC, plus a numeric readout, a threshold
    spinbox, and a status line. Coloured threshold zones make it obvious
    when the signal has crossed the firing line."""

    threshold_changed = QtCore.pyqtSignal(float)
    reset_requested = QtCore.pyqtSignal()

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
        self.setFixedSize(220, 560)
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

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_requested.emit)
        layout.addWidget(reset_btn)

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


class _HoldTargetDialog(QtWidgets.QDialog):
    """Live readout for the Hold-In-Target hook: a rolling time-series of
    % MVC with horizontal target/low/high band lines, a numeric readout,
    a horizontal dwell-progress bar, and spinboxes for target and T."""

    target_changed = QtCore.pyqtSignal(float)
    dwell_changed = QtCore.pyqtSignal(float)
    reset_requested = QtCore.pyqtSignal()

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
        self.resize(540, 500)
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

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self.reset_requested.emit)
        layout.addWidget(reset_btn)

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
