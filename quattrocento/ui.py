from datetime import datetime
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from scipy.signal import sosfiltfilt

from .models import CapturedWindow
from .processing import (
    EMG_BANDPASS_HIGH_HZ,
    EMG_BANDPASS_LOW_HZ,
    EMG_NOTCH_HZ,
    design_emg_bandpass,
    design_emg_notch,
    detect_onset,
)


_RAW_GRID_COLUMNS = 5
_MONITOR_ROLLING_SECONDS = 5.0
_LIVE_MONITOR_ROLLING_SECONDS = 10.0
_LIVE_MONITOR_GRID_COLUMNS = 5
_EXPECTED_FINGER_COUNT = 10

# Peak-force bins used to color the per-finger max marker and the legend.
_MVC_COLOR_BINS: tuple[tuple[float, str, str], ...] = (
    (5.0, "≤5", "#A0AEC0"),
    (10.0, "≤10", "#F6E05E"),
    (20.0, "≤20", "#ED8936"),
    (40.0, "≤40", "#E53E3E"),
    (60.0, "≤60", "#9F7AEA"),
    (float("inf"), ">60", "#742A2A"),
)


def _mvc_bin_color(peak_force: float) -> str:
    """Returns a color hex string based on which MVC-percentage bin the peak force falls into."""
    for threshold, _label, color in _MVC_COLOR_BINS:
        if peak_force <= threshold:
            return color
    return _MVC_COLOR_BINS[-1][2]


FINGER_COLORS = (
    "#0B4F6C",
    "#9C2D48",
    "#F18F01",
    "#2E8B57",
    "#4A4E69",
    "#2A9D8F",
    "#E76F51",
    "#8B5E34",
    "#0077B6",
    "#6C757D",
)


def _build_mirrored_bar_layout(
    finger_labels: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    finger_order = ("thumb", "index", "middle", "ring", "little")

    left_map: dict[str, int] = {}
    right_map: dict[str, int] = {}
    for idx, label in enumerate(finger_labels):
        token = label.strip().lower()
        side: str | None = None
        if token.startswith("l ") or token.startswith("left "):
            side = "left"
        elif token.startswith("r ") or token.startswith("right "):
            side = "right"
        if side is None:
            continue

        finger: str | None = None
        for name in finger_order:
            if name in token:
                finger = name
                break
        if finger is None:
            continue

        if side == "left":
            left_map[finger] = idx
        else:
            right_map[finger] = idx

    if len(left_map) == 5 and len(right_map) == 5:
        mirrored_indices = (
            left_map["little"],
            left_map["ring"],
            left_map["middle"],
            left_map["index"],
            left_map["thumb"],
            right_map["thumb"],
            right_map["index"],
            right_map["middle"],
            right_map["ring"],
            right_map["little"],
        )
        mirrored_labels = tuple(finger_labels[idx] for idx in mirrored_indices)
        return mirrored_indices, mirrored_labels

    identity = tuple(range(len(finger_labels)))
    return identity, tuple(finger_labels)


class TriggerMonitorWindow(QtWidgets.QWidget):
    """Rolling live view of the trigger channel with adaptive threshold overlay."""

    def __init__(
        self,
        sample_rate_hz: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Trigger Channel Monitor")
        self.resize(860, 400)

        self._sample_rate_hz = sample_rate_hz
        self._max_samples = int(sample_rate_hz * _MONITOR_ROLLING_SECONDS)
        self._times = np.empty(self._max_samples, dtype=np.float64)
        self._values = np.empty(self._max_samples, dtype=np.float64)
        self._write_pos = 0
        self._filled = 0
        self._trigger_lines: list[pg.InfiniteLine] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        chips = QtWidgets.QHBoxLayout()
        chips.setSpacing(8)
        self._dc_label = QtWidgets.QLabel("DC: —")
        self._noise_label = QtWidgets.QLabel("Noise RMS: —")
        self._threshold_label = QtWidgets.QLabel("Threshold: ±—")
        self._warmup_label = QtWidgets.QLabel("Warmup: —")
        for lbl in (self._dc_label, self._noise_label, self._threshold_label, self._warmup_label):
            lbl.setProperty("kind", "chip")
            chips.addWidget(lbl)
        chips.addStretch(1)
        layout.addLayout(chips)

        self._plot = pg.PlotWidget()
        self._plot.setTitle(f"Trigger Channel — rolling {_MONITOR_ROLLING_SECONDS:.0f} s")
        self._plot.setLabel("left", "Signal (a.u.)")
        self._plot.setLabel("bottom", "Time (s)")
        self._plot.setMenuEnabled(False)
        self._plot.setMouseEnabled(x=False, y=True)
        self._plot.showGrid(x=True, y=True, alpha=0.18)
        layout.addWidget(self._plot)

        self._curve = self._plot.plot([], [], pen=pg.mkPen("#0077B6", width=1.5))

        pen_dc = pg.mkPen("#2A9D8F", width=1.5, style=QtCore.Qt.DashLine)
        self._dc_line = pg.InfiniteLine(angle=0, pos=0, pen=pen_dc,
                                        label="DC", labelOpts={"position": 0.05, "color": "#2A9D8F"})
        self._plot.addItem(self._dc_line)

        pen_thr = pg.mkPen("#E63946", width=1.5, style=QtCore.Qt.DashLine)
        self._upper_line = pg.InfiniteLine(angle=0, pos=0, pen=pen_thr,
                                           label="+thr", labelOpts={"position": 0.92, "color": "#E63946"})
        self._lower_line = pg.InfiniteLine(angle=0, pos=0, pen=pen_thr,
                                           label="−thr", labelOpts={"position": 0.92, "color": "#E63946"})
        self._plot.addItem(self._upper_line)
        self._plot.addItem(self._lower_line)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Rejects the close event and hides the window instead."""
        self.hide()
        event.ignore()

    def push_batch(
        self,
        timestamps: np.ndarray,
        trigger_col: np.ndarray,
        dc: float | None,
        noise: float,
        effective_threshold: float,
        warmup_remaining: int,
    ) -> None:
        """Pushes a new batch of trigger-channel data into the rolling buffer and updates UI chips."""
        n = len(timestamps)
        if n == 0:
            return

        end = self._write_pos + n
        if end <= self._max_samples:
            self._times[self._write_pos:end] = timestamps
            self._values[self._write_pos:end] = trigger_col
        else:
            first = self._max_samples - self._write_pos
            self._times[self._write_pos:] = timestamps[:first]
            self._values[self._write_pos:] = trigger_col[:first]
            rest = n - first
            self._times[:rest] = timestamps[first:]
            self._values[:rest] = trigger_col[first:]
        self._write_pos = end % self._max_samples
        self._filled = min(self._max_samples, self._filled + n)

        if self._filled < self._max_samples:
            t_out = self._times[:self._filled]
            v_out = self._values[:self._filled]
        else:
            rp = self._write_pos
            t_out = np.concatenate((self._times[rp:], self._times[:rp]))
            v_out = np.concatenate((self._values[rp:], self._values[:rp]))

        self._curve.setData(t_out, v_out)

        if t_out.size > 0:
            t_now = float(t_out[-1])
            self._plot.setXRange(t_now - _MONITOR_ROLLING_SECONDS, t_now, padding=0.0)

        dc_val = dc if dc is not None else 0.0
        self._dc_line.setPos(dc_val)
        self._upper_line.setPos(dc_val + effective_threshold)
        self._lower_line.setPos(dc_val - effective_threshold)

        if v_out.size > 0:
            lo = min(float(np.min(v_out)), dc_val - effective_threshold)
            hi = max(float(np.max(v_out)), dc_val + effective_threshold)
            span = hi - lo if hi != lo else 2000.0
            self._plot.setYRange(lo - span * 0.1, hi + span * 0.1, padding=0.0)

        self._dc_label.setText(f"DC: {dc_val:.3f}" if dc is not None else "DC: —")
        self._noise_label.setText(f"Noise RMS: {noise:.3f}")
        self._threshold_label.setText(f"Threshold: ±{effective_threshold:.3f}")
        if warmup_remaining > 0:
            self._warmup_label.setText(f"Warmup: {warmup_remaining / self._sample_rate_hz:.1f} s left")
        else:
            self._warmup_label.setText("Warmup: done")

        if t_out.size > 0:
            cutoff = float(t_out[0])
            keep = []
            for line in self._trigger_lines:
                if line.value() >= cutoff:
                    keep.append(line)
                else:
                    self._plot.removeItem(line)
            self._trigger_lines = keep

    def mark_trigger(self, t: float) -> None:
        """Adds a vertical marker at the specified time to the rolling trigger plot."""
        line = pg.InfiniteLine(
            pos=t, angle=90,
            pen=pg.mkPen("#F6B73C", width=2),
        )
        self._plot.addItem(line)
        self._trigger_lines.append(line)


class RollingChannelMonitor(QtWidgets.QWidget):
    """Rolling live grid view of multiple channels, raw post-scale values.

    Maintains an N-sample circular buffer per monitored channel and renders
    each channel in its own panel with independent Y autoscaling. The X
    axis on every panel is locked to the rolling window ending at the
    most recently received sample.
    """

    def __init__(
        self,
        channels: Sequence[tuple[int, str]],
        sample_rate_hz: int,
        title: str,
        rolling_seconds: float = _LIVE_MONITOR_ROLLING_SECONDS,
        grid_columns: int = _LIVE_MONITOR_GRID_COLUMNS,
        show_filters: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(title)
        self.resize(1100, 520)

        self._channel_indices: list[int] = [idx for idx, _ in channels]
        self._channel_labels: list[str] = [label for _, label in channels]
        self._sample_rate_hz = sample_rate_hz
        self._rolling_seconds = rolling_seconds
        self._max_samples = max(1, int(sample_rate_hz * rolling_seconds))

        n_ch = len(channels)
        self._times = np.empty(self._max_samples, dtype=np.float64)
        self._values = np.empty((self._max_samples, n_ch), dtype=np.float64)
        self._write_pos = 0
        self._filled = 0
        self._manual_y: list[bool] = [False] * n_ch

        self._filter_sos = design_emg_bandpass(sample_rate_hz) if show_filters else None
        self._notch_sos = design_emg_notch(sample_rate_hz) if show_filters else None
        self._filter_enabled = False
        self._notch_enabled = False

        self._plot_widgets: list[pg.PlotWidget] = []
        self._curves: list[pg.PlotDataItem] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(8)
        self._reset_y_button = QtWidgets.QPushButton("Reset Auto Y")
        self._reset_y_button.setToolTip(
            "Re-enable Y autoscale on all panels (cancels manual mouse-wheel zoom)."
        )
        self._reset_y_button.clicked.connect(self._reset_manual_y)
        toolbar.addWidget(self._reset_y_button)
        if show_filters:
            self._filter_button = QtWidgets.QPushButton(
                f"Bandpass {EMG_BANDPASS_LOW_HZ:.0f}–{EMG_BANDPASS_HIGH_HZ:.0f} Hz"
            )
            self._filter_button.setObjectName("filterButton")
            self._filter_button.setCheckable(True)
            self._filter_button.setEnabled(self._filter_sos is not None)
            self._filter_button.toggled.connect(self._on_filter_toggled)
            toolbar.addWidget(self._filter_button)

            self._notch_button = QtWidgets.QPushButton(
                f"Notch {EMG_NOTCH_HZ:.0f} Hz + harmonics"
            )
            self._notch_button.setObjectName("notchButton")
            self._notch_button.setCheckable(True)
            self._notch_button.setEnabled(self._notch_sos is not None)
            self._notch_button.toggled.connect(self._on_notch_toggled)
            toolbar.addWidget(self._notch_button)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        grid_container = QtWidgets.QWidget()
        grid_container.setStyleSheet("background: #D2DDEA;")
        grid = QtWidgets.QGridLayout(grid_container)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        n_cols = max(1, min(grid_columns, n_ch)) if n_ch > 0 else 1
        n_rows = (n_ch + n_cols - 1) // n_cols if n_ch > 0 else 0
        last_row = n_rows - 1

        for local_idx, label in enumerate(self._channel_labels):
            row = local_idx // n_cols
            col = local_idx % n_cols

            panel = pg.PlotWidget(background="#F4F7FB")
            panel.setMenuEnabled(False)
            panel.setMouseEnabled(x=False, y=True)
            panel.showGrid(x=True, y=True, alpha=0.18)
            panel.setFrameShape(QtWidgets.QFrame.NoFrame)
            panel.setTitle(f"<b>{label}</b>", size="14pt", color="#2E3A46")
            panel.plotItem.titleLabel.item.setTextWidth(-1)
            panel.getViewBox().sigRangeChangedManually.connect(
                lambda _mask, i=local_idx: self._on_manual_y_change(i)
            )

            if row == last_row:
                panel.getAxis("bottom").setHeight(28)
                panel.getAxis("bottom").setTickFont(
                    QtGui.QFont("Segoe UI", 11, QtGui.QFont.Bold)
                )
                panel.setLabel("bottom", "Time (s)")
            else:
                panel.getAxis("bottom").setHeight(0)
                panel.getAxis("bottom").setStyle(showValues=False)

            if col == 0:
                panel.getAxis("left").setWidth(48)
                panel.setLabel("left", "Signal", units="a.u.")
            else:
                panel.getAxis("left").setWidth(0)
                panel.getAxis("left").setStyle(showValues=False)

            pen = pg.mkPen(FINGER_COLORS[local_idx % len(FINGER_COLORS)], width=1.5)
            curve = panel.plot([], [], pen=pen)

            self._plot_widgets.append(panel)
            self._curves.append(curve)
            grid.addWidget(panel, row, col)

        if self._plot_widgets:
            reference = self._plot_widgets[0]
            for panel in self._plot_widgets[1:]:
                panel.setXLink(reference)

        root.addWidget(grid_container, stretch=1)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Rejects the close event and hides the window instead."""
        self.hide()
        event.ignore()

    def push_batch(self, timestamps: np.ndarray, signals: np.ndarray) -> None:
        """Appends a batch into the rolling buffer and refreshes all curves."""
        n = len(timestamps)
        if n == 0 or not self._plot_widgets:
            return

        selected = signals[:, self._channel_indices]

        end = self._write_pos + n
        if end <= self._max_samples:
            self._times[self._write_pos:end] = timestamps
            self._values[self._write_pos:end, :] = selected
        else:
            first = self._max_samples - self._write_pos
            self._times[self._write_pos:] = timestamps[:first]
            self._values[self._write_pos:, :] = selected[:first, :]
            rest = n - first
            self._times[:rest] = timestamps[first:]
            self._values[:rest, :] = selected[first:, :]
        self._write_pos = end % self._max_samples
        self._filled = min(self._max_samples, self._filled + n)

        if self._filled < self._max_samples:
            t_out = self._times[:self._filled]
            v_out = self._values[:self._filled, :]
        else:
            rp = self._write_pos
            t_out = np.concatenate((self._times[rp:], self._times[:rp]))
            v_out = np.concatenate(
                (self._values[rp:, :], self._values[:rp, :]), axis=0
            )

        if self._filter_enabled and self._filter_sos is not None:
            v_out = sosfiltfilt(self._filter_sos, v_out, axis=0)
        if self._notch_enabled and self._notch_sos is not None:
            v_out = sosfiltfilt(self._notch_sos, v_out, axis=0)

        if t_out.size == 0:
            return

        t_now = float(t_out[-1])
        x_min = t_now - self._rolling_seconds
        x_max = t_now

        for local_idx, curve in enumerate(self._curves):
            col = v_out[:, local_idx]
            curve.setData(t_out, col)
            panel = self._plot_widgets[local_idx]
            panel.setXRange(x_min, x_max, padding=0.0)
            if self._manual_y[local_idx]:
                continue
            c_min = float(np.min(col))
            c_max = float(np.max(col))
            span = c_max - c_min
            pad = max(0.01, span * 0.08) if span > 0 else 0.5
            panel.setYRange(c_min - pad, c_max + pad, padding=0.0)

    def _on_filter_toggled(self, checked: bool) -> None:
        self._filter_enabled = checked

    def _on_notch_toggled(self, checked: bool) -> None:
        self._notch_enabled = checked

    def _on_manual_y_change(self, local_idx: int) -> None:
        """Marks a panel as user-zoomed so push_batch stops autoscaling its Y axis."""
        self._manual_y[local_idx] = True

    def _reset_manual_y(self) -> None:
        """Re-enables Y autoscale on every panel by clearing the manual-zoom flags."""
        for i in range(len(self._manual_y)):
            self._manual_y[i] = False


class EmgMonitorWindow(QtWidgets.QWidget):
    """Per-event view of EMG channels with P2P and onset overlays.

    Receives the same CapturedWindow as the main finger view, but renders
    only the channels declared `kind = "emg"` in the channels TOML. No
    baseline/MVC normalization — values are the scaled signal as captured.
    """

    def __init__(
        self,
        emg_channels: Sequence[tuple[int, str]],
        sample_rate_hz: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("EMG Monitor")
        self.resize(1100, 360)

        self._channel_indices: list[int] = [idx for idx, _ in emg_channels]
        self._channel_labels: list[str] = [label for _, label in emg_channels]
        self._sample_rate_hz = sample_rate_hz
        self._global_scale: bool = True
        self._post_skip_ms: float = 0.0
        self._last_capture: CapturedWindow | None = None
        self._filter_sos = design_emg_bandpass(sample_rate_hz)
        self._filter_enabled: bool = False
        self._notch_sos = design_emg_notch(sample_rate_hz)
        self._notch_enabled: bool = False

        self._plot_widgets: list[pg.PlotWidget] = []
        self._curves: list[pg.PlotDataItem] = []
        self._onset_lines: list[pg.InfiniteLine] = []
        self._skip_regions: list[pg.LinearRegionItem] = []
        self._info_labels: list[QtWidgets.QLabel] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(8)
        self._scale_button = QtWidgets.QPushButton("Global Scale")
        self._scale_button.setObjectName("scaleButton")
        self._scale_button.setCheckable(True)
        self._scale_button.toggled.connect(self._on_scale_toggled)
        toolbar.addWidget(self._scale_button)

        self._filter_button = QtWidgets.QPushButton(
            f"Bandpass {EMG_BANDPASS_LOW_HZ:.0f}–{EMG_BANDPASS_HIGH_HZ:.0f} Hz"
        )
        self._filter_button.setObjectName("filterButton")
        self._filter_button.setCheckable(True)
        if self._filter_sos is None:
            self._filter_button.setEnabled(False)
            self._filter_button.setToolTip(
                f"Sample rate {sample_rate_hz} Hz is too low for a "
                f"{EMG_BANDPASS_HIGH_HZ:.0f} Hz upper cutoff."
            )
        else:
            self._filter_button.setToolTip(
                f"Zero-phase 4th-order Butterworth bandpass "
                f"({EMG_BANDPASS_LOW_HZ:.0f}–{EMG_BANDPASS_HIGH_HZ:.0f} Hz)."
            )
        self._filter_button.toggled.connect(self._on_filter_toggled)
        toolbar.addWidget(self._filter_button)

        self._notch_button = QtWidgets.QPushButton(
            f"Notch {EMG_NOTCH_HZ:.0f} Hz + harmonics"
        )
        self._notch_button.setObjectName("notchButton")
        self._notch_button.setCheckable(True)
        if self._notch_sos is None:
            self._notch_button.setEnabled(False)
            self._notch_button.setToolTip(
                f"Sample rate {sample_rate_hz} Hz is too low for a "
                f"{EMG_NOTCH_HZ:.0f} Hz notch."
            )
        else:
            self._notch_button.setToolTip(
                f"Zero-phase IIR notch at {EMG_NOTCH_HZ:.0f} Hz and harmonics "
                f"present below Nyquist ({sample_rate_hz / 2:.0f} Hz)."
            )
        self._notch_button.toggled.connect(self._on_notch_toggled)
        toolbar.addWidget(self._notch_button)

        skip_label = QtWidgets.QLabel("Skip after trigger (ms):")
        skip_label.setStyleSheet("color: #334155; font-size: 12px; font-weight: 600;")
        self._skip_spin = QtWidgets.QDoubleSpinBox()
        self._skip_spin.setRange(0.0, 1000.0)
        self._skip_spin.setDecimals(1)
        self._skip_spin.setSingleStep(1.0)
        self._skip_spin.setValue(self._post_skip_ms)
        self._skip_spin.setSuffix(" ms")
        self._skip_spin.setToolTip(
            "Excludes the first N ms after the trigger from P2P and onset "
            "(useful to ignore TMS or stimulator artifact)."
        )
        self._skip_spin.valueChanged.connect(self._on_skip_changed)
        toolbar.addWidget(skip_label)
        toolbar.addWidget(self._skip_spin)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        grid_container = QtWidgets.QWidget()
        grid_container.setStyleSheet("background: #D2DDEA;")
        grid = QtWidgets.QGridLayout(grid_container)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(2)
        grid.setVerticalSpacing(2)

        for local_idx, label in enumerate(self._channel_labels):
            panel = pg.PlotWidget(background="#F4F7FB")
            panel.setMenuEnabled(False)
            panel.setMouseEnabled(x=False, y=False)
            panel.showGrid(x=True, y=True, alpha=0.18)
            panel.setFrameShape(QtWidgets.QFrame.NoFrame)
            panel.setTitle(f"<b>{label}</b>", size="16pt", color="#2E3A46")
            panel.plotItem.titleLabel.item.setTextWidth(-1)
            panel.getAxis("bottom").setHeight(32)
            panel.getAxis("bottom").setTickFont(
                QtGui.QFont("Segoe UI", 13, QtGui.QFont.Bold)
            )
            if local_idx == 0:
                panel.getAxis("left").setWidth(48)
            else:
                panel.getAxis("left").setWidth(0)
                panel.getAxis("left").setStyle(showValues=False)
            panel.setLabel("left", "EMG", units="a.u.")
            panel.setLabel("bottom", "Time (s)")

            trigger_line = pg.InfiniteLine(
                pos=0.0, angle=90,
                pen=pg.mkPen("#E63946", width=1.5, style=QtCore.Qt.DashLine),
            )
            panel.addItem(trigger_line)

            skip_region = pg.LinearRegionItem(
                values=(0.0, 0.0),
                movable=False,
                brush=pg.mkBrush(230, 57, 70, 60),
                pen=pg.mkPen(None),
            )
            skip_region.setZValue(-10)
            skip_region.setVisible(False)
            panel.addItem(skip_region)
            self._skip_regions.append(skip_region)

            pen = pg.mkPen(FINGER_COLORS[local_idx % len(FINGER_COLORS)], width=2.0)
            curve = panel.plot([], [], pen=pen)

            onset_line = pg.InfiniteLine(
                pos=0.0, angle=90,
                pen=pg.mkPen("#2A9D8F", width=2.0),
            )
            onset_line.setVisible(False)
            panel.addItem(onset_line)

            info_label = QtWidgets.QLabel("P2P: —\nOnset: —", panel)
            info_label.setStyleSheet(
                "font-weight: bold; font-size: 22px; color: #1E2933;"
                " background: rgba(244,247,251,200); padding: 3px 6px; border-radius: 4px;"
            )
            info_label.adjustSize()
            info_label.move(4, 36)
            info_label.raise_()

            self._plot_widgets.append(panel)
            self._curves.append(curve)
            self._onset_lines.append(onset_line)
            self._info_labels.append(info_label)
            grid.addWidget(panel, 0, local_idx)

        if self._plot_widgets:
            reference = self._plot_widgets[0]
            for panel in self._plot_widgets[1:]:
                panel.setXLink(reference)

        root.addWidget(grid_container, stretch=1)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Rejects the close event and hides the window instead."""
        self.hide()
        event.ignore()

    def _on_scale_toggled(self, checked: bool) -> None:
        """Toggles between global and per-channel vertical scaling."""
        self._global_scale = not checked
        self._scale_button.setText("Per Channel Scale" if checked else "Global Scale")
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_skip_changed(self, value: float) -> None:
        """Updates the artifact pre-skip duration and refreshes the current view."""
        self._post_skip_ms = float(value)
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_filter_toggled(self, checked: bool) -> None:
        """Enables or disables the EMG bandpass and refreshes the current view."""
        self._filter_enabled = checked
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_notch_toggled(self, checked: bool) -> None:
        """Enables or disables the power-line notch and refreshes the current view."""
        self._notch_enabled = checked
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def update_capture(self, captured: CapturedWindow) -> None:
        """Updates the EMG panels with data from a new CapturedWindow."""
        self._last_capture = captured
        if not self._plot_widgets:
            return

        sig = captured.batch.signals
        relative_time = (
            captured.batch.timestamps
            - captured.batch.timestamps[captured.trigger_sample]
        )

        skip_samples = max(
            0, int(round(self._post_skip_ms / 1000.0 * self._sample_rate_hz))
        )
        measure_start = captured.trigger_sample + skip_samples
        skip_seconds = self._post_skip_ms / 1000.0
        for region in self._skip_regions:
            if skip_samples > 0:
                region.setRegion((0.0, skip_seconds))
                region.setVisible(True)
            else:
                region.setVisible(False)

        bandpass_active = self._filter_enabled and self._filter_sos is not None
        notch_active = self._notch_enabled and self._notch_sos is not None

        emg_columns: list[np.ndarray] = []
        for local_idx, curve in enumerate(self._curves):
            ch_idx = self._channel_indices[local_idx]
            emg = sig[:, ch_idx]
            if bandpass_active:
                emg = sosfiltfilt(self._filter_sos, emg, axis=0)
            if notch_active:
                emg = sosfiltfilt(self._notch_sos, emg, axis=0)
            emg_columns.append(emg)
            curve.setData(relative_time, emg)

            onset_ms = detect_onset(
                emg,
                captured.trigger_sample,
                self._sample_rate_hz,
                post_skip_samples=skip_samples,
            )
            onset_line = self._onset_lines[local_idx]
            if onset_ms is not None:
                onset_line.setPos(onset_ms / 1000.0)
                onset_line.setVisible(True)
            else:
                onset_line.setVisible(False)

            measured = emg[measure_start:]
            p2p = float(np.ptp(measured)) if measured.size > 0 else 0.0
            p2p_str = f"P2P: {p2p:.2f}"
            onset_str = (
                f"Onset: {onset_ms:.0f} ms" if onset_ms is not None else "Onset: —"
            )
            lbl = self._info_labels[local_idx]
            lbl.setText(f"{p2p_str}\n{onset_str}")
            lbl.adjustSize()

        if relative_time.size == 0:
            return
        x_min = float(relative_time[0])
        x_max = float(relative_time[-1])

        if self._global_scale:
            stacked = np.stack(emg_columns, axis=1)
            y_min = float(np.min(stacked))
            y_max = float(np.max(stacked))
            y_span = y_max - y_min
            y_padding = max(0.01, y_span * 0.08)
            for panel in self._plot_widgets:
                panel.setXRange(x_min, x_max, padding=0.0)
                panel.setYRange(y_min - y_padding, y_max + y_padding, padding=0.0)
        else:
            for local_idx, panel in enumerate(self._plot_widgets):
                col = emg_columns[local_idx]
                f_min = float(np.min(col))
                f_max = float(np.max(col))
                f_span = f_max - f_min
                f_padding = max(0.01, f_span * 0.08)
                panel.setXRange(x_min, x_max, padding=0.0)
                panel.setYRange(f_min - f_padding, f_max + f_padding, padding=0.0)


class _OnsetLine(pg.InfiniteLine):
    """Movable vertical onset marker with a right-click 'Reset to auto' menu."""

    reset_requested = QtCore.pyqtSignal()

    def mouseClickEvent(self, ev) -> None:
        """Shows a right-click context menu to reset the onset marker."""
        if ev.button() == QtCore.Qt.RightButton:
            menu = QtWidgets.QMenu()
            action = menu.addAction("Reset to auto-detected")
            if menu.exec_(QtGui.QCursor.pos()) == action:
                self.reset_requested.emit()
            ev.accept()
        else:
            super().mouseClickEvent(ev)


class QuattrocentoMainWindow(QtWidgets.QMainWindow):
    """Main visualization window for trigger-captured force events."""

    previous_requested = QtCore.pyqtSignal()
    next_requested = QtCore.pyqtSignal()
    baseline_toggled = QtCore.pyqtSignal(bool)
    peak_toggled = QtCore.pyqtSignal(bool)
    empty_toggled = QtCore.pyqtSignal(bool)
    save_calibration_requested = QtCore.pyqtSignal(str)
    load_calibration_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        channel_labels: Mapping[int, str],
        trigger_channel: int,
        sample_rate_hz: int,
        trigger_threshold: float,
        emg_channels: Sequence[tuple[int, str]] = (),
    ) -> None:
        super().__init__()

        emg_indices: set[int] = {idx for idx, _ in emg_channels}

        # Ordered non-trigger, non-EMG finger channels, sorted by channel index.
        finger_channels = sorted(
            (idx, label)
            for idx, label in channel_labels.items()
            if idx != trigger_channel and idx not in emg_indices
        )
        if len(finger_channels) != _EXPECTED_FINGER_COUNT:
            raise ValueError(
                f"Expected exactly {_EXPECTED_FINGER_COUNT} labeled non-trigger "
                f"channels, got {len(finger_channels)}. "
                "Update your channels file."
            )
        self._finger_channel_indices: list[int] = [idx for idx, _ in finger_channels]
        self._finger_labels: tuple[str, ...] = tuple(label for _, label in finger_channels)

        # Bar display order (mirrored hand layout if labels parse correctly).
        bar_local_indices, bar_labels = _build_mirrored_bar_layout(self._finger_labels)
        self._bar_display_indices: tuple[int, ...] = bar_local_indices
        self._bar_display_labels: tuple[str, ...] = bar_labels
        self._bar_x = np.arange(len(bar_labels), dtype=np.float64)
        self._bar_item: pg.BarGraphItem | None = None

        self._capture_count_label = QtWidgets.QLabel("Events: 0")
        self._acquisition_label = QtWidgets.QLabel("State: Waiting for trigger")
        self._last_trigger_label = QtWidgets.QLabel("Last trigger: -")
        self._sampling_label = QtWidgets.QLabel("Sample rate: -")
        self._cal_status_label = QtWidgets.QLabel("Cal: Rest ✗ | MVC ✗")
        self._event_position_label = QtWidgets.QLabel("Viewing: -/-")
        self._previous_button = QtWidgets.QPushButton("< Prev")
        self._next_button = QtWidgets.QPushButton("Next >")
        self._baseline_action = QtWidgets.QAction("Calibrate Rest", self)
        self._baseline_action.setCheckable(True)
        self._peak_action = QtWidgets.QAction("Calibrate MVC", self)
        self._peak_action.setCheckable(True)
        self._empty_action = QtWidgets.QAction("Calibrate Zero", self)
        self._empty_action.setCheckable(True)
        self._save_cal_action = QtWidgets.QAction("Save Calibration", self)
        self._load_cal_action = QtWidgets.QAction("Load Calibration", self)
        _cal_menu = QtWidgets.QMenu(self)
        _cal_menu.addAction(self._baseline_action)
        _cal_menu.addAction(self._peak_action)
        _cal_menu.addAction(self._empty_action)
        _cal_menu.addSeparator()
        _cal_menu.addAction(self._save_cal_action)
        _cal_menu.addAction(self._load_cal_action)
        self._cal_menu_button = QtWidgets.QToolButton(self)
        self._cal_menu_button.setText("Cal ▾")
        self._cal_menu_button.setMenu(_cal_menu)
        self._cal_menu_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self._pretrigger_baseline_button = QtWidgets.QPushButton("Rest from Pre-Trig")
        self._pretrigger_baseline_button.setObjectName("preTrigRestButton")
        self._pretrigger_baseline_button.setCheckable(True)
        self._use_pretrigger_baseline: bool = False
        self._hook_controls_layout: QtWidgets.QHBoxLayout | None = None
        self._hooks_menu_button: QtWidgets.QToolButton | None = None
        self._raw_plot_widgets: list[pg.PlotWidget] = []
        self._raw_curves: list[pg.PlotDataItem] = []
        self._raw_max_markers: list[pg.PlotDataItem] = []
        self._mvc_ref_lines: list[pg.InfiniteLine] = []
        self._zero_ref_lines: list[pg.InfiniteLine] = []
        self._rest_ref_lines: list[pg.InfiniteLine] = []
        self._previous_shortcut: QtWidgets.QShortcut | None = None
        self._next_shortcut: QtWidgets.QShortcut | None = None
        self._sample_rate_hz = sample_rate_hz
        self._global_scale: bool = True
        self._show_mvc: bool = True
        self._show_rest_ref: bool = False
        self._show_zero_ref: bool = False
        self._last_capture: CapturedWindow | None = None
        self._finger_info_labels: list[QtWidgets.QLabel] = []
        self._onset_lines: list[_OnsetLine] = []
        self._auto_onset_ms: list[float | None] = [None] * _EXPECTED_FINGER_COUNT
        self._finger_p2p_strs: list[str] = ["P2P: —"] * _EXPECTED_FINGER_COUNT
        self._rest_line_button = QtWidgets.QPushButton("Rest Line")
        self._rest_line_button.setObjectName("restLineButton")
        self._rest_line_button.setCheckable(True)
        self._zero_line_button = QtWidgets.QPushButton("Zero Line")
        self._zero_line_button.setObjectName("zeroLineButton")
        self._zero_line_button.setCheckable(True)
        self._scale_button = QtWidgets.QPushButton("Global Scale")
        self._scale_button.setObjectName("scaleButton")
        self._scale_button.setCheckable(True)
        self._display_mode_button = QtWidgets.QPushButton("% MVC")
        self._display_mode_button.setObjectName("displayModeButton")
        self._display_mode_button.setCheckable(True)
        self._trigger_monitor = TriggerMonitorWindow(
            sample_rate_hz=sample_rate_hz,
            parent=self,
        )
        self._emg_button = QtWidgets.QPushButton("EMG Monitor")
        self._emg_window: EmgMonitorWindow | None = (
            EmgMonitorWindow(
                emg_channels=emg_channels,
                sample_rate_hz=sample_rate_hz,
                parent=self,
            )
            if emg_channels
            else None
        )
        self._emg_button.setEnabled(self._emg_window is not None)

        finger_channel_pairs = list(zip(self._finger_channel_indices, self._finger_labels))
        self._force_live_monitor = RollingChannelMonitor(
            channels=finger_channel_pairs,
            sample_rate_hz=sample_rate_hz,
            title="Force Live Monitor",
            parent=self,
        )
        self._emg_live_monitor: RollingChannelMonitor | None = (
            RollingChannelMonitor(
                channels=list(emg_channels),
                sample_rate_hz=sample_rate_hz,
                title="EMG Live Monitor",
                show_filters=True,
                parent=self,
            )
            if emg_channels
            else None
        )
        self._trigger_monitor_action = QtWidgets.QAction("Trigger Monitor", self)
        self._force_live_action = QtWidgets.QAction("Force Live", self)
        self._emg_live_action = QtWidgets.QAction("EMG Live", self)
        self._emg_live_action.setEnabled(self._emg_live_monitor is not None)
        _live_menu = QtWidgets.QMenu(self)
        _live_menu.addAction(self._trigger_monitor_action)
        _live_menu.addAction(self._force_live_action)
        _live_menu.addAction(self._emg_live_action)
        self._live_menu_button = QtWidgets.QToolButton(self)
        self._live_menu_button.setText("Live ▾")
        self._live_menu_button.setMenu(_live_menu)
        self._live_menu_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self._apply_palette()
        self._build_layout()
        self._install_navigation_shortcuts()
        self.set_event_navigation(current_index=None, total_events=0)
        self.set_calibration_status(baseline_done=False, peak_done=False)
        self._apply_unit_labels("a.u.")

    def _apply_palette(self) -> None:
        """Applies the global pyqtgraph config options and QSS stylesheet."""
        pg.setConfigOptions(antialias=True, foreground="#27313D", background="#F4F7FB")
        self.setStyleSheet(
            """
            QMainWindow {
                background: #F4F7FB;
            }
            QLabel#title {
                color: #1E2933;
                font-size: 18px;
                font-weight: 600;
                letter-spacing: 0.2px;
            }
            QLabel[kind="chip"] {
                background: #E5EDF6;
                border: 1px solid #D2DDEA;
                border-radius: 10px;
                color: #334155;
                font-size: 12px;
                font-weight: 500;
                padding: 5px 9px;
            }
            QLabel#eventPosition {
                color: #334155;
                font-size: 12px;
                font-weight: 600;
                padding-left: 8px;
            }
            QPushButton, QToolButton {
                background: #DCE7F4;
                border: 1px solid #BFD0E1;
                border-radius: 8px;
                color: #1F3242;
                font-size: 12px;
                font-weight: 600;
                padding: 5px 12px;
            }
            QPushButton:disabled, QToolButton:disabled {
                background: #EEF3F9;
                border-color: #DCE5EF;
                color: #9AABBE;
            }
            QPushButton:hover:!disabled, QToolButton:hover:!disabled {
                background: #CCDDF0;
            }
            QToolButton::menu-indicator {
                image: none;
            }
            QPushButton#restButton:checked {
                background: #0077B6;
                border-color: #005A8C;
                color: white;
            }
            QPushButton#mvcButton:checked {
                background: #E63946;
                border-color: #D62828;
                color: white;
            }
            QPushButton#zeroButton:checked {
                background: #4A5568;
                border-color: #2D3748;
                color: white;
            }
            QPushButton#hookToggle:checked {
                background: #2A9D8F;
                border-color: #1E7268;
                color: white;
            }
            QPushButton#scaleButton:checked {
                background: #5E60CE;
                border-color: #4B4DBF;
                color: white;
            }
            QPushButton#displayModeButton:checked {
                background: #E07A10;
                border-color: #B85E00;
                color: white;
            }
            QPushButton#preTrigRestButton:checked {
                background: #2E8B57;
                border-color: #1F6B40;
                color: white;
            }
            QPushButton#restLineButton:checked {
                background: #0077B6;
                border-color: #005A8C;
                color: white;
            }
            QPushButton#zeroLineButton:checked {
                background: #4A5568;
                border-color: #2D3748;
                color: white;
            }
            QPushButton#filterButton:checked {
                background: #2A9D8F;
                border-color: #1E7268;
                color: white;
            }
            QPushButton#notchButton:checked {
                background: #8E44AD;
                border-color: #6C3483;
                color: white;
            }
            """
        )

    def _build_layout(self) -> None:
        """Constructs and wires all child widgets into the main window layout."""
        self.setWindowTitle("Quattrocento Triggered Force Application")
        self.resize(1140, 760)
        self.setFocusPolicy(Qt.StrongFocus)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QtWidgets.QVBoxLayout(central_widget)
        root_layout.setContentsMargins(14, 12, 14, 12)
        root_layout.setSpacing(10)

        title = QtWidgets.QLabel("Triggered Force Analysis")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        root_layout.addWidget(title)

        chips_layout = QtWidgets.QHBoxLayout()
        chips_layout.setSpacing(8)
        for chip in (
            self._acquisition_label,
            self._capture_count_label,
            self._last_trigger_label,
            self._sampling_label,
            self._cal_status_label,
        ):
            chip.setProperty("kind", "chip")
            chips_layout.addWidget(chip)
        chips_layout.addStretch(1)
        root_layout.addLayout(chips_layout)

        navigation_layout = QtWidgets.QHBoxLayout()
        navigation_layout.setSpacing(8)
        self._event_position_label.setObjectName("eventPosition")
        self._previous_button.clicked.connect(self.previous_requested.emit)
        self._next_button.clicked.connect(self.next_requested.emit)
        self._baseline_action.toggled.connect(self.baseline_toggled.emit)
        self._pretrigger_baseline_button.toggled.connect(self._on_pretrigger_baseline_toggled)
        self._peak_action.toggled.connect(self.peak_toggled.emit)
        self._empty_action.toggled.connect(self.empty_toggled.emit)
        self._save_cal_action.triggered.connect(self._on_save_cal_clicked)
        self._load_cal_action.triggered.connect(self._on_load_cal_clicked)
        self._rest_line_button.toggled.connect(self._on_rest_line_toggled)
        self._zero_line_button.toggled.connect(self._on_zero_line_toggled)
        self._scale_button.toggled.connect(self._on_scale_toggled)
        self._display_mode_button.toggled.connect(self._on_display_mode_toggled)
        self._trigger_monitor_action.triggered.connect(self._toggle_trigger_monitor)
        self._emg_button.clicked.connect(self._toggle_emg_monitor)
        self._force_live_action.triggered.connect(self._toggle_force_live)
        self._emg_live_action.triggered.connect(self._toggle_emg_live)
        self._previous_button.setToolTip("Previous event (Left Arrow)")
        self._next_button.setToolTip("Next event (Right Arrow)")
        self._pretrigger_baseline_button.setToolTip(
            "Use the pre-trigger window of each displayed event as its display baseline"
        )
        self._hook_controls_layout = QtWidgets.QHBoxLayout()
        self._hook_controls_layout.setSpacing(8)
        navigation_layout.addWidget(self._previous_button)
        navigation_layout.addWidget(self._next_button)
        navigation_layout.addWidget(self._event_position_label)
        navigation_layout.addStretch(1)
        navigation_layout.addWidget(self._build_legend())
        navigation_layout.addWidget(self._display_mode_button)
        navigation_layout.addWidget(self._scale_button)
        navigation_layout.addWidget(self._rest_line_button)
        navigation_layout.addWidget(self._zero_line_button)
        navigation_layout.addWidget(self._pretrigger_baseline_button)
        navigation_layout.addWidget(self._cal_menu_button)
        navigation_layout.addWidget(self._emg_button)
        navigation_layout.addWidget(self._live_menu_button)
        navigation_layout.addLayout(self._hook_controls_layout)
        root_layout.addLayout(navigation_layout)

        self.range_plot = pg.PlotWidget()
        self._style_range_plot()
        root_layout.addWidget(self.range_plot, stretch=2)

        raw_grid_container = QtWidgets.QWidget()
        raw_grid_container.setStyleSheet("background: #D2DDEA;")
        raw_grid_layout = QtWidgets.QGridLayout(raw_grid_container)
        raw_grid_layout.setContentsMargins(2, 2, 2, 2)
        raw_grid_layout.setHorizontalSpacing(2)
        raw_grid_layout.setVerticalSpacing(2)
        self._build_raw_grid(raw_grid_layout)
        raw_grid_layout.setRowStretch(0, 1)
        raw_grid_layout.setRowStretch(1, 1)
        root_layout.addWidget(raw_grid_container, stretch=5)

    def _style_range_plot(self) -> None:
        """Configures visual appearance and initial state of the range bar chart."""
        self.range_plot.setTitle("Force Range by Finger (max - min)")
        self.range_plot.setMenuEnabled(False)
        self.range_plot.showGrid(x=False, y=True, alpha=0.2)
        self.range_plot.setMouseEnabled(x=False, y=False)
        self.range_plot.setLabel("left", "Range", units="a.u.")
        self.range_plot.getAxis("bottom").setTicks(
            [list(zip(self._bar_x.tolist(), self._bar_display_labels))]
        )
        self.range_plot.getAxis("bottom").setTickFont(
            QtGui.QFont("Segoe UI", 15, QtGui.QFont.Bold)
        )
        self.range_plot.setYRange(0.0, 1.0, padding=0.0)
        self.range_plot.setXRange(-0.6, len(self._bar_display_labels) - 0.4, padding=0.0)
        self.range_plot.addItem(pg.InfiniteLine(
            pos=1.0, angle=0,
            pen=pg.mkPen("#94A3B8", width=1.0, style=QtCore.Qt.DashLine),
        ))
        self._draw_range_bars(np.zeros(len(self._bar_display_labels), dtype=np.float64))

    def _build_raw_grid(self, grid_layout: QtWidgets.QGridLayout) -> None:
        """Creates and inserts one PlotWidget per finger into the grid layout."""
        n_rows = (_EXPECTED_FINGER_COUNT + _RAW_GRID_COLUMNS - 1) // _RAW_GRID_COLUMNS
        last_row = n_rows - 1

        for finger_idx, finger_name in enumerate(self._finger_labels):
            row = finger_idx // _RAW_GRID_COLUMNS
            col = finger_idx % _RAW_GRID_COLUMNS

            panel = pg.PlotWidget(background="#F4F7FB")
            panel.setMenuEnabled(False)
            panel.setMouseEnabled(x=False, y=False)
            panel.showGrid(x=True, y=True, alpha=0.18)
            panel.setFrameShape(QtWidgets.QFrame.NoFrame)
            panel.setTitle(
                f"<b>{finger_name}</b>", size="16pt", color="#2E3A46"
            )
            panel.plotItem.titleLabel.item.setTextWidth(-1)

            # Bottom axis: tick values only on last row, hidden otherwise.
            if row == last_row:
                panel.getAxis("bottom").setHeight(32)
                panel.getAxis("bottom").setTickFont(
                    QtGui.QFont("Segoe UI", 13, QtGui.QFont.Bold)
                )
            else:
                panel.getAxis("bottom").setHeight(0)
                panel.getAxis("bottom").setStyle(showValues=False)

            # Left axis: tick values only on first column.
            if col == 0:
                panel.getAxis("left").setWidth(40)
            else:
                panel.getAxis("left").setWidth(0)
                panel.getAxis("left").setStyle(showValues=False)

            trigger_line = pg.InfiniteLine(
                pos=0.0, angle=90,
                pen=pg.mkPen("#E63946", width=1.5, style=QtCore.Qt.DashLine),
            )
            panel.addItem(trigger_line)

            # MVC reference line (peak = 100%), initially hidden.
            pen_ref = pg.mkPen("#64748B", width=1.0, style=QtCore.Qt.DotLine)
            peak_ref = pg.InfiniteLine(angle=0, pos=0, pen=pen_ref)
            peak_ref.setVisible(False)
            panel.addItem(peak_ref)
            self._mvc_ref_lines.append(peak_ref)

            # Zero (no-contact) reference line, initially hidden.
            pen_zero = pg.mkPen("#4A5568", width=1.2, style=QtCore.Qt.DashLine)
            zero_ref = pg.InfiniteLine(
                angle=0, pos=0, pen=pen_zero,
                label="zero", labelOpts={"position": 0.04, "color": "#4A5568"},
            )
            zero_ref.setVisible(False)
            panel.addItem(zero_ref)
            self._zero_ref_lines.append(zero_ref)

            # Rest (baseline) reference line, initially hidden.
            pen_rest = pg.mkPen("#0077B6", width=1.2, style=QtCore.Qt.DashLine)
            rest_ref = pg.InfiniteLine(
                angle=0, pos=0, pen=pen_rest,
                label="rest", labelOpts={"position": 0.96, "color": "#0077B6"},
            )
            rest_ref.setVisible(False)
            panel.addItem(rest_ref)
            self._rest_ref_lines.append(rest_ref)

            pen = pg.mkPen(FINGER_COLORS[finger_idx % len(FINGER_COLORS)], width=2.0)
            curve = panel.plot([], [], pen=pen)

            marker = panel.plot(
                [], [], pen=None, symbol="o", symbolSize=8,
                symbolBrush="#E63946", symbolPen="w"
            )

            onset_line = _OnsetLine(
                pos=0.0, angle=90, movable=True,
                pen=pg.mkPen("#2A9D8F", width=2.0),
            )
            onset_line.setVisible(False)
            onset_line.sigPositionChangeFinished.connect(
                lambda line, i=finger_idx: self._on_onset_dragged(i, line.value())
            )
            onset_line.reset_requested.connect(
                lambda i=finger_idx: self._reset_onset(i)
            )
            panel.addItem(onset_line)
            self._onset_lines.append(onset_line)

            info_label = QtWidgets.QLabel("P2P: —\nOnset: —", panel)
            info_label.setStyleSheet(
                "font-weight: bold; font-size: 22px; color: #1E2933;"
                " background: rgba(244,247,251,200); padding: 3px 6px; border-radius: 4px;"
            )
            info_label.adjustSize()
            info_label.move(4, 36)
            info_label.raise_()
            self._finger_info_labels.append(info_label)

            self._raw_plot_widgets.append(panel)
            self._raw_curves.append(curve)
            self._raw_max_markers.append(marker)
            grid_layout.addWidget(panel, row, col)

        if not self._raw_plot_widgets:
            return

        reference = self._raw_plot_widgets[0]
        for panel in self._raw_plot_widgets[1:]:
            panel.setXLink(reference)

    def _build_legend(self) -> QtWidgets.QWidget:
        """Builds the % MVC color-bin legend widget."""
        legend_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(legend_widget)
        layout.setContentsMargins(0, 0, 16, 0)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Peak % MVC:")
        title.setStyleSheet("color: #64748B; font-weight: 600; font-size: 11px;")
        layout.addWidget(title)

        for _threshold, text, color in _MVC_COLOR_BINS:
            layout.addWidget(self._build_legend_item(text, color))

        return legend_widget

    @staticmethod
    def _build_legend_item(text: str, color: str) -> QtWidgets.QWidget:
        """Creates a single color-swatch + label legend entry."""
        item = QtWidgets.QWidget()
        item_layout = QtWidgets.QHBoxLayout(item)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(4)

        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(8, 8)
        swatch.setStyleSheet(f"background-color: {color}; border-radius: 4px;")

        label = QtWidgets.QLabel(text)
        label.setStyleSheet("color: #64748B; font-size: 11px;")

        item_layout.addWidget(swatch)
        item_layout.addWidget(label)
        return item

    def _draw_range_bars(self, heights: np.ndarray) -> None:
        """Updates or creates the bar graph item with the given per-finger heights."""
        if self._bar_item is not None:
            self._bar_item.setOpts(height=heights)
            return
        self._bar_item = pg.BarGraphItem(
            x=self._bar_x,
            height=heights,
            width=0.62,
            pen=pg.mkPen("#35576E"),
            brush=pg.mkBrush("#6FA8DC"),
        )
        self.range_plot.addItem(self._bar_item)

    def _install_navigation_shortcuts(self) -> None:
        """Binds Left/Right arrow keys to the previous/next event signals."""
        self._previous_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(Qt.Key_Left), self
        )
        self._previous_shortcut.setContext(Qt.WindowShortcut)
        self._previous_shortcut.activated.connect(self.previous_requested.emit)

        self._next_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence(Qt.Key_Right), self
        )
        self._next_shortcut.setContext(Qt.WindowShortcut)
        self._next_shortcut.activated.connect(self.next_requested.emit)

    def _toggle_trigger_monitor(self) -> None:
        """Shows the trigger monitor window if hidden, hides it if visible."""
        if self._trigger_monitor.isVisible():
            self._trigger_monitor.hide()
        else:
            self._trigger_monitor.show()
            self._trigger_monitor.raise_()

    def _toggle_emg_monitor(self) -> None:
        """Shows the EMG monitor window if hidden, hides it if visible."""
        if self._emg_window is None:
            return
        if self._emg_window.isVisible():
            self._emg_window.hide()
        else:
            self._emg_window.show()
            self._emg_window.raise_()

    def _toggle_force_live(self) -> None:
        """Shows the live force monitor window if hidden, hides it if visible."""
        if self._force_live_monitor.isVisible():
            self._force_live_monitor.hide()
        else:
            self._force_live_monitor.show()
            self._force_live_monitor.raise_()

    def _toggle_emg_live(self) -> None:
        """Shows the live EMG monitor window if hidden, hides it if visible."""
        if self._emg_live_monitor is None:
            return
        if self._emg_live_monitor.isVisible():
            self._emg_live_monitor.hide()
        else:
            self._emg_live_monitor.show()
            self._emg_live_monitor.raise_()

    def push_live_batch(self, timestamps: np.ndarray, signals: np.ndarray) -> None:
        """Forwards a raw batch to any visible live channel monitors."""
        if self._force_live_monitor.isVisible():
            self._force_live_monitor.push_batch(timestamps, signals)
        if self._emg_live_monitor is not None and self._emg_live_monitor.isVisible():
            self._emg_live_monitor.push_batch(timestamps, signals)

    def push_trigger_batch(
        self,
        timestamps: np.ndarray,
        trigger_col: np.ndarray,
        dc: float | None,
        noise: float,
        effective_threshold: float,
        warmup_remaining: int,
    ) -> None:
        """Forwards live trigger data to the monitor window when it is visible."""
        if self._trigger_monitor.isVisible():
            self._trigger_monitor.push_batch(
                timestamps, trigger_col, dc, noise, effective_threshold, warmup_remaining
            )

    def mark_trigger(self, t: float) -> None:
        """Passes a trigger timestamp to the monitor window when it is visible."""
        if self._trigger_monitor.isVisible():
            self._trigger_monitor.mark_trigger(t)

    def add_hook_toggle(self, name: str, on_toggle: Callable[[bool], None]) -> None:
        """Append a checkable entry to the 'Hooks ▾' menu. Builds the menu
        lazily on first call."""
        if self._hooks_menu_button is None:
            menu = QtWidgets.QMenu(self)
            self._hooks_menu_button = QtWidgets.QToolButton(self)
            self._hooks_menu_button.setText("Hooks ▾")
            self._hooks_menu_button.setMenu(menu)
            self._hooks_menu_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
            self._hook_controls_layout.addWidget(self._hooks_menu_button)
        action = QtWidgets.QAction(name, self)
        action.setCheckable(True)
        action.toggled.connect(on_toggle)
        self._hooks_menu_button.menu().addAction(action)

    def _on_save_cal_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration", "", "NumPy archive (*.npz)"
        )
        if path:
            self.save_calibration_requested.emit(path)

    def _on_load_cal_clicked(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "NumPy archive (*.npz)"
        )
        if path:
            self.load_calibration_requested.emit(path)

    def revert_baseline_button(self) -> None:
        """Unchecks the baseline calibration action without triggering signals."""
        self._baseline_action.blockSignals(True)
        self._baseline_action.setChecked(False)
        self._baseline_action.blockSignals(False)

    def revert_peak_button(self) -> None:
        """Unchecks the peak calibration action without triggering signals."""
        self._peak_action.blockSignals(True)
        self._peak_action.setChecked(False)
        self._peak_action.blockSignals(False)

    def revert_empty_button(self) -> None:
        """Unchecks the zero calibration action without triggering signals."""
        self._empty_action.blockSignals(True)
        self._empty_action.setChecked(False)
        self._empty_action.blockSignals(False)

    def show_error(self, message: str) -> None:
        """Displays a critical error message in a popup dialog."""
        QtWidgets.QMessageBox.warning(self, "Error", message)

    def set_stream_error(self) -> None:
        """Updates the acquisition label to indicate a stream disconnect or error."""
        self._acquisition_label.setText("State: Stream error")

    def set_stream_state(self, sample_rate_hz: int, captures: int, capturing: bool) -> None:
        """Updates the stream status labels in the UI."""
        state_text = "Capturing window..." if capturing else "Waiting for trigger"
        self._acquisition_label.setText(f"State: {state_text}")
        self._capture_count_label.setText(f"Events: {captures}")
        self._sampling_label.setText(f"Sample rate: {sample_rate_hz} Hz")

    def set_last_trigger_now(self) -> None:
        """Records and displays the current wall-clock time as the last trigger time."""
        self._last_trigger_label.setText(
            f"Last trigger: {datetime.now().strftime('%H:%M:%S')}"
        )

    def set_calibration_status(
        self, baseline_done: bool, peak_done: bool, empty_done: bool = False
    ) -> None:
        """Updates the calibration summary label with rest, MVC, and zero status marks."""
        rest_mark = "✓" if baseline_done else "✗"
        mvc_mark = "✓" if peak_done else "✗"
        zero_mark = "✓" if empty_done else "✗"
        self._cal_status_label.setText(
            f"Cal: Rest {rest_mark} | MVC {mvc_mark} | Zero {zero_mark}"
        )

    def show_calibration_report(
        self,
        display_channels: list[tuple[int, str]],
        baseline: np.ndarray | None,
        peak: np.ndarray | None,
        empty: np.ndarray | None = None,
    ) -> None:
        """Pop up a per-channel calibration summary."""
        header = (
            f"{'Channel':<12} {'Zero':>10} {'Rest':>10} {'MVC':>10} {'Span':>10}"
        )
        rows = [header, "-" * len(header)]
        for ch_idx, label in display_channels:
            zero_val = "—" if empty is None else f"{empty[ch_idx]:.2f}"
            rest_val = "—" if baseline is None else f"{baseline[ch_idx]:.2f}"
            mvc_val = "—" if peak is None else f"{peak[ch_idx]:.2f}"
            if baseline is not None and peak is not None:
                span_val = f"{peak[ch_idx] - baseline[ch_idx]:.2f}"
            else:
                span_val = "—"
            rows.append(
                f"{label:<12} {zero_val:>10} {rest_val:>10} {mvc_val:>10} {span_val:>10}"
            )

        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Calibration Report")
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setText("Per-channel calibration values:")
        dialog.setFont(QtGui.QFont("Courier New", 10))
        dialog.setInformativeText("\n".join(rows))
        dialog.exec_()

    def _apply_unit_labels(self, unit: str) -> None:
        """Updates the Y-axis label of the range plot with the current data units."""
        self.range_plot.setLabel("left", "Range", units=unit)

    def set_event_navigation(self, current_index: int | None, total_events: int) -> None:
        """Updates the navigation buttons and status based on the current history position."""
        if total_events <= 0 or current_index is None:
            self._event_position_label.setText("Viewing: -/-")
            self._previous_button.setEnabled(False)
            self._next_button.setEnabled(False)
            return

        self._event_position_label.setText(
            f"Viewing: {current_index + 1}/{total_events}"
        )
        self._previous_button.setEnabled(current_index > 0)
        self._next_button.setEnabled(current_index < (total_events - 1))

    def _on_display_mode_toggled(self, checked: bool) -> None:
        """Toggles between display modes and refreshes the visualization."""
        self._show_mvc = not checked
        self._display_mode_button.setText("Raw" if checked else "% MVC")
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_pretrigger_baseline_toggled(self, checked: bool) -> None:
        """Toggles using the pre-trigger window of each event as its display baseline."""
        self._use_pretrigger_baseline = checked
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_rest_line_toggled(self, checked: bool) -> None:
        """Toggles the rest baseline reference line on the per-finger plots."""
        self._show_rest_ref = checked
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_zero_line_toggled(self, checked: bool) -> None:
        """Toggles the zero (no-contact) reference line on the per-finger plots."""
        self._show_zero_ref = checked
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_scale_toggled(self, checked: bool) -> None:
        """Toggles between global and per-finger Y-axis scaling."""
        self._global_scale = not checked
        self._scale_button.setText("Per Finger Scale" if checked else "Global Scale")
        self._set_per_finger_axes(per_finger=not self._global_scale)
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _set_per_finger_axes(self, *, per_finger: bool) -> None:
        """Show or hide Y-axis tick values on non-first-column panels."""
        for i, panel in enumerate(self._raw_plot_widgets):
            if i % _RAW_GRID_COLUMNS == 0:
                continue
            axis = panel.getAxis("left")
            if per_finger:
                axis.setWidth(40)
                axis.setStyle(showValues=True)
            else:
                axis.setWidth(0)
                axis.setStyle(showValues=False)

    def update_capture(self, captured: CapturedWindow) -> None:
        """Render one captured event in both plots."""
        self._last_capture = captured
        sig = captured.batch.signals

        if self._use_pretrigger_baseline:
            end = max(captured.trigger_sample, 1)
            baseline: np.ndarray | None = np.mean(captured.batch.signals[:end, :], axis=0)
        else:
            baseline = captured.meta.baseline

        has_cal = baseline is not None and captured.meta.peak is not None
        empty = captured.meta.empty
        zero_positions: np.ndarray | None = None  # per-channel zero ref in display units

        # % MVC mode: normalize signals; Raw mode: normalize by global max MVC span if
        # calibrated, otherwise subtract baseline or show raw.
        raw_mvc_span: float | None = None  # global max span used for raw-mode ref lines
        if self._show_mvc and has_cal:
            span = captured.meta.peak - baseline
            safe_span = np.where(span != 0, span, 1.0)
            sig = (sig - baseline) / safe_span * 100.0
            unit = "% MVC"
            if empty is not None:
                zero_positions = (empty - baseline) / safe_span * 100.0
        else:
            if has_cal:
                span = captured.meta.peak - baseline
                max_span = float(np.max(span[self._finger_channel_indices]))
                raw_mvc_span = max_span if max_span != 0 else 1.0
                sig = (sig - baseline) / raw_mvc_span * 100.0
                unit = "% max MVC"
                if empty is not None:
                    zero_positions = (empty - baseline) / raw_mvc_span * 100.0
            elif baseline is not None:
                sig = sig - baseline
                unit = "a.u."
                if empty is not None:
                    zero_positions = empty - baseline
            else:
                unit = "a.u."
                if empty is not None:
                    zero_positions = empty.astype(np.float64, copy=True)
        self._apply_unit_labels(unit)

        relative_time = (
            captured.batch.timestamps
            - captured.batch.timestamps[captured.trigger_sample]
        )

        for local_idx, curve in enumerate(self._raw_curves):
            ch_idx = self._finger_channel_indices[local_idx]
            force_data = sig[:, ch_idx]
            curve.setData(relative_time, force_data)

            marker = self._raw_max_markers[local_idx]
            if force_data.size > 0:
                max_idx = int(np.argmax(force_data))
                peak_force = float(force_data[max_idx])
                marker.setData([float(relative_time[max_idx])], [peak_force])
                color = _mvc_bin_color(peak_force) if self._show_mvc else "#AAAAAA"
                marker.setSymbolBrush(pg.mkBrush(color))
            else:
                marker.setData([], [])

            # MVC reference line: visible only in raw mode when full calibration exists.
            peak_ref = self._mvc_ref_lines[local_idx]
            show_ref = not self._show_mvc and has_cal
            peak_ref.setVisible(show_ref)
            if show_ref and raw_mvc_span is not None:
                finger_span = float(captured.meta.peak[ch_idx] - baseline[ch_idx])
                peak_ref.setPos(finger_span / raw_mvc_span * 100.0)

            # Zero (no-contact) reference line.
            zero_ref = self._zero_ref_lines[local_idx]
            if self._show_zero_ref and zero_positions is not None:
                zero_ref.setPos(float(zero_positions[ch_idx]))
                zero_ref.setVisible(True)
            else:
                zero_ref.setVisible(False)

            # Rest (baseline) reference line — always at 0 in display units.
            rest_ref = self._rest_ref_lines[local_idx]
            rest_ref.setVisible(self._show_rest_ref and baseline is not None)

            # P2P and onset overlay.
            p2p = float(np.ptp(force_data)) if force_data.size > 0 else 0.0
            onset_ms = detect_onset(force_data, captured.trigger_sample, self._sample_rate_hz)
            self._auto_onset_ms[local_idx] = onset_ms
            onset_line = self._onset_lines[local_idx]
            if onset_ms is not None:
                onset_line.setPos(onset_ms / 1000.0)
                onset_line.setVisible(True)
            else:
                onset_line.setVisible(False)
            p2p_str = f"P2P: {p2p:.1f} {unit}"
            self._finger_p2p_strs[local_idx] = p2p_str
            onset_str = f"Onset: {onset_ms:.0f} ms" if onset_ms is not None else "Onset: —"
            lbl = self._finger_info_labels[local_idx]
            lbl.setText(f"{p2p_str}\n{onset_str}")
            lbl.adjustSize()

        finger_signals = sig[:, self._finger_channel_indices]
        finger_zero_positions: np.ndarray | None = (
            zero_positions[self._finger_channel_indices]
            if zero_positions is not None
            else None
        )
        if relative_time.size > 0 and self._raw_plot_widgets:
            x_min = float(relative_time[0])
            x_max = float(relative_time[-1])

            if self._global_scale:
                y_min = float(np.min(finger_signals))
                y_max = float(np.max(finger_signals))
                if self._show_zero_ref and finger_zero_positions is not None:
                    y_min = min(y_min, float(np.min(finger_zero_positions)))
                y_span = y_max - y_min
                y_padding = max(0.8, y_span * 0.08)
                for panel in self._raw_plot_widgets:
                    panel.setXRange(x_min, x_max, padding=0.0)
                    panel.setYRange(y_min - y_padding, y_max + y_padding, padding=0.0)
            else:
                for local_idx, panel in enumerate(self._raw_plot_widgets):
                    finger_data = finger_signals[:, local_idx]
                    f_min = float(np.min(finger_data))
                    f_max = float(np.max(finger_data))
                    if self._show_zero_ref and finger_zero_positions is not None:
                        f_min = min(f_min, float(finger_zero_positions[local_idx]))
                    f_span = f_max - f_min
                    f_padding = max(0.8, f_span * 0.08)
                    panel.setXRange(x_min, x_max, padding=0.0)
                    panel.setYRange(f_min - f_padding, f_max + f_padding, padding=0.0)
        ordered_ranges = np.ptp(
            finger_signals[:, list(self._bar_display_indices)], axis=0
        )
        self._draw_range_bars(ordered_ranges)
        self.range_plot.setYRange(
            0.0, max(1.0, float(np.max(ordered_ranges)) * 1.2), padding=0.0
        )

        # Force a full viewport repaint: pyqtgraph's dirty-rect omits the
        # antialias fringe, leaving ghost trails of the prior capture.
        for panel in self._raw_plot_widgets:
            panel.viewport().update()

        if self._emg_window is not None:
            self._emg_window.update_capture(captured)

    def _on_onset_dragged(self, finger_idx: int, pos_seconds: float) -> None:
        """Updates the info label when a user manually drags an onset marker."""
        onset_ms = pos_seconds * 1000.0
        lbl = self._finger_info_labels[finger_idx]
        lbl.setText(f"{self._finger_p2p_strs[finger_idx]}\nOnset: {onset_ms:.0f} ms")
        lbl.adjustSize()

    def _reset_onset(self, finger_idx: int) -> None:
        """Resets a finger's onset marker to its automatically detected position."""
        onset_ms = self._auto_onset_ms[finger_idx]
        onset_line = self._onset_lines[finger_idx]
        if onset_ms is not None:
            onset_line.setPos(onset_ms / 1000.0)
            onset_line.setVisible(True)
        else:
            onset_line.setVisible(False)
        lbl = self._finger_info_labels[finger_idx]
        onset_str = f"Onset: {onset_ms:.0f} ms" if onset_ms is not None else "Onset: —"
        lbl.setText(f"{self._finger_p2p_strs[finger_idx]}\n{onset_str}")
        lbl.adjustSize()
