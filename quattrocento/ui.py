from __future__ import annotations

from datetime import datetime
from typing import Callable, Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from .models import CapturedWindow
from .processing import detect_onset


_RAW_GRID_COLUMNS = 5
_MONITOR_ROLLING_SECONDS = 5.0
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
        line = pg.InfiniteLine(
            pos=t, angle=90,
            pen=pg.mkPen("#F6B73C", width=2),
        )
        self._plot.addItem(line)
        self._trigger_lines.append(line)


class EmgMonitorWindow(QtWidgets.QWidget):
    """Per-event view of rectified EMG channels with P2P and onset overlays.

    Receives the same CapturedWindow as the main finger view, but renders
    only the channels declared `kind = "emg"` in the channels TOML, after
    rectification. No baseline/MVC — values are |scaled signal|.
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
        self.hide()
        event.ignore()

    def _on_scale_toggled(self, checked: bool) -> None:
        self._global_scale = not checked
        self._scale_button.setText("Per Channel Scale" if checked else "Global Scale")
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_skip_changed(self, value: float) -> None:
        self._post_skip_ms = float(value)
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def update_capture(self, captured: CapturedWindow) -> None:
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

        emg_columns: list[np.ndarray] = []
        for local_idx, curve in enumerate(self._curves):
            ch_idx = self._channel_indices[local_idx]
            emg = sig[:, ch_idx]
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
        self._baseline_button = QtWidgets.QPushButton("Calibrate Rest")
        self._baseline_button.setObjectName("restButton")
        self._baseline_button.setCheckable(True)
        self._peak_button = QtWidgets.QPushButton("Calibrate MVC")
        self._peak_button.setObjectName("mvcButton")
        self._peak_button.setCheckable(True)
        self._hook_controls_layout: QtWidgets.QHBoxLayout | None = None
        self._raw_plot_widgets: list[pg.PlotWidget] = []
        self._raw_curves: list[pg.PlotDataItem] = []
        self._raw_max_markers: list[pg.PlotDataItem] = []
        self._mvc_ref_lines: list[pg.InfiniteLine] = []
        self._previous_shortcut: QtWidgets.QShortcut | None = None
        self._next_shortcut: QtWidgets.QShortcut | None = None
        self._sample_rate_hz = sample_rate_hz
        self._global_scale: bool = True
        self._show_mvc: bool = True
        self._last_capture: CapturedWindow | None = None
        self._finger_info_labels: list[QtWidgets.QLabel] = []
        self._onset_lines: list[_OnsetLine] = []
        self._auto_onset_ms: list[float | None] = [None] * _EXPECTED_FINGER_COUNT
        self._finger_p2p_strs: list[str] = ["P2P: —"] * _EXPECTED_FINGER_COUNT
        self._scale_button = QtWidgets.QPushButton("Global Scale")
        self._scale_button.setObjectName("scaleButton")
        self._scale_button.setCheckable(True)
        self._display_mode_button = QtWidgets.QPushButton("% MVC")
        self._display_mode_button.setObjectName("displayModeButton")
        self._display_mode_button.setCheckable(True)
        self._monitor_button = QtWidgets.QPushButton("Trigger Monitor")
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

        self._apply_palette()
        self._build_layout()
        self._install_navigation_shortcuts()
        self.set_event_navigation(current_index=None, total_events=0)
        self.set_calibration_status(baseline_done=False, peak_done=False)
        self._apply_unit_labels("a.u.")

    def _apply_palette(self) -> None:
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
            QPushButton {
                background: #DCE7F4;
                border: 1px solid #BFD0E1;
                border-radius: 8px;
                color: #1F3242;
                font-size: 12px;
                font-weight: 600;
                padding: 5px 12px;
            }
            QPushButton:disabled {
                background: #EEF3F9;
                border-color: #DCE5EF;
                color: #9AABBE;
            }
            QPushButton:hover:!disabled {
                background: #CCDDF0;
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
            """
        )

    def _build_layout(self) -> None:
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
        self._baseline_button.toggled.connect(self.baseline_toggled.emit)
        self._peak_button.toggled.connect(self.peak_toggled.emit)
        self._scale_button.toggled.connect(self._on_scale_toggled)
        self._display_mode_button.toggled.connect(self._on_display_mode_toggled)
        self._monitor_button.clicked.connect(self._toggle_trigger_monitor)
        self._emg_button.clicked.connect(self._toggle_emg_monitor)
        self._previous_button.setToolTip("Previous event (Left Arrow)")
        self._next_button.setToolTip("Next event (Right Arrow)")
        self._baseline_button.setToolTip("Toggle Rest (baseline) calibration")
        self._peak_button.setToolTip("Toggle Maximum Voluntary Contraction calibration")
        self._hook_controls_layout = QtWidgets.QHBoxLayout()
        self._hook_controls_layout.setSpacing(8)
        navigation_layout.addWidget(self._previous_button)
        navigation_layout.addWidget(self._next_button)
        navigation_layout.addWidget(self._event_position_label)
        navigation_layout.addStretch(1)
        navigation_layout.addWidget(self._build_legend())
        navigation_layout.addWidget(self._display_mode_button)
        navigation_layout.addWidget(self._scale_button)
        navigation_layout.addWidget(self._baseline_button)
        navigation_layout.addWidget(self._peak_button)
        navigation_layout.addWidget(self._monitor_button)
        navigation_layout.addWidget(self._emg_button)
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
        self._draw_range_bars(np.zeros(len(self._bar_display_labels), dtype=np.float64))

    def _build_raw_grid(self, grid_layout: QtWidgets.QGridLayout) -> None:
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
        if self._trigger_monitor.isVisible():
            self._trigger_monitor.hide()
        else:
            self._trigger_monitor.show()
            self._trigger_monitor.raise_()

    def _toggle_emg_monitor(self) -> None:
        if self._emg_window is None:
            return
        if self._emg_window.isVisible():
            self._emg_window.hide()
        else:
            self._emg_window.show()
            self._emg_window.raise_()

    def push_trigger_batch(
        self,
        timestamps: np.ndarray,
        trigger_col: np.ndarray,
        dc: float | None,
        noise: float,
        effective_threshold: float,
        warmup_remaining: int,
    ) -> None:
        if self._trigger_monitor.isVisible():
            self._trigger_monitor.push_batch(
                timestamps, trigger_col, dc, noise, effective_threshold, warmup_remaining
            )

    def mark_trigger(self, t: float) -> None:
        if self._trigger_monitor.isVisible():
            self._trigger_monitor.mark_trigger(t)

    def add_hook_controls(
        self,
        name: str,
        *,
        on_toggle: Callable[[bool], None],
        on_reset: Callable[[], None],
    ) -> None:
        toggle_btn = QtWidgets.QPushButton(name)
        toggle_btn.setObjectName("hookToggle")
        toggle_btn.setCheckable(True)
        toggle_btn.toggled.connect(on_toggle)

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(on_reset)

        self._hook_controls_layout.addWidget(toggle_btn)
        self._hook_controls_layout.addWidget(reset_btn)

    def revert_baseline_button(self) -> None:
        self._baseline_button.blockSignals(True)
        self._baseline_button.setChecked(False)
        self._baseline_button.blockSignals(False)

    def revert_peak_button(self) -> None:
        self._peak_button.blockSignals(True)
        self._peak_button.setChecked(False)
        self._peak_button.blockSignals(False)

    def show_error(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Error", message)

    def set_stream_error(self) -> None:
        self._acquisition_label.setText("State: Stream error")

    def set_stream_state(self, sample_rate_hz: int, captures: int, capturing: bool) -> None:
        state_text = "Capturing window..." if capturing else "Waiting for trigger"
        self._acquisition_label.setText(f"State: {state_text}")
        self._capture_count_label.setText(f"Events: {captures}")
        self._sampling_label.setText(f"Sample rate: {sample_rate_hz} Hz")

    def set_last_trigger_now(self) -> None:
        self._last_trigger_label.setText(
            f"Last trigger: {datetime.now().strftime('%H:%M:%S')}"
        )

    def set_calibration_status(self, baseline_done: bool, peak_done: bool) -> None:
        rest_mark = "✓" if baseline_done else "✗"
        mvc_mark = "✓" if peak_done else "✗"
        self._cal_status_label.setText(f"Cal: Rest {rest_mark} | MVC {mvc_mark}")

    def show_calibration_report(
        self,
        display_channels: list[tuple[int, str]],
        baseline: np.ndarray | None,
        peak: np.ndarray | None,
    ) -> None:
        """Pop up a per-channel calibration summary."""
        header = f"{'Channel':<12} {'Rest':>10} {'MVC':>10} {'Span':>10}"
        rows = [header, "-" * len(header)]
        for ch_idx, label in display_channels:
            rest_val = "—" if baseline is None else f"{baseline[ch_idx]:.2f}"
            mvc_val = "—" if peak is None else f"{peak[ch_idx]:.2f}"
            if baseline is not None and peak is not None:
                span_val = f"{peak[ch_idx] - baseline[ch_idx]:.2f}"
            else:
                span_val = "—"
            rows.append(f"{label:<12} {rest_val:>10} {mvc_val:>10} {span_val:>10}")

        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Calibration Report")
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setText("Per-channel calibration values:")
        dialog.setFont(QtGui.QFont("Courier New", 10))
        dialog.setInformativeText("\n".join(rows))
        dialog.exec_()

    def _apply_unit_labels(self, unit: str) -> None:
        self.range_plot.setLabel("left", "Range", units=unit)

    def set_event_navigation(self, current_index: int | None, total_events: int) -> None:
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
        self._show_mvc = not checked
        self._display_mode_button.setText("Raw" if checked else "% MVC")
        if self._last_capture is not None:
            self.update_capture(self._last_capture)

    def _on_scale_toggled(self, checked: bool) -> None:
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
        has_cal = captured.meta.baseline is not None and captured.meta.peak is not None

        # % MVC mode: normalize signals; Raw mode: normalize by global max MVC span if
        # calibrated, otherwise subtract baseline or show raw.
        raw_mvc_span: float | None = None  # global max span used for raw-mode ref lines
        if self._show_mvc and has_cal:
            span = captured.meta.peak - captured.meta.baseline
            safe_span = np.where(span != 0, span, 1.0)
            sig = (sig - captured.meta.baseline) / safe_span * 100.0
            unit = "% MVC"
        else:
            has_baseline = captured.meta.baseline is not None
            if has_cal:
                span = captured.meta.peak - captured.meta.baseline
                max_span = float(np.max(span[self._finger_channel_indices]))
                raw_mvc_span = max_span if max_span != 0 else 1.0
                sig = (sig - captured.meta.baseline) / raw_mvc_span * 100.0
                unit = "% max MVC"
            elif has_baseline:
                sig = sig - captured.meta.baseline
                unit = "a.u."
            else:
                unit = "a.u."
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
                finger_span = float(captured.meta.peak[ch_idx] - captured.meta.baseline[ch_idx])
                peak_ref.setPos(finger_span / raw_mvc_span * 100.0)

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
        if relative_time.size > 0 and self._raw_plot_widgets:
            x_min = float(relative_time[0])
            x_max = float(relative_time[-1])

            if self._global_scale:
                y_min = float(np.min(finger_signals))
                y_max = float(np.max(finger_signals))
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

        if self._emg_window is not None:
            self._emg_window.update_capture(captured)

    def _on_onset_dragged(self, finger_idx: int, pos_seconds: float) -> None:
        onset_ms = pos_seconds * 1000.0
        lbl = self._finger_info_labels[finger_idx]
        lbl.setText(f"{self._finger_p2p_strs[finger_idx]}\nOnset: {onset_ms:.0f} ms")
        lbl.adjustSize()

    def _reset_onset(self, finger_idx: int) -> None:
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
