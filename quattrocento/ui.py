from __future__ import annotations

from datetime import datetime
from typing import Callable, Mapping, Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from .models import CapturedWindow


_RAW_GRID_COLUMNS = 5
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
    ) -> None:
        super().__init__()

        # Ordered non-trigger finger channels, sorted by channel index.
        finger_channels = sorted(
            (idx, label)
            for idx, label in channel_labels.items()
            if idx != trigger_channel
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
        self._previous_shortcut: QtWidgets.QShortcut | None = None
        self._next_shortcut: QtWidgets.QShortcut | None = None

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
        navigation_layout.addWidget(self._baseline_button)
        navigation_layout.addWidget(self._peak_button)
        navigation_layout.addLayout(self._hook_controls_layout)
        root_layout.addLayout(navigation_layout)

        self.range_plot = pg.PlotWidget()
        self._style_range_plot()
        root_layout.addWidget(self.range_plot, stretch=2)

        raw_grid_container = QtWidgets.QGroupBox("Raw Finger Force (Post Trigger)")
        raw_grid_container.setFlat(True)
        raw_grid_layout = QtWidgets.QGridLayout(raw_grid_container)
        raw_grid_layout.setContentsMargins(2, 8, 2, 2)
        raw_grid_layout.setHorizontalSpacing(6)
        raw_grid_layout.setVerticalSpacing(6)
        self._build_raw_grid(raw_grid_layout)
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
        self.range_plot.getAxis("bottom").setTickFont(QtGui.QFont("Segoe UI", 8))
        self.range_plot.setYRange(0.0, 1.0, padding=0.0)
        self.range_plot.setXRange(-0.6, len(self._bar_display_labels) - 0.4, padding=0.0)
        self._draw_range_bars(np.zeros(len(self._bar_display_labels), dtype=np.float64))

    def _build_raw_grid(self, grid_layout: QtWidgets.QGridLayout) -> None:
        for finger_idx, finger_name in enumerate(self._finger_labels):
            row = finger_idx // _RAW_GRID_COLUMNS
            col = finger_idx % _RAW_GRID_COLUMNS

            panel = pg.PlotWidget()
            panel.setMenuEnabled(False)
            panel.setMouseEnabled(x=False, y=False)
            panel.showGrid(x=True, y=True, alpha=0.18)
            panel.setTitle(finger_name, size="9pt", color="#2E3A46")

            if row == 1:
                panel.setLabel("bottom", "Time", units="s")
            else:
                panel.getAxis("bottom").setStyle(showValues=False)

            if col == 0:
                panel.setLabel("left", "Force", units="a.u.")
            else:
                panel.getAxis("left").setStyle(showValues=False)

            pen = pg.mkPen(FINGER_COLORS[finger_idx % len(FINGER_COLORS)], width=2.0)
            curve = panel.plot([], [], pen=pen)

            marker = panel.plot(
                [], [], pen=None, symbol="o", symbolSize=8,
                symbolBrush="#E63946", symbolPen="w"
            )

            self._raw_plot_widgets.append(panel)
            self._raw_curves.append(curve)
            self._raw_max_markers.append(marker)
            grid_layout.addWidget(panel, row, col)

        if not self._raw_plot_widgets:
            return

        reference = self._raw_plot_widgets[0]
        for panel in self._raw_plot_widgets[1:]:
            panel.setXLink(reference)
            panel.setYLink(reference)

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
        for i, panel in enumerate(self._raw_plot_widgets):
            if i % _RAW_GRID_COLUMNS == 0:
                panel.setLabel("left", "Force", units=unit)

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

    def update_capture(self, captured: CapturedWindow) -> None:
        """Render one captured event in both plots."""
        sig = captured.batch.signals
        if captured.meta.baseline is not None and captured.meta.peak is not None:
            span = captured.meta.peak - captured.meta.baseline
            safe_span = np.where(span != 0, span, 1.0)
            sig = (sig - captured.meta.baseline) / safe_span * 100.0
            unit = "% MVC"
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
                marker.setSymbolBrush(pg.mkBrush(_mvc_bin_color(peak_force)))
            else:
                marker.setData([], [])

        finger_signals = sig[:, self._finger_channel_indices]
        if relative_time.size > 0 and self._raw_plot_widgets:
            x_min = float(relative_time[0])
            x_max = float(relative_time[-1])
            y_min = float(np.min(finger_signals))
            y_max = float(np.max(finger_signals))
            y_span = y_max - y_min
            y_padding = max(0.8, y_span * 0.08)

            for panel in self._raw_plot_widgets:
                panel.setXRange(x_min, x_max, padding=0.0)
                panel.setYRange(y_min - y_padding, y_max + y_padding, padding=0.0)
        ordered_ranges = np.ptp(
            finger_signals[:, list(self._bar_display_indices)], axis=0
        )
        self._draw_range_bars(ordered_ranges)
        self.range_plot.setYRange(
            0.0, max(1.0, float(np.max(ordered_ranges)) * 1.2), padding=0.0
        )
