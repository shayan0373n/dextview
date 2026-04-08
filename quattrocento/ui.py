from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from .models import CapturedWindow


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

    def __init__(self, finger_labels: Sequence[str]) -> None:
        """Create the UI for finger-range and raw-force event plots."""
        super().__init__()
        self._finger_labels = tuple(finger_labels)
        self._bar_display_indices, self._bar_display_labels = _build_mirrored_bar_layout(
            self._finger_labels
        )
        self._bar_x = np.arange(len(self._bar_display_labels), dtype=np.float64)
        self._bar_item: pg.BarGraphItem | None = None

        self._capture_count_label = QtWidgets.QLabel("Events: 0")
        self._acquisition_label = QtWidgets.QLabel("State: Waiting for trigger")
        self._last_trigger_label = QtWidgets.QLabel("Last trigger: -")
        self._sampling_label = QtWidgets.QLabel("Sample rate: -")
        self._event_position_label = QtWidgets.QLabel("Viewing: -/-")
        self._previous_button = QtWidgets.QPushButton("< Prev")
        self._next_button = QtWidgets.QPushButton("Next >")
        self._raw_plot_widgets: list[pg.PlotWidget] = []
        self._raw_curves: list[pg.PlotDataItem] = []
        self._previous_shortcut: QtWidgets.QShortcut | None = None
        self._next_shortcut: QtWidgets.QShortcut | None = None

        self._apply_palette()
        self._build_layout()
        self._install_navigation_shortcuts()
        self.set_event_navigation(current_index=None, total_events=0)

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
        self._previous_button.setToolTip("Previous event (Left Arrow)")
        self._next_button.setToolTip("Next event (Right Arrow)")
        navigation_layout.addWidget(self._previous_button)
        navigation_layout.addWidget(self._next_button)
        navigation_layout.addWidget(self._event_position_label)
        navigation_layout.addStretch(1)
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
        columns = 5
        for finger_idx, finger_name in enumerate(self._finger_labels):
            row = finger_idx // columns
            col = finger_idx % columns

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

            self._raw_plot_widgets.append(panel)
            self._raw_curves.append(curve)
            grid_layout.addWidget(panel, row, col)

        if not self._raw_plot_widgets:
            return

        reference = self._raw_plot_widgets[0]
        for panel in self._raw_plot_widgets[1:]:
            panel.setXLink(reference)
            panel.setYLink(reference)

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

    def set_stream_error(self) -> None:
        """Indicate that the stream has disconnected or failed."""
        self._acquisition_label.setText("State: Stream error")

    def set_stream_state(self, sample_rate_hz: int, captures: int, capturing: bool) -> None:
        """Update acquisition status chips."""
        state_text = "Capturing window..." if capturing else "Waiting for trigger"
        self._acquisition_label.setText(f"State: {state_text}")
        self._capture_count_label.setText(f"Events: {captures}")
        self._sampling_label.setText(f"Sample rate: {sample_rate_hz} Hz")

    def set_last_trigger_now(self) -> None:
        """Stamp the status area with the current local trigger time."""
        self._last_trigger_label.setText(
            f"Last trigger: {datetime.now().strftime('%H:%M:%S')}"
        )

    def set_event_navigation(self, current_index: int | None, total_events: int) -> None:
        """Refresh event navigation state and button availability."""
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
        relative_time = captured.timestamps - captured.timestamps[0]
        for finger_idx, curve in enumerate(self._raw_curves):
            curve.setData(relative_time, captured.finger_forces[:, finger_idx])

        if relative_time.size > 0 and self._raw_plot_widgets:
            x_max = float(relative_time[-1])
            y_min = float(np.min(captured.finger_forces))
            y_max = float(np.max(captured.finger_forces))
            y_span = y_max - y_min
            y_padding = max(0.8, y_span * 0.08)
            shared_y_min = y_min - y_padding
            shared_y_max = y_max + y_padding

            for panel in self._raw_plot_widgets:
                panel.setXRange(0.0, x_max, padding=0.0)
                panel.setYRange(shared_y_min, shared_y_max, padding=0.0)

        ordered_ranges = captured.finger_ranges[np.array(self._bar_display_indices)]
        self._draw_range_bars(ordered_ranges)
        y_max = float(np.max(ordered_ranges))
        self.range_plot.setYRange(0.0, max(1.0, y_max * 1.2), padding=0.0)
