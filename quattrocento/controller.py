from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtCore

from .config import QuattrocentoConfig
from .device import QuattrocentoStream
from .models import CapturedWindow, DataBatch
from .processing import TriggerWindowProcessor, aggregate_finger_forces
from .ui import QuattrocentoMainWindow

_DEFAULT_MAX_HISTORY = 200


class QuattrocentoController(QtCore.QObject):
    """Coordinate stream polling, trigger processing, and UI updates."""

    def __init__(
        self,
        config: QuattrocentoConfig,
        stream: QuattrocentoStream,
        processor: TriggerWindowProcessor,
        window: QuattrocentoMainWindow,
        max_history: int = _DEFAULT_MAX_HISTORY,
    ) -> None:
        super().__init__()
        self._config = config
        self._stream = stream
        self._processor = processor
        self._window = window
        self._history: deque[CapturedWindow] = deque(maxlen=max_history)
        self._current_event_index: int | None = None
        self._mvc_calibrating: bool = False
        self._mvc_mins: NDArray[np.float64] | None = None
        self._mvc_maxs: NDArray[np.float64] | None = None

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._window.previous_requested.connect(self._show_previous_event)
        self._window.next_requested.connect(self._show_next_event)
        self._window.mvc_toggled.connect(self._on_mvc_toggled)

    def start(self) -> None:
        """Show the window and start periodic acquisition updates."""
        self._refresh_status()
        self._window.show()
        self._timer.start(self._config.ui_refresh_ms)

    def _on_mvc_toggled(self, active: bool) -> None:
        self._mvc_calibrating = active
        if active:
            self._mvc_mins = None
            self._mvc_maxs = None
            self._window.set_mvc_session_calibrated(False)
        else:
            calibrated = self._mvc_mins is not None and self._mvc_maxs is not None
            self._window.set_mvc_session_calibrated(calibrated)

    def _on_timer_tick(self) -> None:
        try:
            batch = self._stream.read_batch()
        except Exception:
            self._timer.stop()
            self._window.set_stream_error()
            return

        captured = self._processor.process_batch(batch)

        if self._mvc_calibrating:
            # Keep running process_batch so the trigger processor's adaptive
            # baseline keeps tracking the live signal during calibration;
            # discard any window it emits.
            self._update_mvc(batch)
            return

        if captured is not None:
            if self._mvc_mins is not None and self._mvc_maxs is not None:
                captured = self._apply_mvc_scaling(captured)
            self._append_capture(captured)

        self._refresh_status()

    def _update_mvc(self, batch: DataBatch) -> None:
        # Rest baseline is approximated as the per-finger min observed during
        # calibration, which assumes the operator relaxes at some point and
        # that a single low sample is representative. Both assumptions are
        # weak: min is biased by noise and vulnerable to spikes, and nothing
        # enforces a relaxation phase. A dynamic/dedicated rest estimator
        # (e.g. averaging a known-relaxed window, or continuous baseline
        # tracking) is the right fix — deferred.
        if batch.forces.shape[0] == 0:
            return

        finger_forces, _ = aggregate_finger_forces(
            batch.forces, self._config.finger_sensor_map
        )

        batch_mins = np.min(finger_forces, axis=0)
        batch_maxs = np.max(finger_forces, axis=0)

        if self._mvc_mins is None or self._mvc_maxs is None:
            self._mvc_mins = batch_mins
            self._mvc_maxs = batch_maxs
        else:
            self._mvc_mins = np.minimum(self._mvc_mins, batch_mins)
            self._mvc_maxs = np.maximum(self._mvc_maxs, batch_maxs)

    def _apply_mvc_scaling(self, captured: CapturedWindow) -> CapturedWindow:
        span = self._mvc_maxs - self._mvc_mins
        span = np.where(span == 0, 1.0, span)
        scaled_forces = (captured.finger_forces - self._mvc_mins) / span * 100.0
        return replace(
            captured,
            finger_forces=scaled_forces,
            finger_ranges=np.ptp(scaled_forces, axis=0),
            is_scaled=True,
        )

    def _append_capture(self, captured: CapturedWindow) -> None:
        was_showing_latest = self._current_event_index is None or (
            self._current_event_index == len(self._history) - 1
        )
        was_full = len(self._history) == self._history.maxlen
        self._history.append(captured)
        self._window.set_last_trigger_now()

        # When the deque evicts the oldest entry, shift the viewed index back.
        # If the user is viewing index 0, the evicted event is silently replaced
        # by its neighbor rather than clearing the view.
        if was_full and self._current_event_index is not None and not was_showing_latest:
            self._current_event_index = max(0, self._current_event_index - 1)

        if was_showing_latest:
            self._current_event_index = len(self._history) - 1
            self._window.update_capture(captured)

    def _show_previous_event(self) -> None:
        if self._current_event_index is None or self._current_event_index <= 0:
            return
        self._current_event_index -= 1
        self._window.update_capture(self._history[self._current_event_index])
        self._refresh_status()

    def _show_next_event(self) -> None:
        if self._current_event_index is None:
            return
        if self._current_event_index >= len(self._history) - 1:
            return
        self._current_event_index += 1
        self._window.update_capture(self._history[self._current_event_index])
        self._refresh_status()

    def _refresh_status(self) -> None:
        self._window.set_stream_state(
            sample_rate_hz=self._config.sample_rate_hz,
            captures=len(self._history),
            capturing=self._processor.is_capturing,
        )
        self._window.set_event_navigation(
            current_index=self._current_event_index,
            total_events=len(self._history),
        )
