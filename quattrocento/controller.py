from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtCore

from .config import QuattrocentoConfig
from .device import QuattrocentoStream
from .models import CapturedWindow, DataBatch, EventHook, StreamHook, StreamMeta
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
        stream_hooks: Sequence[StreamHook] = (),
        event_hooks: Sequence[EventHook] = (),
    ) -> None:
        super().__init__()
        self._config = config
        self._stream = stream
        self._processor = processor
        self._window = window
        self._stream_hooks: list[StreamHook] = list(stream_hooks)
        self._event_hooks: list[EventHook] = list(event_hooks)
        self._history: deque[CapturedWindow] = deque(maxlen=max_history)
        self._current_event_index: int | None = None
        self._mvc_calibrating: bool = False
        self._rest_calibrating: bool = False
        self._mvc_buffer: list[NDArray[np.float64]] = []
        self._rest_buffer: list[NDArray[np.float64]] = []
        self._rest_means: NDArray[np.float64] | None = None
        self._mvc_maxs: NDArray[np.float64] | None = None
        self._meta = StreamMeta(
            finger_sensor_map=config.finger_sensor_map,
            finger_labels=config.finger_labels,
            sample_rate_hz=config.sample_rate_hz,
        )

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._window.previous_requested.connect(self._show_previous_event)
        self._window.next_requested.connect(self._show_next_event)
        self._window.rest_toggled.connect(self._on_rest_toggled)
        self._window.mvc_toggled.connect(self._on_mvc_toggled)

        for hook in (*self._stream_hooks, *self._event_hooks):
            if hook.ui_controls:
                self._window.add_hook_controls(
                    hook.name,
                    on_toggle=hook.set_active,
                    on_reset=hook.reset,
                )

    def start(self) -> None:
        """Show the window and start periodic acquisition updates."""
        self._refresh_status()
        self._window.show()
        self._timer.start(self._config.ui_refresh_ms)

    def _on_rest_toggled(self, active: bool) -> None:
        if active and self._processor.is_capturing:
            self._window.show_error(
                "Cannot start rest calibration while a trigger window is in progress."
            )
            self._window.revert_rest_button()
            return
        self._rest_calibrating = active
        if active:
            self._rest_buffer.clear()
            self._rest_means = None
            self._refresh_meta()
            self._update_calibration_ui()
        else:
            if self._rest_buffer:
                data = np.concatenate(self._rest_buffer, axis=0)
                self._rest_means = np.mean(data, axis=0)
                self._show_calibration_report()
            else:
                self._window.show_error("Rest calibration failed: no data collected.")
            self._refresh_meta()
            self._update_calibration_ui()

    def _on_mvc_toggled(self, active: bool) -> None:
        if active and self._processor.is_capturing:
            self._window.show_error(
                "Cannot start MVC calibration while a trigger window is in progress."
            )
            self._window.revert_mvc_button()
            return
        self._mvc_calibrating = active
        if active:
            self._mvc_buffer.clear()
            self._mvc_maxs = None
            self._refresh_meta()
            self._update_calibration_ui()
        else:
            if self._mvc_buffer:
                data = np.concatenate(self._mvc_buffer, axis=0)
                self._mvc_maxs = np.max(data, axis=0)
                self._show_calibration_report()
            else:
                self._window.show_error("MVC calibration failed: no data collected.")
            self._refresh_meta()
            self._update_calibration_ui()

    def _refresh_meta(self) -> None:
        self._meta = replace(
            self._meta, rest_means=self._rest_means, mvc_maxs=self._mvc_maxs
        )

    def _show_calibration_report(self) -> None:
        if self._rest_means is None and self._mvc_maxs is None:
            return
        self._window.show_calibration_report(
            finger_labels=self._config.finger_labels,
            rest_means=self._rest_means,
            mvc_maxs=self._mvc_maxs,
        )

    def _update_calibration_ui(self) -> None:
        self._window.set_calibration_status(
            rest_calibrated=self._rest_means is not None,
            mvc_calibrated=self._mvc_maxs is not None,
        )

    def _on_timer_tick(self) -> None:
        try:
            batch = self._stream.read_batch()
        except Exception:
            self._timer.stop()
            self._window.set_stream_error()
            return

        for hook in self._stream_hooks:
            hook(batch, self._meta)

        captured_list = self._processor.process_batch(batch)

        if self._rest_calibrating or self._mvc_calibrating:
            # Keep running process_batch so the trigger processor's adaptive
            # baseline keeps tracking the live signal during calibration;
            # discard any window it emits.
            if self._rest_calibrating:
                self._update_rest(batch)
            if self._mvc_calibrating:
                self._update_mvc(batch)
            return

        for captured in captured_list:
            if self._rest_means is not None and self._mvc_maxs is not None:
                captured = self._apply_mvc_scaling(captured)
            self._append_capture(captured)

        self._refresh_status()

    def _update_rest(self, batch: DataBatch) -> None:
        if batch.forces.shape[0] == 0:
            return

        finger_forces, _ = aggregate_finger_forces(
            batch.forces, self._config.finger_sensor_map
        )
        self._rest_buffer.append(finger_forces)

    def _update_mvc(self, batch: DataBatch) -> None:
        if batch.forces.shape[0] == 0:
            return

        finger_forces, _ = aggregate_finger_forces(
            batch.forces, self._config.finger_sensor_map
        )
        self._mvc_buffer.append(finger_forces)

    def _apply_mvc_scaling(self, captured: CapturedWindow) -> CapturedWindow:
        span = self._mvc_maxs - self._rest_means
        if np.any(span == 0):
            zero_fingers = [
                self._config.finger_labels[i]
                for i in np.flatnonzero(span == 0)
            ]
            raise ValueError(
                f"MVC equals rest for finger(s) {zero_fingers}; "
                "recalibrate before scaling."
            )
        scaled_forces = (captured.finger_forces - self._rest_means) / span * 100.0
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
        for hook in self._event_hooks:
            hook(captured, self._meta)
        self._window.set_last_trigger_now()

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
