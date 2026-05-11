from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtCore

from .config import QuattrocentoConfig
from .models import CapturedWindow, DataBatch, EventHook, Stream, StreamHook, StreamMeta
from .processing import TriggerWindowProcessor
from .ui import QuattrocentoMainWindow

_DEFAULT_MAX_HISTORY = 200


class QuattrocentoController(QtCore.QObject):
    """Coordinate stream polling, trigger processing, and UI updates."""

    def __init__(
        self,
        config: QuattrocentoConfig,
        stream: Stream,
        processor: TriggerWindowProcessor,
        window: QuattrocentoMainWindow,
        meta: StreamMeta,
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
        self._baseline_calibrating: bool = False
        self._peak_calibrating: bool = False
        self._baseline_buffer: list[NDArray[np.float64]] = []
        self._peak_buffer: list[NDArray[np.float64]] = []
        self._meta = meta

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._window.previous_requested.connect(self._show_previous_event)
        self._window.next_requested.connect(self._show_next_event)
        self._window.baseline_toggled.connect(self._on_baseline_toggled)
        self._window.peak_toggled.connect(self._on_peak_toggled)

        for hook in (*self._stream_hooks, *self._event_hooks):
            if hook.ui_controls:
                self._window.add_hook_controls(
                    hook.name,
                    on_toggle=hook.set_active,
                    on_reset=hook.reset,
                )

    def start(self) -> None:
        self._refresh_status()
        self._window.show()
        self._timer.start(self._config.ui_refresh_ms)

    def _on_baseline_toggled(self, active: bool) -> None:
        if active and self._processor.is_capturing:
            self._window.show_error(
                "Cannot start rest calibration while a trigger window is in progress."
            )
            self._window.revert_baseline_button()
            return
        self._baseline_calibrating = active
        if active:
            self._baseline_buffer.clear()
            self._meta = replace(self._meta, baseline=None)
            self._update_calibration_ui()
        else:
            if self._baseline_buffer:
                data = np.concatenate(self._baseline_buffer, axis=0)
                self._meta = replace(self._meta, baseline=np.mean(data, axis=0))
                self._show_calibration_report()
            else:
                self._window.show_error("Rest calibration failed: no data collected.")
            self._update_calibration_ui()

    def _on_peak_toggled(self, active: bool) -> None:
        if active and self._processor.is_capturing:
            self._window.show_error(
                "Cannot start MVC calibration while a trigger window is in progress."
            )
            self._window.revert_peak_button()
            return
        self._peak_calibrating = active
        if active:
            self._peak_buffer.clear()
            self._meta = replace(self._meta, peak=None)
            self._update_calibration_ui()
        else:
            if self._peak_buffer:
                data = np.concatenate(self._peak_buffer, axis=0)
                self._meta = replace(self._meta, peak=np.max(data, axis=0))
                self._show_calibration_report()
            else:
                self._window.show_error("MVC calibration failed: no data collected.")
            self._update_calibration_ui()

    def _show_calibration_report(self) -> None:
        if self._meta.baseline is None and self._meta.peak is None:
            return
        display_channels = sorted(
            (idx, label) for idx, label in self._meta.channel_labels.items()
            if idx != self._config.trigger_channel
        )
        self._window.show_calibration_report(
            display_channels=display_channels,
            baseline=self._meta.baseline,
            peak=self._meta.peak,
        )

    def _update_calibration_ui(self) -> None:
        self._window.set_calibration_status(
            baseline_done=self._meta.baseline is not None,
            peak_done=self._meta.peak is not None,
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

        captured_list = self._processor.process_batch(batch, self._meta)

        if self._baseline_calibrating or self._peak_calibrating:
            if self._baseline_calibrating:
                self._update_baseline(batch)
            if self._peak_calibrating:
                self._update_peak(batch)
            return

        for captured in captured_list:
            self._append_capture(captured)

        self._refresh_status()

    def _update_baseline(self, batch: DataBatch) -> None:
        if batch.signals.shape[0] == 0:
            return
        self._baseline_buffer.append(batch.signals)

    def _update_peak(self, batch: DataBatch) -> None:
        if batch.signals.shape[0] == 0:
            return
        self._peak_buffer.append(batch.signals)

    def _append_capture(self, captured: CapturedWindow) -> None:
        was_showing_latest = self._current_event_index is None or (
            self._current_event_index == len(self._history) - 1
        )
        was_full = len(self._history) == self._history.maxlen
        self._history.append(captured)
        for hook in self._event_hooks:
            hook(captured)
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
