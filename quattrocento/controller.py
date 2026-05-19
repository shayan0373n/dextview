from collections import deque
from dataclasses import replace
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtCore

from .config import QuattrocentoConfig
from .models import CapturedWindow, DataBatch, EventHook, Stream, StreamHook, StreamMeta
from .processing import TriggerWindowProcessor, detect_onset
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
        self._empty_calibrating: bool = False
        self._baseline_buffer: list[NDArray[np.float64]] = []
        self._peak_buffer: list[NDArray[np.float64]] = []
        self._empty_buffer: list[NDArray[np.float64]] = []
        self._meta = meta

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._window.previous_requested.connect(self._show_previous_event)
        self._window.next_requested.connect(self._show_next_event)
        self._window.baseline_toggled.connect(self._on_baseline_toggled)
        self._window.peak_toggled.connect(self._on_peak_toggled)
        self._window.empty_toggled.connect(self._on_empty_toggled)
        self._window.save_calibration_requested.connect(self._on_save_calibration)
        self._window.load_calibration_requested.connect(self._on_load_calibration)

    def start(self) -> None:
        """Starts the stream polling and shows the UI."""
        self._refresh_status()
        self._window.show()
        self._timer.start(self._config.ui_refresh_ms)

    def _on_baseline_toggled(self, active: bool) -> None:
        """Handles toggling of the baseline calibration mode."""
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
        """Handles toggling of the peak calibration mode."""
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

    def _on_empty_toggled(self, active: bool) -> None:
        """Handles toggling of the zero (no-contact) calibration mode."""
        if active and self._processor.is_capturing:
            self._window.show_error(
                "Cannot start zero calibration while a trigger window is in progress."
            )
            self._window.revert_empty_button()
            return
        self._empty_calibrating = active
        if active:
            self._empty_buffer.clear()
            self._meta = replace(self._meta, empty=None)
            self._update_calibration_ui()
        else:
            if self._empty_buffer:
                data = np.concatenate(self._empty_buffer, axis=0)
                self._meta = replace(self._meta, empty=np.mean(data, axis=0))
                self._show_calibration_report()
            else:
                self._window.show_error("Zero calibration failed: no data collected.")
            self._update_calibration_ui()

    def _show_calibration_report(self) -> None:
        """Updates and displays the calibration report in the UI."""
        if (
            self._meta.baseline is None
            and self._meta.peak is None
            and self._meta.empty is None
        ):
            return
        display_channels = sorted(
            (idx, label) for idx, label in self._meta.channel_labels.items()
            if idx != self._config.trigger_channel
        )
        self._window.show_calibration_report(
            display_channels=display_channels,
            baseline=self._meta.baseline,
            peak=self._meta.peak,
            empty=self._meta.empty,
        )

    def _on_save_calibration(self, path: str) -> None:
        """Saves non-None calibration arrays (zero/rest/MVC) to a .npz file."""
        arrays: dict[str, NDArray[np.float64]] = {}
        if self._meta.baseline is not None:
            arrays["baseline"] = self._meta.baseline
        if self._meta.peak is not None:
            arrays["peak"] = self._meta.peak
        if self._meta.empty is not None:
            arrays["empty"] = self._meta.empty
        if not arrays:
            self._window.show_error("No calibration data to save.")
            return
        try:
            np.savez(path, **arrays)
        except (OSError, ValueError) as exc:
            self._window.show_error(f"Failed to save calibration: {exc}")

    def _on_load_calibration(self, path: str) -> None:
        """Loads calibration arrays from a .npz file and replaces the current meta."""
        try:
            data = np.load(path)
        except (OSError, ValueError) as exc:
            self._window.show_error(f"Failed to load calibration: {exc}")
            return

        n = self._config.n_channels

        def _get(key: str) -> NDArray[np.float64] | None:
            if key not in data:
                return None
            arr = data[key]
            if arr.shape != (n,):
                raise ValueError(f"{key} has shape {arr.shape}, expected ({n},)")
            return arr.astype(np.float64)

        try:
            baseline = _get("baseline")
            peak = _get("peak")
            empty = _get("empty")
        except ValueError as exc:
            self._window.show_error(f"Calibration file mismatch: {exc}")
            return

        self._meta = replace(self._meta, baseline=baseline, peak=peak, empty=empty)
        self._update_calibration_ui()
        self._show_calibration_report()

    def _update_calibration_ui(self) -> None:
        """Refreshes the calibration status indicators in the UI."""
        self._window.set_calibration_status(
            baseline_done=self._meta.baseline is not None,
            peak_done=self._meta.peak is not None,
            empty_done=self._meta.empty is not None,
        )

    def _on_timer_tick(self) -> None:
        """Periodically polls the stream for new data and updates the UI."""
        try:
            batch = self._stream.read_batch()
        except Exception:
            self._timer.stop()
            self._window.set_stream_error()
            return

        for hook in self._stream_hooks:
            hook(batch, self._meta)

        captured_list = self._processor.process_batch(batch, self._meta)

        self._window.push_trigger_batch(
            timestamps=batch.timestamps,
            trigger_col=batch.signals[:, self._config.trigger_channel],
            dc=self._processor.trigger_dc,
            noise=self._processor.trigger_noise,
            effective_threshold=self._processor.effective_threshold,
            warmup_remaining=self._processor.warmup_remaining_samples,
        )
        self._window.push_live_batch(batch.timestamps, batch.signals)

        if self._baseline_calibrating or self._peak_calibrating or self._empty_calibrating:
            if self._baseline_calibrating:
                self._update_baseline(batch)
            if self._peak_calibrating:
                self._update_peak(batch)
            if self._empty_calibrating:
                self._update_empty(batch)
            return

        for captured in captured_list:
            self._append_capture(captured)

        self._refresh_status()

    def _update_baseline(self, batch: DataBatch) -> None:
        """Appends new batch data to the baseline calibration buffer."""
        if batch.signals.shape[0] == 0:
            return
        self._baseline_buffer.append(batch.signals)

    def _update_peak(self, batch: DataBatch) -> None:
        """Appends new batch data to the peak calibration buffer."""
        if batch.signals.shape[0] == 0:
            return
        self._peak_buffer.append(batch.signals)

    def _update_empty(self, batch: DataBatch) -> None:
        """Appends new batch data to the zero (no-contact) calibration buffer."""
        if batch.signals.shape[0] == 0:
            return
        self._empty_buffer.append(batch.signals)

    def _append_capture(self, captured: CapturedWindow) -> None:
        """Stores a new capture window and updates the UI to show it."""
        was_showing_latest = self._current_event_index is None or (
            self._current_event_index == len(self._history) - 1
        )
        was_full = len(self._history) == self._history.maxlen
        self._history.append(captured)
        for hook in self._event_hooks:
            hook(captured)
        self._window.set_last_trigger_now()
        self._window.mark_trigger(float(captured.batch.timestamps[captured.trigger_sample]))

        if was_full and self._current_event_index is not None and not was_showing_latest:
            self._current_event_index = max(0, self._current_event_index - 1)

        if was_showing_latest:
            self._current_event_index = len(self._history) - 1
            self._update_window_capture(captured)

    def _update_window_capture(self, captured: CapturedWindow) -> None:
        """Helper to deconstruct CapturedWindow and call the UI boundary."""
        # Pre-calculate onsets for all fingers to pass across the boundary.
        onset_ms_list = []
        # Get finger indices from window (we could also get them from meta if we had a mapping)
        # But controller knows the config.
        finger_indices = self._window._finger_channel_indices
        for idx in finger_indices:
            # We use the raw signals for onset detection.
            # detect_onset expects signal, trigger_idx, sample_rate
            onset = detect_onset(
                captured.batch.signals[:, idx],
                captured.trigger_sample,
                captured.meta.config.sample_rate_hz
            )
            onset_ms_list.append(onset)

        self._window.update_capture(
            timestamps=captured.batch.timestamps,
            signals=captured.batch.signals,
            trigger_sample=captured.trigger_sample,
            sample_rate_hz=captured.meta.config.sample_rate_hz,
            baseline=captured.meta.baseline,
            peak=captured.meta.peak,
            empty=captured.meta.empty,
            onset_ms_list=onset_ms_list,
        )

    def _show_previous_event(self) -> None:
        """Display the previous event in the history."""
        if self._current_event_index is not None and self._current_event_index > 0:
            self._current_event_index -= 1
            self._update_window_capture(self._history[self._current_event_index])
            self._refresh_status()

    def _show_next_event(self) -> None:
        """Navigates to the next captured event in history."""
        if self._current_event_index is None:
            return
        if self._current_event_index >= len(self._history) - 1:
            return
        self._current_event_index += 1
        self._update_window_capture(self._history[self._current_event_index])
        self._refresh_status()

    def _refresh_status(self) -> None:
        """Updates stream and navigation status indicators in the UI."""
        self._window.set_stream_state(
            sample_rate_hz=self._config.sample_rate_hz,
            captures=len(self._history),
            capturing=self._processor.is_capturing,
        )
        self._window.set_event_navigation(
            current_index=self._current_event_index,
            total_events=len(self._history),
        )
