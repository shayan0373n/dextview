import logging

import numpy as np

from ..models import DataBatch, StreamMeta
from .logic import _HoldInBandDetector, LabJackPulse, _RampOnsetDetector
from .ui import _ForceMeterDialog, _HoldTargetDialog

logger = logging.getLogger("quattrocento.hooks")


class PassedThresholdAnyFinger:
    """Compositor: wires _RampOnsetDetector → _ForceMeterDialog → LabJackPulse.

    Stays inert until `set_active(True)`. On activation a small always-on-top
    dialog opens showing the max % MVC across all finger channels and a
    spinbox to edit the firing threshold live; the dialog closes on
    deactivation. Fires when any finger first crosses the threshold after
    sufficient time since onset. After firing, auto-re-arms once force drops
    below `onset_floor_pct` for `release_dwell_s`; a new ramp then triggers
    a fresh fire.
    """

    name = "Any Finger Threshold"

    def __init__(
        self,
        finger_indices: list[int],
        pulse: LabJackPulse,
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
        self._hw = pulse
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
                self._meter.reset_requested.connect(self.reset)
            self._meter.set_pct(0.0)
            self._meter.set_status("Armed — waiting for onset")
            self._meter.show()
            self._meter.raise_()
        else:
            self._active = False
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


class HoldInTargetAnyFinger:
    """Compositor: wires _HoldInBandDetector → _HoldTargetDialog → LabJackPulse.

    Stays inert until `set_active(True)`. On activation a small always-on-top
    dialog opens showing the max % MVC across all finger channels, the
    configured target band, and a horizontal dwell-progress bar. Fires when
    the % MVC has been continuously inside the target band for `dwell_s`.
    During a sustained hold, re-fires every `dwell_s`; any out-of-band sample
    resets the timer. (Contrast: PassedThresholdAnyFinger requires a release
    dip before re-arming.)
    """

    name = "Hold In Target"

    def __init__(
        self,
        finger_indices: list[int],
        pulse: LabJackPulse,
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
        self._hw = pulse
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
                self._meter.reset_requested.connect(self.reset)
            self._meter.clear_trace()
            self._meter.set_time_in_band(0.0)
            self._meter.set_status("Armed — waiting to enter band")
            self._meter.show()
            self._meter.raise_()
        else:
            self._active = False
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
