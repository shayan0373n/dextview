import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger("quattrocento.hooks")


class LabJackPulse:
    """Manages a LabJack T4 connection and fires a 5 ms TTL pulse on FIO4.

    To avoid latency spikes on the first threshold crossing/fire (which could
    violate the <10 ms latency spec due to LJM USB discovery delays), this
    connection must be opened upfront on hook activation (via set_active(True)
    in the compositors) rather than opened lazily during fire().
    """

    def __init__(self) -> None:
        self._ljm: Any = None
        self._handle: Any = None

    def open(self) -> None:
        """Open the LabJack T4 connection upfront on activation.
        No-op if already open, so safe to call from multiple compositors
        sharing the same instance.
        """
        if self._handle is not None:
            return
        from labjack import ljm
        self._ljm = ljm
        self._handle = ljm.openS("T4", "ANY", "ANY")

    def close(self) -> None:
        """Close the LabJack connection."""
        if self._handle is not None:
            self._ljm.close(self._handle)
            self._handle = None
            self._ljm = None

    def fire(self) -> None:
        """Fire a 5 ms TTL pulse on FIO4. Assumes open() has already been called."""
        self._ljm.eWriteName(self._handle, "FIO4", 1)
        time.sleep(0.005)
        self._ljm.eWriteName(self._handle, "FIO4", 0)


class _RampOnsetDetector:
    """Pure onset/release/threshold-crossing state machine. No I/O or Qt."""

    def __init__(
        self,
        threshold_pct: float,
        onset_floor_pct: float,
        onset_dwell_s: float,
        min_elapsed_s: float,
        release_dwell_s: float,
    ) -> None:
        self._threshold_pct = threshold_pct
        self._onset_floor_pct = onset_floor_pct
        self._onset_dwell_s = onset_dwell_s
        self._min_elapsed_s = min_elapsed_s
        self._release_dwell_s = release_dwell_s
        self._streak_samples = 0
        self._streak_start_t: float | None = None
        self._below_streak = 0
        self._ramp_onset_t: float | None = None
        self._fired = False

    @property
    def onset_t(self) -> float | None:
        return self._ramp_onset_t

    def reset(self) -> None:
        """Clears all streak and onset state, resetting the detector to idle."""
        self._streak_samples = 0
        self._streak_start_t = None
        self._below_streak = 0
        self._ramp_onset_t = None
        self._fired = False

    def set_threshold(self, pct: float) -> None:
        """Update the crossing threshold and reset all streak/onset state.

        Resetting prevents the new threshold from firing on a streak that
        built up under the old one.
        """
        self._threshold_pct = pct
        self.reset()

    def update(
        self,
        pct: np.ndarray,
        timestamps: np.ndarray,
        sample_rate_hz: int,
    ) -> float | None:
        """Advance by one batch. Returns the crossing timestamp the first time
        the threshold is crossed with sufficient elapsed time since onset, else None.
        onset_t remains set after firing so callers can compute elapsed.

        Auto-re-arms when force returns below `onset_floor_pct` for
        `release_dwell_s` after a fire (release branch clears `_fired`).
        At most one crossing is returned per call: if a batch contains both
        a release and a new crossing, the second fire is deferred to the
        next batch."""
        self._update_onset_state(pct, timestamps, sample_rate_hz)

        if self._fired:
            return None

        if self._ramp_onset_t is None:
            return None

        crossings = np.flatnonzero(pct >= self._threshold_pct)
        if crossings.size == 0:
            return None
        crossing_t = float(timestamps[crossings[0]])
        elapsed = crossing_t - self._ramp_onset_t
        if elapsed < self._min_elapsed_s:
            logger.debug(
                f"{self._threshold_pct:.0f}% crossing at t={crossing_t:.3f}s "
                f"rejected — only {elapsed:.3f}s since onset "
                f"(min {self._min_elapsed_s:.1f}s)"
            )
            return None

        self._fired = True
        return crossing_t

    def _update_onset_state(
        self,
        pct: np.ndarray,
        timestamps: np.ndarray,
        sample_rate_hz: int,
    ) -> None:
        """Advances the onset/release streak counters by one batch of samples."""
        onset_dwell = max(1, int(round(self._onset_dwell_s * sample_rate_hz)))
        release_dwell = max(1, int(round(self._release_dwell_s * sample_rate_hz)))
        floor = self._onset_floor_pct

        for k in range(pct.shape[0]):
            if pct[k] >= floor:
                self._below_streak = 0
                if self._streak_samples == 0:
                    self._streak_start_t = float(timestamps[k])
                self._streak_samples += 1
                if (
                    self._ramp_onset_t is None
                    and self._streak_samples >= onset_dwell
                ):
                    self._ramp_onset_t = self._streak_start_t
                    return  # remaining samples in batch still checked for threshold crossing in update()
            else:
                self._streak_samples = 0
                self._streak_start_t = None
                if self._ramp_onset_t is not None:
                    self._below_streak += 1
                    if self._below_streak >= release_dwell:
                        self._ramp_onset_t = None
                        self._below_streak = 0
                        self._fired = False


class _HoldInBandDetector:
    """Pure hold-in-band state machine. No I/O or Qt.

    Tracks how long the input has been inside [target × (1-rel), target × (1+rel)].
    Fires whenever the in-band streak reaches `dwell_s`; on each fire the
    timer restarts from the fire instant, so a continuous hold produces
    periodic fires every `dwell_s`. Any out-of-band sample resets the timer."""

    def __init__(
        self,
        target_pct: float,
        tolerance_rel: float,
        dwell_s: float,
    ) -> None:
        self._target_pct = target_pct
        self._tolerance_rel = tolerance_rel
        self._dwell_s = dwell_s
        self._in_band_start_t: float | None = None
        self._time_in_band_s: float = 0.0

    @property
    def time_in_band_s(self) -> float:
        return self._time_in_band_s

    @property
    def in_band(self) -> bool:
        """True when the input is currently inside the target band."""
        return self._in_band_start_t is not None

    @property
    def low_pct(self) -> float:
        return self._target_pct * (1.0 - self._tolerance_rel)

    @property
    def high_pct(self) -> float:
        return self._target_pct * (1.0 + self._tolerance_rel)

    def reset(self) -> None:
        """Clears the in-band streak."""
        self._in_band_start_t = None
        self._time_in_band_s = 0.0

    def set_target(self, pct: float) -> None:
        """Update the target % and reset the streak. Resetting prevents the
        new band from firing on a streak built up under the old one."""
        self._target_pct = pct
        self.reset()

    def set_dwell(self, s: float) -> None:
        """Update the required dwell duration and reset the streak. Resetting
        avoids an immediate fire if dwell is shortened below the current
        time-in-band."""
        self._dwell_s = s
        self.reset()

    def update(
        self,
        pct: np.ndarray,
        timestamps: np.ndarray,
    ) -> float | None:
        """Advance by one batch. Returns the timestamp at which the in-band
        streak first reaches `dwell_s`, else None. On fire the timer
        restarts from that instant so a sustained hold re-fires every
        `dwell_s`. Any out-of-band sample resets the timer.

        At most one fire is returned per call; if a sustained hold spans
        multiple `dwell_s` intervals within a single batch (only possible
        when the batch is wider than `dwell_s`), the second fire is deferred
        to the next batch."""
        if pct.shape[0] == 0:
            return None

        low = self.low_pct
        high = self.high_pct
        fired_t: float | None = None

        for k in range(pct.shape[0]):
            t = float(timestamps[k])
            in_band = low <= pct[k] <= high
            if in_band:
                if self._in_band_start_t is None:
                    self._in_band_start_t = t
                    self._time_in_band_s = 0.0
                else:
                    self._time_in_band_s = t - self._in_band_start_t
                if self._time_in_band_s >= self._dwell_s and fired_t is None:
                    fired_t = t
                    self._in_band_start_t = t
                    self._time_in_band_s = 0.0
            else:
                self._in_band_start_t = None
                self._time_in_band_s = 0.0

        return fired_t
