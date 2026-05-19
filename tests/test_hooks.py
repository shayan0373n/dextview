import unittest
from unittest.mock import MagicMock

import numpy as np

from quattrocento.config import QuattrocentoConfig
from quattrocento.hooks import HoldInTargetAnyFinger, PassedThresholdAnyFinger
from quattrocento.hooks.logic import _HoldInBandDetector, _RampOnsetDetector
from quattrocento.models import DataBatch, StreamMeta


def _detector(
    threshold_pct: float = 10.0,
    onset_floor_pct: float = 3.0,
    onset_dwell_s: float = 0.0,
    min_elapsed_s: float = 0.0,
    release_dwell_s: float = 0.0,
) -> _RampOnsetDetector:
    return _RampOnsetDetector(
        threshold_pct=threshold_pct,
        onset_floor_pct=onset_floor_pct,
        onset_dwell_s=onset_dwell_s,
        min_elapsed_s=min_elapsed_s,
        release_dwell_s=release_dwell_s,
    )


def _hook(**kwargs) -> PassedThresholdAnyFinger:
    kwargs.setdefault("finger_indices", [0])
    kwargs.setdefault("pulse", MagicMock())
    h = PassedThresholdAnyFinger(**kwargs)
    h._hw = MagicMock()
    h._meter = MagicMock()
    h._active = True
    return h


def _meta(rest: float = 0.0, mvc_max: float = 100.0) -> StreamMeta:
    """StreamMeta with a single R Index finger at channel 0.

    With rest=0 and mvc_max=100 the force values equal % MVC directly."""
    return StreamMeta(
        channel_labels={0: "R Index"},
        config=QuattrocentoConfig(sample_rate_hz=2, n_channels=2),
        baseline=np.array([rest, 0.0]),
        peak=np.array([mvc_max, 1.0]),
    )


def _batch(forces_col0: list[float], timestamps: list[float]) -> DataBatch:
    ts = np.array(timestamps, dtype=np.float64)
    signals = np.zeros((len(ts), 2), dtype=np.float64)
    signals[:, 0] = forces_col0
    return DataBatch(timestamps=ts, signals=signals)


# ---------------------------------------------------------------------------
# _RampOnsetDetector
# ---------------------------------------------------------------------------

class OnsetDetectorInitialStateTests(unittest.TestCase):
    def test_onset_t_is_none_before_any_update(self) -> None:
        det = _detector()
        self.assertIsNone(det.onset_t)

    def test_update_returns_none_on_empty_array(self) -> None:
        det = _detector()
        result = det.update(np.array([]), np.array([]), sample_rate_hz=10)
        self.assertIsNone(result)


class OnsetDetectorOnsetLatchTests(unittest.TestCase):
    def test_no_onset_while_all_samples_below_floor(self) -> None:
        det = _detector(onset_floor_pct=3.0)
        pct = np.array([1.0, 2.0, 2.9])
        ts = np.arange(3, dtype=float)
        det.update(pct, ts, sample_rate_hz=1)
        self.assertIsNone(det.onset_t)

    def test_onset_latches_after_required_dwell_samples(self) -> None:
        det = _detector(onset_floor_pct=3.0, onset_dwell_s=0.2, min_elapsed_s=999)
        det.update(np.array([5.0]), np.array([0.0]), sample_rate_hz=10)
        self.assertIsNone(det.onset_t)

        det.update(np.array([5.0]), np.array([0.1]), sample_rate_hz=10)
        self.assertEqual(det.onset_t, 0.0)

    def test_onset_t_is_streak_start_not_latch_sample(self) -> None:
        det = _detector(onset_floor_pct=3.0, onset_dwell_s=0.2, min_elapsed_s=999)
        pct = np.array([5.0, 5.0, 5.0])
        ts = np.array([1.0, 1.1, 1.2])
        det.update(pct, ts, sample_rate_hz=10)
        self.assertEqual(det.onset_t, 1.0)


class OnsetDetectorReleaseTests(unittest.TestCase):
    def test_release_clears_onset_t_after_dwell_below_floor(self) -> None:
        det = _detector(onset_floor_pct=3.0, min_elapsed_s=999, release_dwell_s=0.0)
        det.update(np.array([5.0]), np.array([0.0]), sample_rate_hz=1)
        self.assertIsNotNone(det.onset_t)

        det.update(np.array([1.0]), np.array([1.0]), sample_rate_hz=1)
        self.assertIsNone(det.onset_t)

    def test_onset_t_survives_short_dip_below_floor(self) -> None:
        det = _detector(onset_floor_pct=3.0, min_elapsed_s=999, release_dwell_s=0.2)
        det.update(np.array([5.0]), np.array([0.0]), sample_rate_hz=10)
        self.assertIsNotNone(det.onset_t)

        det.update(np.array([1.0]), np.array([0.1]), sample_rate_hz=10)
        self.assertIsNotNone(det.onset_t)


class OnsetDetectorCrossingTests(unittest.TestCase):
    def test_crossing_accepted_when_elapsed_meets_minimum(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=2.0)
        pct = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 15.0])
        ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        result = det.update(pct, ts, sample_rate_hz=2)
        self.assertAlmostEqual(result, 2.5)

    def test_crossing_rejected_when_elapsed_below_minimum(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=2.0)
        pct = np.array([5.0, 15.0])
        ts = np.array([0.0, 0.5])
        result = det.update(pct, ts, sample_rate_hz=2)
        self.assertIsNone(result)

    def test_returns_first_crossing_timestamp_in_batch(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        pct = np.array([5.0, 15.0, 20.0])
        ts = np.array([0.0, 1.0, 2.0])
        result = det.update(pct, ts, sample_rate_hz=1)
        self.assertAlmostEqual(result, 1.0)

    def test_no_crossing_when_all_samples_below_threshold(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        pct = np.array([5.0, 8.0, 9.9])
        ts = np.arange(3, dtype=float)
        result = det.update(pct, ts, sample_rate_hz=1)
        self.assertIsNone(result)


class OnsetDetectorOneShotTests(unittest.TestCase):
    def test_subsequent_update_returns_none_after_fire(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        pct = np.array([5.0, 15.0])
        ts = np.array([0.0, 1.0])
        first = det.update(pct, ts, sample_rate_hz=1)
        self.assertIsNotNone(first)

        second = det.update(pct, ts, sample_rate_hz=1)
        self.assertIsNone(second)

    def test_onset_t_persists_after_fire_for_elapsed_calculation(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        det.update(np.array([5.0, 15.0]), np.array([0.0, 1.0]), sample_rate_hz=1)
        self.assertIsNotNone(det.onset_t)

    def test_reset_clears_fired_flag_and_rearms(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        pct = np.array([5.0, 15.0])
        ts = np.array([0.0, 1.0])
        det.update(pct, ts, sample_rate_hz=1)
        det.reset()

        self.assertIsNone(det.onset_t)
        result = det.update(pct, ts, sample_rate_hz=1)
        self.assertIsNotNone(result)


class OnsetDetectorAutoRearmTests(unittest.TestCase):
    def test_release_after_fire_clears_fired_flag(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        det.update(np.array([5.0, 15.0]), np.array([0.0, 1.0]), sample_rate_hz=1)
        self.assertIsNone(
            det.update(np.array([15.0]), np.array([1.5]), sample_rate_hz=1)
        )

        det.update(np.array([1.0, 1.0]), np.array([2.0, 3.0]), sample_rate_hz=1)
        self.assertFalse(det._fired)
        self.assertIsNone(det.onset_t)

    def test_second_ramp_fires_after_auto_rearm(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        first = det.update(
            np.array([5.0, 15.0]), np.array([0.0, 1.0]), sample_rate_hz=1
        )
        self.assertAlmostEqual(first, 1.0)

        det.update(np.array([1.0, 1.0]), np.array([2.0, 3.0]), sample_rate_hz=1)

        second = det.update(
            np.array([5.0, 15.0]), np.array([4.0, 5.0]), sample_rate_hz=1
        )
        self.assertAlmostEqual(second, 5.0)

    def test_high_force_after_fire_does_not_rearm(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        det.update(np.array([5.0, 15.0]), np.array([0.0, 1.0]), sample_rate_hz=1)
        self.assertTrue(det._fired)

        det.update(np.array([15.0, 15.0]), np.array([2.0, 3.0]), sample_rate_hz=1)
        self.assertTrue(det._fired)


class OnsetDetectorSetThresholdTests(unittest.TestCase):
    def test_set_threshold_clears_onset_and_fired_state(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        det.update(np.array([5.0, 15.0]), np.array([0.0, 1.0]), sample_rate_hz=1)
        self.assertIsNotNone(det.onset_t)

        det.set_threshold(20.0)
        self.assertIsNone(det.onset_t)

    def test_new_threshold_applies_to_subsequent_updates(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        det.set_threshold(20.0)
        result = det.update(
            np.array([5.0, 15.0]), np.array([0.0, 1.0]), sample_rate_hz=1
        )
        self.assertIsNone(result)
        result = det.update(
            np.array([5.0, 25.0]), np.array([2.0, 3.0]), sample_rate_hz=1
        )
        self.assertAlmostEqual(result, 3.0)


# ---------------------------------------------------------------------------
# PassedThresholdAnyFinger (compositor)
# ---------------------------------------------------------------------------

class CompositorInactiveTests(unittest.TestCase):
    def test_inactive_hook_does_not_touch_hw(self) -> None:
        h = PassedThresholdAnyFinger(finger_indices=[0], pulse=MagicMock())
        h._hw = MagicMock()
        h._meter = MagicMock()
        h(_batch([5.0, 15.0], [0.0, 1.0]), _meta())
        h._hw.fire.assert_not_called()


class CompositorCalibrationTests(unittest.TestCase):
    def test_missing_baseline_sets_meter_status(self) -> None:
        h = _hook()
        meta = StreamMeta(
            channel_labels={0: "R Index"},
            config=QuattrocentoConfig(sample_rate_hz=2, n_channels=2),
            baseline=None,
            peak=np.array([100.0, 1.0]),
        )
        h(_batch([15.0], [0.0]), meta)
        h._meter.set_status.assert_called_with("Calibration missing")
        h._hw.fire.assert_not_called()

    def test_zero_mvc_span_sets_meter_status(self) -> None:
        h = _hook()
        meta = _meta(rest=50.0, mvc_max=50.0)
        h(_batch([50.0], [0.0]), meta)
        h._meter.set_status.assert_called_with("Zero MVC span — recalibrate")
        h._hw.fire.assert_not_called()


class CompositorFiringTests(unittest.TestCase):
    def test_hw_fires_on_qualifying_crossing(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, onset_dwell_s=0.0, min_elapsed_s=2.0)
        batch = _batch([5.0, 5.0, 5.0, 5.0, 5.0, 15.0], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        h(batch, _meta())
        h._hw.fire.assert_called_once()

    def test_hw_not_fired_when_crossing_too_early(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, onset_dwell_s=0.0, min_elapsed_s=2.0)
        batch = _batch([5.0, 15.0], [0.0, 0.5])
        h(batch, _meta())
        h._hw.fire.assert_not_called()

    def test_hw_fires_only_once_across_multiple_batches(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, onset_dwell_s=0.0, min_elapsed_s=0.0)
        meta = _meta()
        batch = _batch([5.0, 15.0], [0.0, 1.0])
        h(batch, meta)
        h(batch, meta)
        self.assertEqual(h._hw.fire.call_count, 1)

    def test_meter_pct_updated_every_batch(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=999.0)
        meta = _meta()
        h(_batch([7.0], [0.0]), meta)
        h(_batch([8.0], [1.0]), meta)
        self.assertEqual(h._meter.set_pct.call_count, 2)


class CompositorSetThresholdTests(unittest.TestCase):
    def test_set_threshold_propagates_to_detector_and_meter(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=0.0)
        h.set_threshold(20.0)
        self.assertEqual(h._threshold_pct, 20.0)
        self.assertEqual(h._detector._threshold_pct, 20.0)
        h._meter.set_threshold.assert_called_with(20.0)

    def test_set_threshold_resets_detector_streak(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=999.0)
        meta = _meta()
        h(_batch([5.0, 5.0, 5.0], [0.0, 0.5, 1.0]), meta)
        self.assertIsNotNone(h._detector.onset_t)

        h.set_threshold(15.0)
        self.assertIsNone(h._detector.onset_t)

    def test_new_threshold_does_not_fire_on_old_streak(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=2.0)
        meta = _meta()
        # Establish an old onset at t=0.0; after >2 s the min-elapsed gate
        # is satisfied for *that* onset.
        h(_batch([5.0, 5.0, 5.0, 5.0, 5.0], [0.0, 0.5, 1.0, 1.5, 2.0]), meta)
        self.assertIsNotNone(h._detector.onset_t)

        # Raising the threshold must reset the onset clock — otherwise a
        # crossing of the new threshold at t=3 s would fire (elapsed=3 s
        # against the stale t=0 onset).
        h.set_threshold(20.0)
        h(_batch([5.0, 22.0], [2.5, 3.0]), meta)
        h._hw.fire.assert_not_called()


# ---------------------------------------------------------------------------
# _HoldInBandDetector
# ---------------------------------------------------------------------------

def _hold_detector(
    target_pct: float = 30.0,
    tolerance_rel: float = 0.10,
    dwell_s: float = 2.0,
) -> _HoldInBandDetector:
    return _HoldInBandDetector(
        target_pct=target_pct,
        tolerance_rel=tolerance_rel,
        dwell_s=dwell_s,
    )


class HoldDetectorBandTests(unittest.TestCase):
    def test_band_bounds_match_relative_tolerance(self) -> None:
        det = _hold_detector(target_pct=30.0, tolerance_rel=0.10)
        self.assertAlmostEqual(det.low_pct, 27.0)
        self.assertAlmostEqual(det.high_pct, 33.0)

    def test_update_returns_none_on_empty_array(self) -> None:
        det = _hold_detector()
        result = det.update(np.array([]), np.array([]))
        self.assertIsNone(result)


class HoldDetectorTimerTests(unittest.TestCase):
    def test_in_band_sample_starts_timer(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=2.0)
        det.update(np.array([30.0]), np.array([0.0]))
        self.assertEqual(det.time_in_band_s, 0.0)

    def test_time_in_band_grows_with_consecutive_in_band_samples(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=999.0)
        det.update(np.array([30.0, 30.0, 30.0]), np.array([0.0, 0.5, 1.0]))
        self.assertAlmostEqual(det.time_in_band_s, 1.0)

    def test_out_of_band_sample_resets_timer(self) -> None:
        det = _hold_detector(target_pct=30.0, tolerance_rel=0.10, dwell_s=999.0)
        det.update(np.array([30.0, 30.0]), np.array([0.0, 0.5]))
        self.assertGreater(det.time_in_band_s, 0.0)
        det.update(np.array([10.0]), np.array([1.0]))
        self.assertEqual(det.time_in_band_s, 0.0)


class HoldDetectorFiringTests(unittest.TestCase):
    def test_fires_when_time_in_band_reaches_dwell(self) -> None:
        det = _hold_detector(target_pct=30.0, tolerance_rel=0.10, dwell_s=2.0)
        pct = np.array([30.0, 30.0, 30.0, 30.0, 30.0])
        ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = det.update(pct, ts)
        self.assertAlmostEqual(result, 2.0)

    def test_does_not_fire_when_dwell_not_yet_reached(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=2.0)
        pct = np.array([30.0, 30.0])
        ts = np.array([0.0, 1.0])
        result = det.update(pct, ts)
        self.assertIsNone(result)

    def test_value_outside_band_does_not_fire(self) -> None:
        det = _hold_detector(target_pct=30.0, tolerance_rel=0.10, dwell_s=0.0)
        pct = np.array([20.0, 35.0])  # below band, above band
        ts = np.array([0.0, 0.5])
        result = det.update(pct, ts)
        self.assertIsNone(result)

    def test_does_not_refire_before_full_dwell_elapses(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=1.0)
        first = det.update(
            np.array([30.0, 30.0, 30.0]), np.array([0.0, 0.5, 1.0])
        )
        self.assertIsNotNone(first)
        # After fire the timer restarts at t=1.0; t=1.5 is only 0.5s later.
        second = det.update(np.array([30.0]), np.array([1.5]))
        self.assertIsNone(second)

    def test_refires_after_dwell_post_fire_during_continuous_hold(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=1.0)
        first = det.update(
            np.array([30.0, 30.0, 30.0]), np.array([0.0, 0.5, 1.0])
        )
        self.assertAlmostEqual(first, 1.0)
        # Still holding; another full dwell window must produce a second fire.
        second = det.update(np.array([30.0, 30.0]), np.array([1.5, 2.0]))
        self.assertAlmostEqual(second, 2.0)

    def test_timer_resets_to_zero_after_fire(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=1.0)
        det.update(np.array([30.0, 30.0, 30.0]), np.array([0.0, 0.5, 1.0]))
        # Time-in-band should reset from the fire instant, not keep growing.
        self.assertEqual(det.time_in_band_s, 0.0)

    def test_rearms_after_leaving_band(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=1.0)
        det.update(np.array([30.0, 30.0, 30.0]), np.array([0.0, 0.5, 1.0]))
        det.update(np.array([10.0]), np.array([1.5]))
        result = det.update(
            np.array([30.0, 30.0, 30.0]), np.array([2.0, 2.5, 3.0])
        )
        self.assertAlmostEqual(result, 3.0)


class HoldDetectorSetTargetTests(unittest.TestCase):
    def test_set_target_resets_streak(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=999.0)
        det.update(np.array([30.0, 30.0]), np.array([0.0, 0.5]))
        self.assertGreater(det.time_in_band_s, 0.0)
        det.set_target(50.0)
        self.assertEqual(det.time_in_band_s, 0.0)

    def test_set_dwell_resets_streak(self) -> None:
        det = _hold_detector(target_pct=30.0, dwell_s=999.0)
        det.update(np.array([30.0, 30.0]), np.array([0.0, 0.5]))
        self.assertGreater(det.time_in_band_s, 0.0)
        det.set_dwell(5.0)
        self.assertEqual(det.time_in_band_s, 0.0)


# ---------------------------------------------------------------------------
# HoldInTargetAnyFinger (compositor)
# ---------------------------------------------------------------------------

def _hold_hook(**kwargs) -> HoldInTargetAnyFinger:
    kwargs.setdefault("finger_indices", [0])
    kwargs.setdefault("pulse", MagicMock())
    h = HoldInTargetAnyFinger(**kwargs)
    h._hw = MagicMock()
    h._meter = MagicMock()
    h._active = True
    return h


class HoldCompositorFiringTests(unittest.TestCase):
    def test_hw_fires_when_held_in_band_long_enough(self) -> None:
        h = _hold_hook(target_pct=30.0, tolerance_rel=0.10, dwell_s=1.0)
        # band is 27%–33%; sustain at 30% for ≥1 s.
        batch = _batch([30.0, 30.0, 30.0], [0.0, 0.5, 1.0])
        h(batch, _meta())
        h._hw.fire.assert_called_once()

    def test_hw_not_fired_when_force_outside_band(self) -> None:
        h = _hold_hook(target_pct=30.0, tolerance_rel=0.10, dwell_s=0.0)
        batch = _batch([20.0, 40.0], [0.0, 0.5])
        h(batch, _meta())
        h._hw.fire.assert_not_called()

    def test_hw_not_fired_when_dwell_not_yet_met(self) -> None:
        h = _hold_hook(target_pct=30.0, dwell_s=2.0)
        batch = _batch([30.0, 30.0], [0.0, 1.0])
        h(batch, _meta())
        h._hw.fire.assert_not_called()


class HoldCompositorSetTargetTests(unittest.TestCase):
    def test_set_target_propagates_to_detector_and_meter(self) -> None:
        h = _hold_hook(target_pct=30.0, dwell_s=1.0)
        h.set_target(50.0)
        self.assertEqual(h._target_pct, 50.0)
        self.assertEqual(h._detector._target_pct, 50.0)
        h._meter.set_target.assert_called_with(50.0)

    def test_set_dwell_propagates_to_detector_and_meter(self) -> None:
        h = _hold_hook(target_pct=30.0, dwell_s=1.0)
        h.set_dwell(3.0)
        self.assertEqual(h._dwell_s, 3.0)
        self.assertEqual(h._detector._dwell_s, 3.0)
        h._meter.set_dwell.assert_called_with(3.0)


if __name__ == "__main__":
    unittest.main()
