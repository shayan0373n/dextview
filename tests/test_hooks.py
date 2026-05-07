import unittest
from unittest.mock import MagicMock

import numpy as np

from quattrocento.hooks import PassedTenPercentRightIndex, _RampOnsetDetector
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


def _hook(**kwargs) -> PassedTenPercentRightIndex:
    """Build a compositor with mocked hw and meter so no Qt or LabJack needed."""
    h = PassedTenPercentRightIndex(**kwargs)
    h._hw = MagicMock()
    h._meter = MagicMock()
    h._active = True
    return h


def _meta(rest: float = 0.0, mvc_max: float = 100.0) -> StreamMeta:
    """StreamMeta with a single R Index finger at sensor 0.

    With rest=0 and mvc_max=100 the force values equal % MVC directly."""
    return StreamMeta(
        finger_sensor_map={"R Index": 0},
        finger_labels=("R Index",),
        sample_rate_hz=2,
        rest_means=np.array([rest]),
        mvc_maxs=np.array([mvc_max]),
    )


def _batch(forces_col0: list[float], timestamps: list[float]) -> DataBatch:
    ts = np.array(timestamps, dtype=np.float64)
    forces = np.array([[f] for f in forces_col0], dtype=np.float64)
    return DataBatch(timestamps=ts, forces=forces, aux_in=np.zeros(len(ts)))


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
        # dwell_s=0.2 at 10 Hz → 2 samples needed
        det = _detector(onset_floor_pct=3.0, onset_dwell_s=0.2, min_elapsed_s=999)
        det.update(np.array([5.0]), np.array([0.0]), sample_rate_hz=10)
        self.assertIsNone(det.onset_t)  # one sample: not yet

        det.update(np.array([5.0]), np.array([0.1]), sample_rate_hz=10)
        self.assertEqual(det.onset_t, 0.0)  # two samples: latched at streak start

    def test_onset_t_is_streak_start_not_latch_sample(self) -> None:
        # onset_t records when the streak *started*, not when dwell was satisfied
        det = _detector(onset_floor_pct=3.0, onset_dwell_s=0.2, min_elapsed_s=999)
        pct = np.array([5.0, 5.0, 5.0])
        ts = np.array([1.0, 1.1, 1.2])
        det.update(pct, ts, sample_rate_hz=10)
        self.assertEqual(det.onset_t, 1.0)


class OnsetDetectorReleaseTests(unittest.TestCase):
    def test_release_clears_onset_t_after_dwell_below_floor(self) -> None:
        # release_dwell_s=0 → 1 sample below floor is enough
        det = _detector(onset_floor_pct=3.0, min_elapsed_s=999, release_dwell_s=0.0)
        det.update(np.array([5.0]), np.array([0.0]), sample_rate_hz=1)
        self.assertIsNotNone(det.onset_t)

        det.update(np.array([1.0]), np.array([1.0]), sample_rate_hz=1)
        self.assertIsNone(det.onset_t)

    def test_onset_t_survives_short_dip_below_floor(self) -> None:
        # release_dwell_s=0.2 at 10 Hz → 2 samples needed to release
        det = _detector(onset_floor_pct=3.0, min_elapsed_s=999, release_dwell_s=0.2)
        det.update(np.array([5.0]), np.array([0.0]), sample_rate_hz=10)
        self.assertIsNotNone(det.onset_t)

        # one sample below floor — not long enough to release
        det.update(np.array([1.0]), np.array([0.1]), sample_rate_hz=10)
        self.assertIsNotNone(det.onset_t)


class OnsetDetectorCrossingTests(unittest.TestCase):
    def test_crossing_accepted_when_elapsed_meets_minimum(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=2.0)
        # onset at t=0.0, crossing at t=2.5 → elapsed=2.5 ≥ 2.0
        pct = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 15.0])
        ts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        result = det.update(pct, ts, sample_rate_hz=2)
        self.assertAlmostEqual(result, 2.5)

    def test_crossing_rejected_when_elapsed_below_minimum(self) -> None:
        det = _detector(threshold_pct=10.0, onset_floor_pct=3.0, min_elapsed_s=2.0)
        # onset at t=0.0, crossing at t=0.5 → elapsed=0.5 < 2.0
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
        det.update(pct, ts, sample_rate_hz=1)  # fires
        det.reset()

        self.assertIsNone(det.onset_t)
        result = det.update(pct, ts, sample_rate_hz=1)
        self.assertIsNotNone(result)  # can fire again after reset


# ---------------------------------------------------------------------------
# PassedTenPercentRightIndex (compositor)
# ---------------------------------------------------------------------------

class CompositorInactiveTests(unittest.TestCase):
    def test_inactive_hook_does_not_touch_hw(self) -> None:
        h = PassedTenPercentRightIndex()
        h._hw = MagicMock()
        h._meter = MagicMock()
        # _active is False by default
        h(_batch([5.0, 15.0], [0.0, 1.0]), _meta())
        h._hw.fire.assert_not_called()


class CompositorCalibrationTests(unittest.TestCase):
    def test_missing_rest_means_sets_meter_status(self) -> None:
        h = _hook()
        meta = StreamMeta(
            finger_sensor_map={"R Index": 0},
            finger_labels=("R Index",),
            sample_rate_hz=2,
            rest_means=None,
            mvc_maxs=np.array([100.0]),
        )
        h(_batch([15.0], [0.0]), meta)
        h._meter.set_status.assert_called_with("Calibration missing")
        h._hw.fire.assert_not_called()

    def test_zero_mvc_span_sets_meter_status(self) -> None:
        h = _hook()
        # rest == mvc_max → span is zero
        meta = _meta(rest=50.0, mvc_max=50.0)
        h(_batch([50.0], [0.0]), meta)
        h._meter.set_status.assert_called_with("Zero MVC span — recalibrate")
        h._hw.fire.assert_not_called()


class CompositorFiringTests(unittest.TestCase):
    def test_hw_fires_on_qualifying_crossing(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, onset_dwell_s=0.0, min_elapsed_s=2.0)
        # onset at t=0.0, crossing at t=2.5 → elapsed=2.5 ≥ 2.0
        batch = _batch([5.0, 5.0, 5.0, 5.0, 5.0, 15.0], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        h(batch, _meta())
        h._hw.fire.assert_called_once()

    def test_hw_not_fired_when_crossing_too_early(self) -> None:
        h = _hook(threshold_pct=10.0, onset_floor_pct=3.0, onset_dwell_s=0.0, min_elapsed_s=2.0)
        # onset at t=0.0, crossing at t=0.5 → elapsed=0.5 < 2.0
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


if __name__ == "__main__":
    unittest.main()
