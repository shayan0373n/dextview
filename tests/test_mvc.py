import unittest
from unittest.mock import MagicMock

import numpy as np

from quattrocento.config import QuattrocentoConfig
from quattrocento.controller import QuattrocentoController
from quattrocento.models import CapturedWindow, DataBatch
from quattrocento.ui import _mvc_bin_color


class MvcBinColorTests(unittest.TestCase):
    """_mvc_bin_color uses ≤ semantics; each boundary must land in its
    own bin, not the next one up."""

    def test_bin_boundaries_are_inclusive(self) -> None:
        # Exact thresholds fall into their own bin.
        self.assertEqual(_mvc_bin_color(5.0), "#A0AEC0")
        self.assertEqual(_mvc_bin_color(10.0), "#F6E05E")
        self.assertEqual(_mvc_bin_color(20.0), "#ED8936")
        self.assertEqual(_mvc_bin_color(40.0), "#E53E3E")
        self.assertEqual(_mvc_bin_color(60.0), "#9F7AEA")

    def test_just_above_threshold_promotes_to_next_bin(self) -> None:
        self.assertEqual(_mvc_bin_color(5.01), "#F6E05E")
        self.assertEqual(_mvc_bin_color(60.01), "#742A2A")

    def test_extremes(self) -> None:
        self.assertEqual(_mvc_bin_color(-1.0), "#A0AEC0")
        self.assertEqual(_mvc_bin_color(0.0), "#A0AEC0")
        self.assertEqual(_mvc_bin_color(1e9), "#742A2A")


def _make_controller() -> QuattrocentoController:
    """Build a controller with mocked collaborators. Window signals are
    MagicMocks whose `.connect` is a no-op, so no QApplication is needed."""
    processor = MagicMock()
    processor.is_capturing = False
    config = QuattrocentoConfig()
    return QuattrocentoController(
        config=config,
        stream=MagicMock(),
        processor=processor,
        window=MagicMock(),
    )


class ApplyMvcScalingTests(unittest.TestCase):
    def test_scales_to_percent_of_span_and_sets_flag(self) -> None:
        controller = _make_controller()
        n_fingers = controller._config.sensor_count
        controller._rest_means = np.zeros(n_fingers, dtype=np.float64)
        controller._mvc_maxs = np.full(n_fingers, 10.0, dtype=np.float64)

        forces = np.tile(
            np.array([[0.0], [5.0], [10.0]], dtype=np.float64), (1, n_fingers)
        )
        captured = CapturedWindow(
            timestamps=np.array([0.0, 0.1, 0.2], dtype=np.float64),
            finger_forces=forces,
            finger_ranges=np.full(n_fingers, 10.0, dtype=np.float64),
            finger_labels=controller._config.finger_labels,
        )

        scaled = controller._apply_mvc_scaling(captured)

        expected = np.tile(
            np.array([[0.0], [50.0], [100.0]], dtype=np.float64), (1, n_fingers)
        )
        np.testing.assert_allclose(scaled.finger_forces, expected)
        np.testing.assert_allclose(scaled.finger_ranges, np.full(n_fingers, 100.0))
        self.assertTrue(scaled.is_scaled)
        # Non-scaled fields pass through unchanged.
        np.testing.assert_allclose(scaled.timestamps, captured.timestamps)
        self.assertEqual(scaled.finger_labels, captured.finger_labels)

    def test_zero_span_raises(self) -> None:
        """A finger whose MVC equals its rest is a calibration failure;
        scaling must surface it loudly rather than divide-by-near-zero."""
        controller = _make_controller()
        n_fingers = controller._config.sensor_count
        controller._rest_means = np.full(n_fingers, 3.0, dtype=np.float64)
        controller._mvc_maxs = np.full(n_fingers, 3.0, dtype=np.float64)

        captured = CapturedWindow(
            timestamps=np.array([0.0, 0.1], dtype=np.float64),
            finger_forces=np.full((2, n_fingers), 3.0, dtype=np.float64),
            finger_ranges=np.zeros(n_fingers, dtype=np.float64),
            finger_labels=controller._config.finger_labels,
        )

        with self.assertRaises(ValueError):
            controller._apply_mvc_scaling(captured)


def _make_batch(forces: np.ndarray) -> DataBatch:
    return DataBatch(
        timestamps=np.arange(forces.shape[0], dtype=np.float64),
        forces=forces,
        aux_in=np.zeros(forces.shape[0], dtype=np.float64),
    )


class CalibrationToggleTests(unittest.TestCase):
    def test_rest_toggle_off_stores_mean_across_batches(self) -> None:
        controller = _make_controller()
        n_sensors = controller._config.sensor_count

        batch_a = np.full((4, n_sensors), 4.0, dtype=np.float64)
        batch_b = np.full((4, n_sensors), 6.0, dtype=np.float64)

        controller._on_rest_toggled(True)
        controller._update_rest(_make_batch(batch_a))
        controller._update_rest(_make_batch(batch_b))
        controller._on_rest_toggled(False)

        np.testing.assert_allclose(
            controller._rest_means, np.full(n_sensors, 5.0)
        )

    def test_mvc_toggle_off_stores_max_across_batches(self) -> None:
        controller = _make_controller()
        n_sensors = controller._config.sensor_count

        batch_a = np.full((4, n_sensors), 5.0, dtype=np.float64)
        batch_b = np.full((4, n_sensors), 5.0, dtype=np.float64)
        batch_b[0, :] = 9.0  # peak appears only in batch B

        controller._on_mvc_toggled(True)
        controller._update_mvc(_make_batch(batch_a))
        controller._update_mvc(_make_batch(batch_b))
        controller._on_mvc_toggled(False)

        np.testing.assert_allclose(
            controller._mvc_maxs, np.full(n_sensors, 9.0)
        )

    def test_empty_buffer_toggle_off_leaves_calibration_unset(self) -> None:
        controller = _make_controller()

        controller._on_rest_toggled(True)
        controller._on_rest_toggled(False)
        controller._on_mvc_toggled(True)
        controller._on_mvc_toggled(False)

        self.assertIsNone(controller._rest_means)
        self.assertIsNone(controller._mvc_maxs)

    def test_update_skips_empty_batch(self) -> None:
        controller = _make_controller()
        n_sensors = controller._config.sensor_count
        empty = DataBatch(
            timestamps=np.array([], dtype=np.float64),
            forces=np.empty((0, n_sensors), dtype=np.float64),
            aux_in=np.array([], dtype=np.float64),
        )

        controller._update_rest(empty)
        controller._update_mvc(empty)

        self.assertEqual(controller._rest_buffer, [])
        self.assertEqual(controller._mvc_buffer, [])


if __name__ == "__main__":
    unittest.main()
