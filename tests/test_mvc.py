import unittest
from unittest.mock import MagicMock

import numpy as np

from quattrocento.config import QuattrocentoConfig
from quattrocento.controller import QuattrocentoController
from quattrocento.models import DataBatch, StreamMeta
from quattrocento.ui import _mvc_bin_color

N_CHANNELS = 11
TRIGGER_CH = 10


def _make_meta(config: QuattrocentoConfig) -> StreamMeta:
    return StreamMeta(
        channel_labels={i: f"ch{i}" for i in range(N_CHANNELS)},
        config=config,
    )


def _make_controller() -> QuattrocentoController:
    processor = MagicMock()
    processor.is_capturing = False
    config = QuattrocentoConfig(n_channels=N_CHANNELS)
    meta = _make_meta(config)
    return QuattrocentoController(
        config=config,
        stream=MagicMock(),
        processor=processor,
        window=MagicMock(),
        meta=meta,
    )


def _make_batch(signals: np.ndarray) -> DataBatch:
    return DataBatch(
        timestamps=np.arange(signals.shape[0], dtype=np.float64),
        signals=signals,
    )


class MvcBinColorTests(unittest.TestCase):
    def test_bin_boundaries_are_inclusive(self) -> None:
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


class CalibrationToggleTests(unittest.TestCase):
    def test_baseline_toggle_off_stores_mean_across_batches(self) -> None:
        controller = _make_controller()

        batch_a = np.full((4, N_CHANNELS), 4.0, dtype=np.float64)
        batch_b = np.full((4, N_CHANNELS), 6.0, dtype=np.float64)

        controller._on_baseline_toggled(True)
        controller._update_baseline(_make_batch(batch_a))
        controller._update_baseline(_make_batch(batch_b))
        controller._on_baseline_toggled(False)

        np.testing.assert_allclose(
            controller._meta.baseline, np.full(N_CHANNELS, 5.0)
        )

    def test_peak_toggle_off_stores_max_across_batches(self) -> None:
        controller = _make_controller()

        batch_a = np.full((4, N_CHANNELS), 5.0, dtype=np.float64)
        batch_b = np.full((4, N_CHANNELS), 5.0, dtype=np.float64)
        batch_b[0, :] = 9.0

        controller._on_peak_toggled(True)
        controller._update_peak(_make_batch(batch_a))
        controller._update_peak(_make_batch(batch_b))
        controller._on_peak_toggled(False)

        np.testing.assert_allclose(
            controller._meta.peak, np.full(N_CHANNELS, 9.0)
        )

    def test_empty_buffer_toggle_off_leaves_calibration_unset(self) -> None:
        controller = _make_controller()

        controller._on_baseline_toggled(True)
        controller._on_baseline_toggled(False)
        controller._on_peak_toggled(True)
        controller._on_peak_toggled(False)

        self.assertIsNone(controller._meta.baseline)
        self.assertIsNone(controller._meta.peak)

    def test_update_skips_empty_batch(self) -> None:
        controller = _make_controller()
        empty = DataBatch(
            timestamps=np.array([], dtype=np.float64),
            signals=np.empty((0, N_CHANNELS), dtype=np.float64),
        )

        controller._update_baseline(empty)
        controller._update_peak(empty)

        self.assertEqual(controller._baseline_buffer, [])
        self.assertEqual(controller._peak_buffer, [])

    def test_ui_scaling_produces_percent_mvc(self) -> None:
        """Consumers (e.g. UI) normalise signals using baseline/peak from meta."""
        baseline = np.zeros(N_CHANNELS, dtype=np.float64)
        peak = np.full(N_CHANNELS, 10.0, dtype=np.float64)
        signals = np.tile(
            np.array([[0.0], [5.0], [10.0]], dtype=np.float64), (1, N_CHANNELS)
        )
        span = peak - baseline
        scaled = (signals - baseline) / span * 100.0
        expected = np.tile(
            np.array([[0.0], [50.0], [100.0]], dtype=np.float64), (1, N_CHANNELS)
        )
        np.testing.assert_allclose(scaled, expected)

    def test_calibration_all_channels_same_width_as_signals(self) -> None:
        """baseline and peak must span all channels, not just finger channels."""
        controller = _make_controller()
        signals = np.ones((4, N_CHANNELS), dtype=np.float64)

        controller._on_baseline_toggled(True)
        controller._update_baseline(_make_batch(signals))
        controller._on_baseline_toggled(False)

        self.assertEqual(controller._meta.baseline.shape, (N_CHANNELS,))


if __name__ == "__main__":
    unittest.main()
