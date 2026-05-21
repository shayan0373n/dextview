import unittest
from unittest.mock import MagicMock

import numpy as np

from dextview.config import DextViewConfig
from dextview.controller import DextViewController
from dextview.models import DataBatch, StreamMeta, ChannelInfo, ChannelKind, Channels
from dextview.ui import _mvc_bin_color

N_CHANNELS = 11
TRIGGER_CH = 10


def _make_meta(config: DextViewConfig) -> StreamMeta:
    """Create StreamMeta using a unified channels dict with one trigger channel."""
    parsed = {}
    for i in range(N_CHANNELS):
        if i == TRIGGER_CH:
            parsed[i] = ChannelInfo(label=f"ch{i}", kind=ChannelKind.TRIGGER)
        else:
            parsed[i] = ChannelInfo(label=f"ch{i}", kind=ChannelKind.FINGER)
    return StreamMeta(
        channels=Channels(parsed),
        config=config,
    )


def _make_controller() -> DextViewController:
    processor = MagicMock()
    processor.is_capturing = False
    config = DextViewConfig(n_channels=N_CHANNELS)
    meta = _make_meta(config)
    return DextViewController(
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


class ChannelsTests(unittest.TestCase):
    """Tests for Channels grouping behavior."""

    def test_by_kind_groups_and_sorts(self) -> None:
        """by_kind returns indices/labels sorted ascending by index, partitioned by kind."""
        channels = Channels({
            2: ChannelInfo(label="F1", kind=ChannelKind.FINGER),
            0: ChannelInfo(label="F0", kind=ChannelKind.FINGER),
            1: ChannelInfo(label="E0", kind=ChannelKind.EMG),
            3: ChannelInfo(label="T0", kind=ChannelKind.TRIGGER),
        })

        fingers = channels.by_kind(ChannelKind.FINGER)
        emgs = channels.by_kind(ChannelKind.EMG)
        triggers = channels.by_kind(ChannelKind.TRIGGER)

        self.assertEqual(fingers.indices, (0, 2))
        self.assertEqual(fingers.labels, ("F0", "F1"))
        self.assertEqual(emgs.indices, (1,))
        self.assertEqual(emgs.labels, ("E0",))
        self.assertEqual(triggers.indices, (3,))
        self.assertEqual(triggers.labels, ("T0",))

    def test_by_kind_returns_empty_group_for_missing_kind(self) -> None:
        """by_kind returns an empty ChannelGroup when no channel has that kind."""
        channels = Channels({0: ChannelInfo(label="F0", kind=ChannelKind.FINGER)})
        emgs = channels.by_kind(ChannelKind.EMG)
        self.assertEqual(emgs.indices, ())
        self.assertEqual(emgs.labels, ())


if __name__ == "__main__":
    unittest.main()
