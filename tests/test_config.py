import textwrap
import unittest
from pathlib import Path

import numpy as np

from quattrocento.channels import load_channels
from quattrocento.config import QuattrocentoConfig
from quattrocento.processing import detect_onset


class QuattrocentoConfigTests(unittest.TestCase):
    def test_default_config_initialises(self) -> None:
        config = QuattrocentoConfig()
        self.assertEqual(config.sample_rate_hz, 512)
        self.assertGreater(config.total_window_samples, 0)

    def test_positive_offset_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QuattrocentoConfig(window_offset_seconds=0.1)

    def test_offset_spanning_full_window_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QuattrocentoConfig(
                sample_rate_hz=8,
                window_seconds=1.0,
                window_offset_seconds=-1.0,
            )


class LoadChannelsTests(unittest.TestCase):
    def _write_toml(self, content: str, tmp_path: Path) -> Path:
        p = tmp_path / "channels.toml"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._tmp_path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_basic_parse(self) -> None:
        p = self._write_toml(
            """
            [labels]
            "L Thumb" = { index = 0, scale = 5.0 }
            "R Index" = { index = 6, scale = 5.0 }
            "trigger" = { index = 10, scale = 5.0, kind = "trigger" }
            """,
            self._tmp_path,
        )
        channels = load_channels(p)
        self.assertEqual(channels[0].label, "L Thumb")
        self.assertEqual(channels[6].label, "R Index")
        self.assertEqual(channels[10].label, "trigger")
        self.assertEqual(channels[6].scale, 5.0)
        self.assertEqual(channels[6].kind, "finger")
        self.assertEqual(channels[10].kind, "trigger")

    def test_duplicate_index_raises(self) -> None:
        p = self._write_toml(
            """
            [labels]
            "A" = { index = 0 }
            "B" = { index = 0 }
            """,
            self._tmp_path,
        )
        with self.assertRaises(ValueError):
            load_channels(p)

    def test_negative_label_index_raises(self) -> None:
        p = self._write_toml(
            """
            [labels]
            "A" = { index = -1 }
            """,
            self._tmp_path,
        )
        with self.assertRaises(ValueError):
            load_channels(p)

    def test_inverts_label_to_index_dict(self) -> None:
        """The TOML [labels] table is label→config; returned dict is index→label."""
        p = self._write_toml(
            """
            [labels]
            "MyChannel" = { index = 5 }
            "trigger"   = { index = 10, kind = "trigger" }
            """,
            self._tmp_path,
        )
        channels = load_channels(p)
        self.assertIn(5, channels)
        self.assertEqual(channels[5].label, "MyChannel")

    def test_emg_kind_parsed(self) -> None:
        p = self._write_toml(
            """
            [labels]
            "R Index" = { index = 0, scale = 5.0 }
            "EMG 1"   = { index = 11, scale = 1.0, kind = "emg" }
            "EMG 2"   = { index = 12, scale = 1.0, kind = "emg" }
            "trigger" = { index = 10, kind = "trigger" }
            """,
            self._tmp_path,
        )
        channels = load_channels(p)
        self.assertEqual(channels[0].kind, "finger")
        self.assertEqual(channels[11].kind, "emg")
        self.assertEqual(channels[12].kind, "emg")
        self.assertEqual(channels[11].scale, 1.0)

    def test_unknown_kind_rejected(self) -> None:
        p = self._write_toml(
            """
            [labels]
            "X" = { index = 0, kind = "nonsense" }
            """,
            self._tmp_path,
        )
        with self.assertRaises(ValueError):
            load_channels(p)

    def test_missing_trigger_rejected(self) -> None:
        """A channels file with no trigger kind is rejected."""
        p = self._write_toml(
            """
            [labels]
            "F0" = { index = 0 }
            "F1" = { index = 1 }
            """,
            self._tmp_path,
        )
        with self.assertRaises(ValueError):
            load_channels(p)

    def test_multiple_triggers_rejected(self) -> None:
        """A channels file with more than one trigger kind is rejected."""
        p = self._write_toml(
            """
            [labels]
            "T0" = { index = 0, kind = "trigger" }
            "T1" = { index = 1, kind = "trigger" }
            """,
            self._tmp_path,
        )
        with self.assertRaises(ValueError):
            load_channels(p)


class DetectOnsetTests(unittest.TestCase):
    # Alternating ±1 baseline: mean=0, sd=1, upper=5, lower=-5.
    _PRE = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(50)])
    _TRIGGER_IDX = 50
    _RATE = 1000  # 1 sample = 1 ms

    def _signal(self, quiet: int, step_val: float, step_len: int) -> np.ndarray:
        """baseline + quiet post-trigger samples + step."""
        return np.concatenate([
            self._PRE,
            np.zeros(quiet),
            np.full(step_len, step_val),
        ])

    def test_onset_detected_at_expected_ms(self) -> None:
        # Step starts 5 samples after trigger; onset confirmed after 5 consecutive.
        signal = self._signal(quiet=5, step_val=10.0, step_len=10)
        result = detect_onset(signal, self._TRIGGER_IDX, self._RATE)
        self.assertAlmostEqual(result, 5.0)

    def test_no_crossing_returns_none(self) -> None:
        signal = np.concatenate([self._PRE, np.zeros(20)])
        self.assertIsNone(detect_onset(signal, self._TRIGGER_IDX, self._RATE))

    def test_too_few_pre_samples_returns_none(self) -> None:
        signal = np.concatenate([np.zeros(5), np.full(20, 10.0)])
        self.assertIsNone(detect_onset(signal, trigger_idx=5, sample_rate_hz=self._RATE))

    def test_flat_pre_signal_returns_none(self) -> None:
        signal = np.concatenate([np.zeros(50), np.full(10, 10.0)])
        self.assertIsNone(detect_onset(signal, self._TRIGGER_IDX, self._RATE))

    def test_post_skip_preserves_trigger_relative_timing(self) -> None:
        # Skip 3 of the 5 quiet samples; onset is still at sample 5 from trigger.
        signal = self._signal(quiet=5, step_val=10.0, step_len=10)
        result = detect_onset(signal, self._TRIGGER_IDX, self._RATE, post_skip_samples=3)
        self.assertAlmostEqual(result, 5.0)


if __name__ == "__main__":
    unittest.main()
