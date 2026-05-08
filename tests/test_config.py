import textwrap
import unittest
from pathlib import Path

from quattrocento.channels import load_channels
from quattrocento.config import QuattrocentoConfig


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
            "L Thumb" = 0
            "R Index" = 6
            "trigger" = 10
            """,
            self._tmp_path,
        )
        labels = load_channels(p)
        self.assertEqual(labels[0], "L Thumb")
        self.assertEqual(labels[6], "R Index")
        self.assertEqual(labels[10], "trigger")

    def test_duplicate_index_raises(self) -> None:
        p = self._write_toml(
            """
            [labels]
            "A" = 0
            "B" = 0
            """,
            self._tmp_path,
        )
        with self.assertRaises(ValueError):
            load_channels(p)

    def test_negative_label_index_raises(self) -> None:
        p = self._write_toml(
            """
            [labels]
            "A" = -1
            """,
            self._tmp_path,
        )
        with self.assertRaises(ValueError):
            load_channels(p)

    def test_inverts_label_to_index_dict(self) -> None:
        """The TOML [labels] table is label→index; returned dict is index→label."""
        p = self._write_toml(
            """
            [labels]
            "MyChannel" = 5
            """,
            self._tmp_path,
        )
        labels = load_channels(p)
        self.assertIn(5, labels)
        self.assertEqual(labels[5], "MyChannel")


if __name__ == "__main__":
    unittest.main()
