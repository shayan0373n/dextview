import unittest
from pathlib import Path

from dextview.protocol import DEFAULT_CONF2_BYTE, INPUT_BLOCK_NAMES
from dextview.settings import load_input_conf2_bytes


class LoadInputConf2BytesTests(unittest.TestCase):
    def test_loads_defaults_and_overrides(self) -> None:
        config_path = Path("tests") / "_tmp_conf2_valid.toml"
        try:
            config_path.write_text(
                (
                    "[conf2_defaults]\n"
                    'side = "left"\n'
                    "hpf = 100\n"
                    "lpf = 900\n"
                    'mode = "bipolar"\n'
                    "[conf2_overrides]\n"
                    'IN1 = { mode = "differential" }\n'
                ),
                encoding="utf-8",
            )
            conf2_bytes = load_input_conf2_bytes(config_path)
            self.assertEqual(len(conf2_bytes), len(INPUT_BLOCK_NAMES))
            self.assertEqual(conf2_bytes[0], 0b01101001)  # IN1 mode override
            self.assertEqual(conf2_bytes[1], 0b01101010)  # IN2 default
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_empty_file_returns_defaults(self) -> None:
        config_path = Path("tests") / "_tmp_conf2_empty.toml"
        try:
            config_path.write_text("", encoding="utf-8")
            conf2_bytes = load_input_conf2_bytes(config_path)
            self.assertEqual(
                conf2_bytes,
                tuple(DEFAULT_CONF2_BYTE for _ in range(len(INPUT_BLOCK_NAMES))),
            )
        finally:
            if config_path.exists():
                config_path.unlink()

    def test_rejects_unknown_top_level_field(self) -> None:
        config_path = Path("tests") / "_tmp_conf2_unknown.toml"
        try:
            config_path.write_text("host = \"x\"\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_input_conf2_bytes(config_path)
        finally:
            if config_path.exists():
                config_path.unlink()


if __name__ == "__main__":
    unittest.main()
