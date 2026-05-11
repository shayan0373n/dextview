import unittest

import numpy as np

from quattrocento.config import QuattrocentoConfig
from quattrocento.stream.direct import DirectStream
from quattrocento.stream.parser import FrameParser


class TestFrameParser(unittest.TestCase):
    def setUp(self):
        self.n_channels = 408
        self.config = QuattrocentoConfig(sample_rate_hz=512, n_channels=self.n_channels)
        self.parser = FrameParser(self.config)

    def test_buffer_capping_preserves_frame_alignment(self):
        frame_bytes = 2 * self.n_channels
        max_buffer_size = 50 * 1024 * 1024

        prefill_samples = (45 * 1024 * 1024) // frame_bytes
        self.parser.feed(b"p" * (prefill_samples * frame_bytes))

        payload_samples = (10 * 1024 * 1024) // frame_bytes
        payload = b"x" * (payload_samples * frame_bytes)
        self.parser.feed(payload)

        total = prefill_samples * frame_bytes + len(payload)
        excess = total - max_buffer_size
        expected_dropped_samples = (excess + frame_bytes - 1) // frame_bytes

        self.assertLessEqual(len(self.parser._byte_buffer), max_buffer_size)
        self.assertEqual(len(self.parser._byte_buffer) % frame_bytes, 0)
        self.assertEqual(self.parser._sample_index, expected_dropped_samples)

    def test_drain_returns_correct_shape(self):
        frame_bytes = 2 * self.n_channels
        self.parser.feed(b"\x00" * (frame_bytes * 3))
        batch = self.parser.drain()
        self.assertEqual(batch.signals.shape, (3, self.n_channels))
        self.assertEqual(batch.timestamps.shape, (3,))

    def test_drain_empty_buffer_returns_empty_batch(self):
        batch = self.parser.drain()
        self.assertEqual(batch.signals.shape, (0, self.n_channels))
        self.assertEqual(batch.timestamps.shape, (0,))


class TestDirectStream(unittest.TestCase):
    def setUp(self):
        self.n_channels = 408
        self.config = QuattrocentoConfig(sample_rate_hz=512, n_channels=self.n_channels)
        self.stream = DirectStream(
            self.config,
            host="169.254.1.10",
            port=23456,
            nch=3,
        )


if __name__ == "__main__":
    unittest.main()
