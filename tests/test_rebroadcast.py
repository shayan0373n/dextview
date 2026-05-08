import unittest
from unittest.mock import Mock, patch

import numpy as np

from quattrocento.config import QuattrocentoConfig
from quattrocento.device import QuattrocentoStream


class RebroadcastStreamConstructionTests(unittest.TestCase):
    def test_rejects_nonpositive_n_channels(self) -> None:
        with self.assertRaises(ValueError):
            QuattrocentoConfig(sample_rate_hz=2048, n_channels=0)


class RebroadcastStreamReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.n_channels = 64
        self.config = QuattrocentoConfig(sample_rate_hz=2048, n_channels=self.n_channels)
        self.stream = QuattrocentoStream(
            config=self.config,
            handshake_kind="rebroadcast",
            host="127.0.0.1",
            port=31000,
        )

    @patch("socket.socket")
    def test_read_batch_parses_signed_int16_frames(self, mock_socket_cls):
        mock_sock = Mock()
        mock_socket_cls.return_value = mock_sock

        sample_count = 5
        frame_values = np.arange(
            sample_count * self.n_channels, dtype=np.int16
        ).reshape(sample_count, self.n_channels)
        frame_values[:, 0] = -100  # Negative value verifies signed parsing
        payload = frame_values.tobytes()

        chunks = [b"HEADER01", payload]

        def mock_recv(size):
            if not chunks:
                raise BlockingIOError
            chunk = chunks.pop(0)
            if len(chunk) > size:
                chunks.insert(0, chunk[size:])
                return chunk[:size]
            return chunk

        mock_sock.recv.side_effect = mock_recv

        batch = self.stream.read_batch()

        self.assertEqual(batch.signals.shape, (sample_count, self.n_channels))
        np.testing.assert_array_equal(batch.signals[:, 0], -100.0)
        np.testing.assert_array_equal(
            batch.signals[:, 10],
            frame_values[:, 10].astype(np.float64),
        )


if __name__ == "__main__":
    unittest.main()
