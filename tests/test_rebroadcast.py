import unittest
from unittest.mock import Mock, patch

import numpy as np

from quattrocento.config import QuattrocentoConfig
from quattrocento.device import QuattrocentoStream


class RebroadcastStreamConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = QuattrocentoConfig(sample_rate_hz=2048)

    def test_rejects_force_channels_count_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            QuattrocentoStream(
                config=self.config,
                handshake_kind="rebroadcast",
                host="127.0.0.1",
                port=31000,
                n_channels=64,
                force_channel_indices=(0, 1, 2),
                aux_in_channel_index=10,
            )

    def test_rejects_out_of_range_force_channel(self) -> None:
        with self.assertRaises(ValueError):
            QuattrocentoStream(
                config=self.config,
                handshake_kind="rebroadcast",
                host="127.0.0.1",
                port=31000,
                n_channels=10,
                force_channel_indices=tuple(range(10)),
                aux_in_channel_index=10,
            )

    def test_rejects_duplicate_force_channels(self) -> None:
        with self.assertRaises(ValueError):
            QuattrocentoStream(
                config=self.config,
                handshake_kind="rebroadcast",
                host="127.0.0.1",
                port=31000,
                n_channels=64,
                force_channel_indices=(0, 0, 1, 2, 3, 4, 5, 6, 7, 8),
                aux_in_channel_index=10,
            )


class RebroadcastStreamReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = QuattrocentoConfig(sample_rate_hz=2048)
        self.n_channels = 64
        self.stream = QuattrocentoStream(
            config=self.config,
            handshake_kind="rebroadcast",
            host="127.0.0.1",
            port=31000,
            n_channels=self.n_channels,
            force_channel_indices=tuple(range(10)),
            aux_in_channel_index=10,
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

        self.assertEqual(batch.forces.shape, (sample_count, 10))
        np.testing.assert_array_equal(batch.forces[:, 0], -100)
        np.testing.assert_array_equal(
            batch.aux_in,
            frame_values[:, 10].astype(np.float64),
        )


if __name__ == "__main__":
    unittest.main()
