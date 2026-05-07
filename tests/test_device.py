import unittest
from unittest.mock import Mock, patch

from quattrocento.config import QuattrocentoConfig
from quattrocento.device import QuattrocentoStream


class TestQuattrocentoStream(unittest.TestCase):
    def setUp(self):
        self.config = QuattrocentoConfig(sample_rate_hz=512)
        self.stream = QuattrocentoStream(
            self.config,
            handshake_kind="real",
            host="169.254.1.10",
            port=23456,
            nch=3,
            force_channel_indices=tuple(range(10)),
            aux_in_channel_index=10,
            n_channels=408,  # NCH 3 is 408 channels
        )

    @patch("socket.socket")
    def test_drain_socket_buffer_capping_preserves_frame_alignment(self, mock_socket_cls):
        mock_sock = Mock()
        mock_socket_cls.return_value = mock_sock

        self.stream._ensure_connected()

        frame_bytes = self.stream._frame_bytes
        max_buffer_size = 50 * 1024 * 1024

        prefill_samples = (45 * 1024 * 1024) // frame_bytes
        initial_buffer_size = prefill_samples * frame_bytes
        self.stream._byte_buffer.extend(b"p" * initial_buffer_size)

        payload_samples = (10 * 1024 * 1024) // frame_bytes
        payload = b"x" * (payload_samples * frame_bytes)
        payload_offset = 0

        def mock_recv(size):
            nonlocal payload_offset
            if payload_offset >= len(payload):
                raise BlockingIOError

            chunk = payload[payload_offset : payload_offset + size]
            payload_offset += len(chunk)
            return chunk

        mock_sock.recv.side_effect = mock_recv

        total_size_before_cap = initial_buffer_size + len(payload)
        excess_bytes = total_size_before_cap - max_buffer_size
        expected_dropped_samples = (excess_bytes + frame_bytes - 1) // frame_bytes

        self.stream._drain_socket()

        self.assertLessEqual(len(self.stream._byte_buffer), max_buffer_size)
        self.assertEqual(len(self.stream._byte_buffer) % frame_bytes, 0)
        self.assertEqual(self.stream._sample_index, expected_dropped_samples)


if __name__ == "__main__":
    unittest.main()
