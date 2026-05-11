import socket
import threading
import unittest

import numpy as np

from quattrocento.config import QuattrocentoConfig
from quattrocento.protocol import (
    NCH_BITS_TO_NUM_CHANNELS,
    StartCommand,
    build_start_command,
    build_stop_command,
    parse_start_command,
)
from quattrocento.stream.proxy import ProxyStream


# ---------------------------------------------------------------------------
# parse_start_command round-trips
# ---------------------------------------------------------------------------

class TestParseStartCommand(unittest.TestCase):
    def _round_trip(self, fsamp, nch_code, decimation, rec_on):
        n_channels = NCH_BITS_TO_NUM_CHANNELS[nch_code]
        config = QuattrocentoConfig(sample_rate_hz=fsamp, n_channels=n_channels)
        encoded = build_start_command(
            decimation_enabled=decimation,
            rec_on=rec_on,
            fsamp=fsamp,
            nch=nch_code,
            input_conf2_bytes=tuple([0b00010100] * 12),
        )
        result = parse_start_command(encoded)
        self.assertEqual(result.fsamp_hz, fsamp)
        self.assertEqual(result.nch_code, nch_code)
        self.assertEqual(result.decimation_enabled, decimation)
        self.assertEqual(result.rec_on, rec_on)
        self.assertEqual(result.input_conf2_bytes, tuple([0b00010100] * 12))

    def test_round_trip_512hz_nch0(self):
        self._round_trip(512, 0, True, False)

    def test_round_trip_2048hz_nch1(self):
        self._round_trip(2048, 1, False, False)

    def test_round_trip_5120hz_nch2_rec_on(self):
        self._round_trip(5120, 2, True, True)

    def test_round_trip_10240hz_nch3(self):
        self._round_trip(10240, 3, False, True)

    def test_bad_crc_raises(self):
        encoded = bytearray(build_start_command(
            decimation_enabled=True,
            rec_on=False,
            fsamp=512,
            nch=0,
            input_conf2_bytes=tuple([0] * 12),
        ))
        encoded[-1] ^= 0xFF  # corrupt CRC
        with self.assertRaises(ValueError, msg="CRC mismatch"):
            parse_start_command(bytes(encoded))

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            parse_start_command(b"\x00" * 39)
        with self.assertRaises(ValueError):
            parse_start_command(b"\x00" * 41)


# ---------------------------------------------------------------------------
# ProxyStream end-to-end socket test
# ---------------------------------------------------------------------------

def _make_connected_pair() -> tuple[socket.socket, socket.socket]:
    """Return a loopback (client, server) socket pair."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    server_side, _ = srv.accept()
    srv.close()
    return client, server_side


class TestProxyStream(unittest.TestCase):
    def setUp(self):
        self.n_channels = 120
        self.fsamp = 512
        self.config = QuattrocentoConfig(
            sample_rate_hz=self.fsamp,
            n_channels=self.n_channels,
        )

    def _make_proxy(self) -> tuple[ProxyStream, socket.socket, socket.socket]:
        """
        Build a ProxyStream with loopback sockets:
          upstream_ctrl  — the test pretends to be the upstream controller
          downstream_dev — the test pretends to be the device

        Returns (proxy, upstream_ctrl, downstream_dev).
        """
        upstream_ctrl, client_sock = _make_connected_pair()
        origin_sock, downstream_dev = _make_connected_pair()

        client_sock.setblocking(False)
        origin_sock.setblocking(False)
        upstream_ctrl.setblocking(True)
        downstream_dev.setblocking(True)

        proxy = ProxyStream(self.config, client_sock=client_sock, origin_sock=origin_sock)
        return proxy, upstream_ctrl, downstream_dev

    def test_data_from_device_appears_in_read_batch(self):
        proxy, upstream_ctrl, downstream_dev = self._make_proxy()
        try:
            sample_count = 10
            frames = np.arange(
                sample_count * self.n_channels, dtype=np.int16
            ).reshape(sample_count, self.n_channels)
            raw = frames.tobytes()
            downstream_dev.sendall(raw)

            # May need a couple of ticks for all bytes to arrive.
            batches = []
            for _ in range(5):
                batch = proxy.read_batch()
                if batch.signals.shape[0]:
                    batches.append(batch)
                if sum(b.signals.shape[0] for b in batches) >= sample_count:
                    break

            total_samples = sum(b.signals.shape[0] for b in batches)
            self.assertEqual(total_samples, sample_count)
        finally:
            proxy.close()
            upstream_ctrl.close()
            downstream_dev.close()

    def test_device_bytes_mirrored_to_upstream_ctrl(self):
        proxy, upstream_ctrl, downstream_dev = self._make_proxy()
        try:
            raw = b"\xAB\xCD" * (self.n_channels * 4)  # 4 complete frames

            downstream_dev.sendall(raw)
            for _ in range(5):
                proxy.read_batch()

            upstream_ctrl.settimeout(1.0)
            received = bytearray()
            try:
                while len(received) < len(raw):
                    chunk = upstream_ctrl.recv(len(raw) - len(received))
                    if not chunk:
                        break
                    received.extend(chunk)
            except socket.timeout:
                pass

            self.assertEqual(bytes(received), raw)
        finally:
            proxy.close()
            upstream_ctrl.close()
            downstream_dev.close()

    def test_ctrl_bytes_forwarded_to_device(self):
        proxy, upstream_ctrl, downstream_dev = self._make_proxy()
        try:
            stop_cmd = build_stop_command()
            upstream_ctrl.sendall(stop_cmd)

            for _ in range(5):
                proxy.read_batch()

            downstream_dev.settimeout(1.0)
            received = bytearray()
            try:
                while len(received) < len(stop_cmd):
                    chunk = downstream_dev.recv(len(stop_cmd) - len(received))
                    if not chunk:
                        break
                    received.extend(chunk)
            except socket.timeout:
                pass

            self.assertEqual(bytes(received), stop_cmd)
        finally:
            proxy.close()
            upstream_ctrl.close()
            downstream_dev.close()

    def test_device_disconnect_raises_connection_error(self):
        proxy, upstream_ctrl, downstream_dev = self._make_proxy()
        try:
            downstream_dev.close()
            with self.assertRaises(ConnectionError):
                for _ in range(10):
                    proxy.read_batch()
        finally:
            proxy.close()
            upstream_ctrl.close()


if __name__ == "__main__":
    unittest.main()
