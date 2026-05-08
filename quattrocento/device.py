from __future__ import annotations

import socket
from typing import Literal

import numpy as np

from .config import QuattrocentoConfig
from .models import DataBatch
from .protocol import (
    DEFAULT_INPUT_CONF2_BYTES,
    NCH_BITS_TO_NUM_CHANNELS,
    SUPPORTED_SAMPLE_RATES,
    build_start_command,
    build_stop_command,
)

HandshakeKind = Literal["real", "rebroadcast"]


class QuattrocentoStream:
    """TCP batch stream for a Quattrocento device or rebroadcast server."""

    _SOCKET_READ_SIZE = 65536
    _MAX_READ_BYTES_PER_TICK = 10 * 1024 * 1024
    _MAX_BUFFER_BYTES = 50 * 1024 * 1024
    _REBROADCAST_HEADER_BYTES = 8

    def __init__(
        self,
        config: QuattrocentoConfig,
        *,
        handshake_kind: HandshakeKind,
        host: str,
        port: int,
        # Real-mode specific settings:
        nch: int | None = None,
        decimation_enabled: bool = True,
        rec_on: bool = False,
        input_conf2_bytes: tuple[int, ...] = DEFAULT_INPUT_CONF2_BYTES,
    ) -> None:
        if handshake_kind == "real":
            if config.sample_rate_hz not in SUPPORTED_SAMPLE_RATES:
                raise ValueError(
                    f"sample_rate_hz must be one of {SUPPORTED_SAMPLE_RATES}, "
                    f"got {config.sample_rate_hz}"
                )
            if nch not in NCH_BITS_TO_NUM_CHANNELS:
                raise ValueError("nch must be one of 0, 1, 2, 3")

        self._config = config
        self._handshake_kind = handshake_kind
        self._host = host
        self._port = port
        self._frame_bytes = 2 * config.n_channels
        self._sample_index = 0

        # Handshake-specific state
        self._nch = nch
        self._decimation_enabled = decimation_enabled
        self._rec_on = rec_on
        self._input_conf2_bytes = input_conf2_bytes

        self._socket: socket.socket | None = None
        self._byte_buffer = bytearray()

    @property
    def config(self) -> QuattrocentoConfig:
        return self._config

    @property
    def n_channels(self) -> int:
        return self._config.n_channels

    def read_batch(self) -> DataBatch:
        """Read all complete samples currently available from the TCP stream."""
        self._ensure_connected()
        self._drain_socket()

        sample_count = len(self._byte_buffer) // self._frame_bytes
        if sample_count == 0:
            return self._empty_batch()

        bytes_to_parse = sample_count * self._frame_bytes
        raw = bytes(self._byte_buffer[:bytes_to_parse])
        del self._byte_buffer[:bytes_to_parse]

        signals = np.frombuffer(raw, dtype="<i2").reshape(
            sample_count, self._config.n_channels
        ).astype(np.float64)

        sample_indices = np.arange(
            self._sample_index, self._sample_index + sample_count, dtype=np.int64
        )
        timestamps = sample_indices.astype(np.float64) / self._config.sample_rate_hz
        self._sample_index += sample_count

        return DataBatch(timestamps=timestamps, signals=signals)

    def close(self) -> None:
        """Stop acquisition and close the TCP socket."""
        if self._socket is None:
            return

        try:
            self._socket.setblocking(True)
            self._stop_acquisition(self._socket)
        except OSError:
            pass
        finally:
            try:
                self._socket.close()
            finally:
                self._socket = None
                self._byte_buffer.clear()

    def _start_acquisition(self, sock: socket.socket) -> None:
        if self._handshake_kind == "real":
            assert self._nch is not None
            sock.sendall(
                build_start_command(
                    decimation_enabled=self._decimation_enabled,
                    rec_on=self._rec_on,
                    fsamp=self._config.sample_rate_hz,
                    nch=self._nch,
                    input_conf2_bytes=self._input_conf2_bytes,
                )
            )
        else:
            sock.sendall(b"startTX")
            header = bytearray()
            while len(header) < self._REBROADCAST_HEADER_BYTES:
                chunk = sock.recv(self._REBROADCAST_HEADER_BYTES - len(header))
                if not chunk:
                    raise ConnectionError("Rebroadcast socket closed before header arrived")
                header.extend(chunk)

    def _stop_acquisition(self, sock: socket.socket) -> None:
        if self._handshake_kind == "real":
            sock.sendall(build_stop_command())
        else:
            sock.sendall(b"stopTX")

    def _ensure_connected(self) -> None:
        if self._socket is not None:
            return

        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            tcp_socket.settimeout(3.0)
            tcp_socket.connect((self._host, self._port))
            self._start_acquisition(tcp_socket)
            tcp_socket.setblocking(False)
        except BaseException:
            tcp_socket.close()
            raise
        self._socket = tcp_socket
        self._byte_buffer.clear()

    def _drain_socket(self) -> None:
        if self._socket is None:
            return

        bytes_read = 0
        while bytes_read < self._MAX_READ_BYTES_PER_TICK:
            try:
                read_size = min(
                    self._SOCKET_READ_SIZE,
                    self._MAX_READ_BYTES_PER_TICK - bytes_read,
                )
                chunk = self._socket.recv(read_size)
            except BlockingIOError:
                break
            except InterruptedError:
                continue

            if not chunk:
                self._socket.close()
                self._socket = None
                self._byte_buffer.clear()
                raise ConnectionError("Stream socket closed the connection")

            self._byte_buffer.extend(chunk)
            bytes_read += len(chunk)

            if len(chunk) < read_size:
                break

        if len(self._byte_buffer) > self._MAX_BUFFER_BYTES:
            excess_bytes = len(self._byte_buffer) - self._MAX_BUFFER_BYTES
            samples_to_drop = (excess_bytes + self._frame_bytes - 1) // self._frame_bytes
            bytes_to_drop = samples_to_drop * self._frame_bytes
            del self._byte_buffer[:bytes_to_drop]
            self._sample_index += samples_to_drop

    def _empty_batch(self) -> DataBatch:
        return DataBatch(
            timestamps=np.empty(0, dtype=np.float64),
            signals=np.empty((0, self._config.n_channels), dtype=np.float64),
        )
