from __future__ import annotations

import socket

from ..config import QuattrocentoConfig
from ..models import DataBatch
from ..protocol import (
    DEFAULT_INPUT_CONF2_BYTES,
    NCH_BITS_TO_NUM_CHANNELS,
    SUPPORTED_SAMPLE_RATES,
    build_start_command,
    build_stop_command,
)
from ._io import drain_socket
from .parser import FrameParser


class DirectStream:
    """TCP stream connected directly to a Quattrocento device."""

    def __init__(
        self,
        config: QuattrocentoConfig,
        *,
        host: str,
        port: int,
        nch: int,
        decimation_enabled: bool = True,
        rec_on: bool = False,
        input_conf2_bytes: tuple[int, ...] = DEFAULT_INPUT_CONF2_BYTES,
    ) -> None:
        if config.sample_rate_hz not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate_hz must be one of {SUPPORTED_SAMPLE_RATES}, "
                f"got {config.sample_rate_hz}"
            )
        if nch not in NCH_BITS_TO_NUM_CHANNELS:
            raise ValueError("nch must be one of 0, 1, 2, 3")

        self._config = config
        self._host = host
        self._port = port
        self._nch = nch
        self._decimation_enabled = decimation_enabled
        self._rec_on = rec_on
        self._input_conf2_bytes = input_conf2_bytes
        self._socket: socket.socket | None = None
        self._parser = FrameParser(config)

    @property
    def config(self) -> QuattrocentoConfig:
        return self._config

    def read_batch(self) -> DataBatch:
        self._ensure_connected()
        try:
            raw = drain_socket(self._socket)
        except ConnectionError:
            self._close_socket()
            raise
        self._parser.feed(raw)
        return self._parser.drain()

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.setblocking(True)
            self._socket.sendall(build_stop_command())
        except OSError:
            pass
        self._close_socket()

    def _ensure_connected(self) -> None:
        if self._socket is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(3.0)
            sock.connect((self._host, self._port))
            sock.sendall(
                build_start_command(
                    decimation_enabled=self._decimation_enabled,
                    rec_on=self._rec_on,
                    fsamp=self._config.sample_rate_hz,
                    nch=self._nch,
                    input_conf2_bytes=self._input_conf2_bytes,
                )
            )
            sock.setblocking(False)
        except BaseException:
            sock.close()
            raise
        self._socket = sock

    def _close_socket(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        except OSError:
            pass
        finally:
            self._socket = None
