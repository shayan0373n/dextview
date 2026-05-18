from __future__ import annotations

import logging
import socket

from ..config import QuattrocentoConfig
from ..models import DataBatch
from ..protocol import (
    COMMAND_LENGTH,
    NCH_BITS_TO_NUM_CHANNELS,
    build_stop_command,
    parse_start_command,
)
from ._io import drain_socket, read_exact
from .parser import FrameParser

_logger = logging.getLogger(__name__)


class ProxyStream:
    """Bidirectional proxy between an upstream controller and the device.

    Accepts an already-connected pair of sockets (both non-blocking):
      client_sock  — accepted from a listener; the upstream controller writes
                     commands here and reads data from here.
      origin_sock  — connected to the actual device; commands are forwarded
                     to it and data flows back from it.

    On each read_batch() tick:
      1. Any bytes pending on client_sock are forwarded to origin_sock.
      2. Bytes from origin_sock are fed to the local parser and mirrored to
         client_sock so the upstream controller also receives them.
    """

    def __init__(
        self,
        config: QuattrocentoConfig,
        *,
        client_sock: socket.socket,
        origin_sock: socket.socket,
    ) -> None:
        self._config = config
        self._client_sock: socket.socket | None = client_sock
        self._origin_sock: socket.socket | None = origin_sock
        self._parser = FrameParser(config)

    @classmethod
    def listen_and_accept(
        cls,
        *,
        listen_host: str,
        listen_port: int,
        device_host: str,
        device_port: int,
        window_seconds: float,
        window_offset_seconds: float,
        trigger_threshold: float,
        trigger_channel: int,
    ) -> "ProxyStream":
        """Listen for an upstream client, sniff its start command, and connect to the device.

        Raises ConnectionError if any socket step fails, ValueError if the start
        command is malformed.
        """
        _logger.info("Proxy listening on %s:%d ...", listen_host, listen_port)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((listen_host, listen_port))
        listener.listen(1)
        try:
            client_sock, client_addr = listener.accept()
        finally:
            listener.close()

        _logger.info("Client connected from %s", client_addr)
        client_sock.settimeout(5.0)
        try:
            init_bytes = read_exact(client_sock, COMMAND_LENGTH)
            start_cmd = parse_start_command(init_bytes)
        except (ConnectionError, OSError, ValueError) as exc:
            client_sock.close()
            raise ConnectionError(f"Failed to read start command from client: {exc}") from exc

        n_channels = NCH_BITS_TO_NUM_CHANNELS[start_cmd.nch_code]
        _logger.info(
            "Sniffed: %d Hz, %d channels. Connecting to device at %s:%d ...",
            start_cmd.fsamp_hz, n_channels, device_host, device_port,
        )

        origin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        origin_sock.settimeout(3.0)
        try:
            origin_sock.connect((device_host, device_port))
        except OSError as exc:
            client_sock.close()
            raise ConnectionError(
                f"Failed to connect to device at {device_host}:{device_port}: {exc}"
            ) from exc

        client_sock.setblocking(False)
        origin_sock.setblocking(False)

        config = QuattrocentoConfig(
            sample_rate_hz=start_cmd.fsamp_hz,
            n_channels=n_channels,
            window_seconds=window_seconds,
            window_offset_seconds=window_offset_seconds,
            trigger_threshold=trigger_threshold,
            trigger_channel=trigger_channel,
        )
        return cls(config, client_sock=client_sock, origin_sock=origin_sock)

    @property
    def config(self) -> QuattrocentoConfig:
        return self._config

    def read_batch(self) -> DataBatch:
        """Forwards control bytes and mirrors data between client and origin."""
        # Forward any control bytes from the client to the origin.
        if self._client_sock is not None:
            try:
                ctrl = drain_socket(self._client_sock)
                if ctrl:
                    self._origin_sock.sendall(ctrl)
            except ConnectionError:
                self._close_sockets()
                raise

        # Drain the origin; process locally and mirror to the client.
        try:
            raw = drain_socket(self._origin_sock)
        except ConnectionError:
            self._close_sockets()
            raise

        if raw:
            self._parser.feed(raw)
            if self._client_sock is not None:
                try:
                    # OSError (incl. BlockingIOError) here means the client's
                    # send buffer is full or the client closed — fatal without
                    # a write-side buffer, so treat as disconnect.
                    self._client_sock.sendall(raw)
                except OSError:
                    self._close_sockets()
                    raise ConnectionError("Client disconnected during forwarding")

        return self._parser.drain()

    def close(self) -> None:
        """Stops upstream/downstream flow and closes all sockets."""
        if self._origin_sock is not None:
            try:
                self._origin_sock.setblocking(True)
                # The upstream may have already forwarded a stop; sending another
                # is harmless — the device ignores duplicate stops.
                self._origin_sock.sendall(build_stop_command())
            except OSError:
                pass
            self._origin_sock.close()
            self._origin_sock = None

        if self._client_sock is not None:
            self._client_sock.close()
            self._client_sock = None

    def _close_sockets(self) -> None:
        """Closes both client and origin sockets and clears their references."""
        for attr in ("_origin_sock", "_client_sock"):
            sock: socket.socket | None = getattr(self, attr)
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                setattr(self, attr, None)
