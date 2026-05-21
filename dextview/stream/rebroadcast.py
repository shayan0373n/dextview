import socket

from ..config import DextViewConfig
from ..models import DataBatch
from ._io import drain_socket
from .parser import FrameParser

_HEADER_BYTES = 8


class RebroadcastStream:
    """TCP stream connected to a rebroadcast server (startTX/stopTX protocol)."""

    def __init__(
        self,
        config: DextViewConfig,
        *,
        host: str,
        port: int,
    ) -> None:
        self._config = config
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None
        self._parser = FrameParser(config)

    @property
    def config(self) -> DextViewConfig:
        return self._config

    def read_batch(self) -> DataBatch:
        """Reads and parses the latest chunk of data from the rebroadcast server."""
        sock = self._ensure_connected()
        try:
            raw = drain_socket(sock)
        except ConnectionError:
            self._close_socket()
            raise
        self._parser.feed(raw)
        return self._parser.drain()

    def close(self) -> None:
        """Sends stopTX and closes the TCP connection."""
        if self._socket is None:
            return
        try:
            self._socket.setblocking(True)
            self._socket.sendall(b"stopTX")
        except OSError:
            pass
        self._close_socket()

    def _ensure_connected(self) -> socket.socket:
        """Establishes the connection (if needed) and returns the live socket."""
        if self._socket is not None:
            return self._socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.settimeout(3.0)
            sock.connect((self._host, self._port))
            sock.sendall(b"startTX")
            header = bytearray()
            while len(header) < _HEADER_BYTES:
                chunk = sock.recv(_HEADER_BYTES - len(header))
                if not chunk:
                    raise ConnectionError("Connection closed before header arrived")
                header.extend(chunk)
            sock.setblocking(False)
        except BaseException:
            sock.close()
            raise
        self._socket = sock
        return sock

    def _close_socket(self) -> None:
        """Closes the socket and clears the reference."""
        if self._socket is None:
            return
        try:
            self._socket.close()
        except OSError:
            pass
        finally:
            self._socket = None
