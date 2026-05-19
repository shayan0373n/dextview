import socket

_SOCKET_READ_SIZE = 65536
_MAX_READ_BYTES_PER_TICK = 10 * 1024 * 1024


def read_exact(sock: socket.socket, n: int) -> bytes:
    """Blocking read of exactly n bytes. Raises ConnectionError on premature close."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"Connection closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def drain_socket(sock: socket.socket) -> bytes:
    """Non-blocking drain of one socket tick.

    Returns all bytes available up to _MAX_READ_BYTES_PER_TICK.
    Raises ConnectionError if the remote side closed the connection.
    """
    buf = bytearray()
    bytes_read = 0
    while bytes_read < _MAX_READ_BYTES_PER_TICK:
        read_size = min(_SOCKET_READ_SIZE, _MAX_READ_BYTES_PER_TICK - bytes_read)
        try:
            chunk = sock.recv(read_size)
        except BlockingIOError:
            break
        except InterruptedError:
            continue
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        buf.extend(chunk)
        bytes_read += len(chunk)
        if len(chunk) < read_size:
            break
    return bytes(buf)
