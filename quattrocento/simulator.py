"""Standalone TCP server that mimics the OT BioLab+ rebroadcast stream.

Run with `python -m quattrocento.simulator` to serve synthetic Quattrocento-
format data on a local port. The main app, in `--source=rebroadcast` mode,
can then connect to this simulator instead of the real device for demos and
local development.

Wire format matches what the rebroadcast client expects:
- Accepts text command `startTX` to begin streaming.
- Replies with an 8-byte ASCII header.
- Streams frames of `<i2` (signed int16, little-endian), one frame per sample,
  `n_channels` int16s per frame.
- Channel layout:
    - Channels 0..9: synthesized force signals (sinusoids + event envelopes).
    - Channel 10: trigger pulse (pass `--trigger-channel=10` when connecting).
    - Channel n_channels-8: monotonic ramp (used by detection).
    - All other channels: low-amplitude baseline noise.
- Accepts text command `stopTX` to stop streaming. The connection stays open
  for another `startTX` until the client disconnects.
"""

from __future__ import annotations

import argparse
import logging
import socket
import time

import numpy as np
from numpy.typing import NDArray

from .protocol import SUPPORTED_SAMPLE_RATES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("quattrocento.simulator")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 31000
DEFAULT_N_CHANNELS = 64
DEFAULT_SAMPLE_RATE = 2048
HEADER = b"OTBSIMv1"  # exactly 8 bytes
RAMP_COLUMN = -8
AUX_CHANNEL = 10
FORCE_BASE_AMPLITUDE = 1000.0
FORCE_EVENT_AMPLITUDE = 8000.0
AUX_PULSE_VALUE = 30000
NOISE_STDDEV = 50.0
BATCH_SECONDS = 0.05


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quattrocento rebroadcast simulator (synthetic data over TCP)."
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--n-channels", type=int, default=DEFAULT_N_CHANNELS,
        help="Total channels per frame.",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
        choices=SUPPORTED_SAMPLE_RATES,
        help="Sample rate in Hz.",
    )
    parser.add_argument(
        "--trigger-interval", type=float, default=8.0,
        help="Seconds between aux-in trigger pulses.",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


class FrameSynthesizer:
    """Generates synthetic int16 frames in the Quattrocento channel layout."""

    def __init__(
        self,
        *,
        n_channels: int,
        sample_rate_hz: int,
        trigger_interval_seconds: float,
        seed: int,
    ) -> None:
        if n_channels < 16:
            raise ValueError("n_channels must be at least 16")
        if AUX_CHANNEL >= n_channels:
            raise ValueError("aux channel index out of range for given n_channels")

        self._n_channels = n_channels
        self._sample_rate_hz = sample_rate_hz
        self._trigger_interval_seconds = trigger_interval_seconds
        self._trigger_duration_seconds = 0.03
        self._sample_index = 0
        self._rng = np.random.default_rng(seed)

        force_count = 10
        self._phase_offsets = np.linspace(0.1, 2.2, force_count, endpoint=True)
        self._base_frequencies_hz = np.linspace(0.28, 0.7, force_count, endpoint=True)
        self._event_profile = np.linspace(7.0, 12.0, force_count, endpoint=True)

    def next_frames(self, sample_count: int) -> bytes:
        timestamps = (
            np.arange(self._sample_index, self._sample_index + sample_count, dtype=np.float64)
            / self._sample_rate_hz
        )
        self._sample_index += sample_count

        frame = np.zeros((sample_count, self._n_channels), dtype=np.int16)

        forces = self._synthesize_forces(timestamps)
        frame[:, :10] = np.clip(forces, -32768, 32767).astype(np.int16)

        aux = self._synthesize_aux_in(timestamps)
        frame[:, AUX_CHANNEL] = aux

        ramp = (
            (self._sample_index - sample_count + np.arange(sample_count, dtype=np.int64))
            % 65536
            - 32768
        )
        frame[:, RAMP_COLUMN] = ramp.astype(np.int16)

        noise = self._rng.normal(0.0, NOISE_STDDEV, size=(sample_count, self._n_channels))
        noise_mask = np.ones(self._n_channels, dtype=bool)
        noise_mask[:10] = False
        noise_mask[AUX_CHANNEL] = False
        noise_mask[RAMP_COLUMN] = False
        frame[:, noise_mask] += noise[:, noise_mask].astype(np.int16)

        return frame.tobytes()

    def _synthesize_forces(self, timestamps: NDArray[np.float64]) -> NDArray[np.float64]:
        base = FORCE_BASE_AMPLITUDE * (
            1.0
            + 0.15 * np.sin(
                2.0 * np.pi
                * self._base_frequencies_hz[np.newaxis, :]
                * timestamps[:, np.newaxis]
                + self._phase_offsets[np.newaxis, :]
            )
        )
        envelope = self._event_envelope(timestamps)
        event_response = (
            FORCE_EVENT_AMPLITUDE
            * envelope[:, np.newaxis]
            * (self._event_profile[np.newaxis, :] / self._event_profile.max())
        )
        return base + event_response

    def _synthesize_aux_in(self, timestamps: NDArray[np.float64]) -> NDArray[np.int16]:
        aux = np.zeros(timestamps.shape[0], dtype=np.int16)
        if self._trigger_interval_seconds <= 0:
            return aux
        phase = np.mod(timestamps, self._trigger_interval_seconds)
        pulse_mask = phase < self._trigger_duration_seconds
        aux[pulse_mask] = AUX_PULSE_VALUE
        return aux

    def _event_envelope(self, timestamps: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._trigger_interval_seconds <= 0:
            return np.zeros_like(timestamps)
        phase = np.mod(timestamps, self._trigger_interval_seconds)
        return np.exp(-0.5 * np.square((phase - 1.15) / 0.55))


def _serve_client(client_sock: socket.socket, args: argparse.Namespace) -> None:
    """Drive one client connection through start/stream/stop cycles."""
    streaming = False
    synthesizer = FrameSynthesizer(
        n_channels=args.n_channels,
        sample_rate_hz=args.sample_rate,
        trigger_interval_seconds=args.trigger_interval,
        seed=args.seed,
    )
    samples_per_batch = max(1, int(round(args.sample_rate * BATCH_SECONDS)))
    next_batch_at = time.perf_counter()
    command_buffer = bytearray()

    client_sock.setblocking(False)

    try:
        while True:
            try:
                command_bytes = _try_read_command(client_sock, command_buffer)
            except ConnectionError:
                return

            if command_bytes == b"startTX":
                if not streaming:
                    client_sock.sendall(HEADER)
                    streaming = True
                    logger.info("Received startTX, beginning stream...")
                    next_batch_at = time.perf_counter()
            elif command_bytes == b"stopTX":
                if streaming:
                    logger.info("Received stopTX, stopping stream.")
                    streaming = False
            elif command_bytes:
                # Unknown command — ignore but log.
                logger.warning(f"Ignoring unknown command {command_bytes!r}")

            if streaming:
                now = time.perf_counter()
                if now >= next_batch_at:
                    payload = synthesizer.next_frames(samples_per_batch)
                    try:
                        client_sock.sendall(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    next_batch_at += BATCH_SECONDS
                    if next_batch_at < now - BATCH_SECONDS:
                        # Fell badly behind — resync to avoid runaway catch-up.
                        next_batch_at = now + BATCH_SECONDS
                else:
                    time.sleep(min(0.005, max(0.0, next_batch_at - now)))
            else:
                time.sleep(0.01)
    finally:
        try:
            client_sock.close()
        except OSError:
            pass


def _try_read_command(
    client_sock: socket.socket, buffer: bytearray
) -> bytes | None:
    """Non-blocking read; searches for tokens in the persistent buffer.

    Returns the token (e.g. b"startTX") if found and consumes the buffer
    up to that point. Returns None if no command is found.
    Raises ConnectionError if the socket is closed.
    """
    try:
        chunk = client_sock.recv(1024)
        if not chunk:
            raise ConnectionError("Client disconnected")
        buffer.extend(chunk)
    except BlockingIOError:
        pass
    except (ConnectionResetError, OSError):
        raise ConnectionError("Socket error")

    for token in (b"startTX", b"stopTX"):
        idx = buffer.find(token)
        if idx != -1:
            # Found a command! Consume the buffer up to and including the token.
            del buffer[: idx + len(token)]
            return token
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    server.settimeout(1.0)  # lets KeyboardInterrupt be processed on Windows
    logger.info(
        f"listening on {args.host}:{args.port} "
        f"(n_channels={args.n_channels}, sample_rate={args.sample_rate} Hz)"
    )

    try:
        while True:
            try:
                client_sock, addr = server.accept()
            except socket.timeout:
                continue
            logger.info(f"client connected from {addr}")
            try:
                _serve_client(client_sock, args)
            except Exception as exc:  # pragma: no cover
                logger.exception(f"client error: {exc}")
            logger.info("client disconnected")
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        server.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
