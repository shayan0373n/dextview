from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("quattrocento.app")

from PyQt5 import QtCore, QtWidgets

from .capture_log import CaptureLogger
from .channels import load_channels
from .hooks import PassedTenPercentAnyFinger
from .config import QuattrocentoConfig
from .controller import QuattrocentoController
from .models import StreamMeta
from .processing import TriggerWindowProcessor
from .protocol import (
    DEFAULT_INPUT_CONF2_BYTES,
    NCH_BITS_TO_NUM_CHANNELS,
    SUPPORTED_SAMPLE_RATES,
    smallest_nch_for_channel_count,
)
from .rebroadcast_detect import detect_stream_params
from .settings import load_input_conf2_bytes
from .stream import DirectStream, ProxyStream, RebroadcastStream
from .ui import QuattrocentoMainWindow


def _parse_auto_or_int(raw: str) -> str | int:
    """Parse a string as 'auto' or an integer."""
    if raw.lower() == "auto":
        return "auto"
    try:
        return int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an integer or 'auto', got {raw!r}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the Quattrocento application."""
    parser = argparse.ArgumentParser(
        description="Run the Quattrocento trigger-based force GUI."
    )
    common = parser.add_argument_group("common settings")
    common.add_argument(
        "--source",
        choices=("real", "rebroadcast", "proxy"),
        required=True,
        help=(
            "Data source type. 'real': connects directly to a Quattrocento device. "
            "'rebroadcast': connects to a rebroadcast server (e.g. OT BioLab+ or "
            "the bundled simulator). "
            "'proxy': listens for an upstream client, forwards its commands to the "
            "device, and taps the data stream locally."
        ),
    )
    common.add_argument(
        "--channels",
        required=True,
        metavar="FILE",
        help=(
            "TOML file defining channel labels and optionally trigger_channel. "
            "See quattrocento/channels_default.toml for format."
        ),
    )
    common.add_argument(
        "--trigger-channel",
        type=int,
        required=True,
        dest="trigger_channel",
        help="Index of the trigger channel.",
    )
    common.add_argument(
        "--window-seconds",
        type=float,
        default=5.0,
        help="Total capture window length (including pre-trigger offset).",
    )
    common.add_argument(
        "--window-offset",
        dest="window_offset_seconds",
        type=float,
        default=0.0,
        help=(
            "Pre-trigger offset in seconds. Negative or zero. "
            "E.g. -1.0 with a 5.0s window captures 1.0s before the "
            "trigger and 4.0s after it."
        ),
    )
    common.add_argument(
        "--trigger-threshold",
        type=float,
        default=0.5,
        help=(
            "Trigger detection threshold in physical units (post-normalization). "
            "Signal range is [-scale, +scale] per channel's scale factor in the "
            "TOML (default 1.0). With trigger scale=5.0, threshold=0.5 fires at "
            "10%% of full scale. Default: %(default)s."
        ),
    )
    common.add_argument(
        "--sample-rate",
        type=_parse_auto_or_int,
        default=None,
        help=(
            f"Sample rate (Hz). One of {SUPPORTED_SAMPLE_RATES}. "
            "Rebroadcast also accepts 'auto' (default 'auto'). "
            "Not used for proxy (sniffed from the wire)."
        ),
    )
    common.add_argument("--host", type=str, default=None, help="Device/server host.")
    common.add_argument("--port", type=int, default=None, help="Device/server port.")
    common.add_argument(
        "--n-channels", dest="n_channels", type=_parse_auto_or_int, default=None,
        help=(
            "Number of channels. Real source: integer. Rebroadcast: integer or 'auto'. "
            "Not used for proxy (sniffed from the wire)."
        ),
    )
    common.add_argument(
        "--log-dir", type=str, default=None,
        help=(
            "If set, write each captured trigger window as JSON under "
            "<log-dir>/session_<timestamp>/event_NNNNN.json."
        ),
    )
    common.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log verbosity (default: WARNING).",
    )

    real = parser.add_argument_group("real source")
    real.add_argument(
        "--decimation",
        dest="decimation_enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    real.add_argument("--rec-on", action="store_true", help="Enable on-device recording.")
    real.add_argument(
        "--conf2-config", type=str, default=None,
        help="Optional TOML file with conf2 input-block settings (real source only).",
    )

    proxy = parser.add_argument_group("proxy source")
    proxy.add_argument(
        "--proxy-listen-host",
        type=str,
        default="127.0.0.1",
        dest="proxy_listen_host",
        help="Host to bind the proxy listener on (default: 127.0.0.1).",
    )
    proxy.add_argument(
        "--proxy-listen-port",
        type=int,
        default=23456,
        dest="proxy_listen_port",
        help="Port to bind the proxy listener on (default: 23456).",
    )

    return parser.parse_args(argv)


def _validate_channel_indices(
    channel_labels: dict[int, str],
    trigger_channel: int,
    n_channels: int,
) -> None:
    """Validate that channel indices and trigger channel are within range."""
    for idx, label in channel_labels.items():
        if idx >= n_channels:
            raise SystemExit(
                f"Channel label {label!r} has index {idx} but stream only "
                f"has {n_channels} channels (indices 0–{n_channels - 1})."
            )
    if trigger_channel >= n_channels:
        raise SystemExit(
            f"Trigger channel {trigger_channel} is out of range for a stream "
            f"with {n_channels} channels (indices 0–{n_channels - 1})."
        )


def _build_real_stream(
    args: argparse.Namespace,
    channel_labels: dict[int, str],
    channel_scales: dict[int, float],
) -> tuple[DirectStream, StreamMeta]:
    """Build a DirectStream for a real Quattrocento device connection."""
    if args.host is None or args.port is None:
        raise SystemExit("--host and --port are required for --source=real")
    if args.sample_rate is None or args.sample_rate == "auto":
        raise SystemExit("--sample-rate is required (and must be an integer) for --source=real")
    if args.n_channels is None:
        raise SystemExit("--n-channels is required for --source=real")
    if not isinstance(args.n_channels, int):
        raise SystemExit("--n-channels must be an integer for --source=real")
    try:
        nch_code = smallest_nch_for_channel_count(args.n_channels)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    n_channels = NCH_BITS_TO_NUM_CHANNELS[nch_code]
    _validate_channel_indices(channel_labels, args.trigger_channel, n_channels)

    if args.conf2_config is not None:
        try:
            input_conf2_bytes = load_input_conf2_bytes(args.conf2_config)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            raise SystemExit(
                f"Failed to load conf2 config {args.conf2_config!r}: {exc}"
            ) from exc
    else:
        input_conf2_bytes = DEFAULT_INPUT_CONF2_BYTES

    config = QuattrocentoConfig(
        sample_rate_hz=args.sample_rate,
        n_channels=n_channels,
        window_seconds=args.window_seconds,
        window_offset_seconds=args.window_offset_seconds,
        trigger_threshold=args.trigger_threshold,
        trigger_channel=args.trigger_channel,
        channel_scales=channel_scales,
    )
    stream = DirectStream(
        config,
        host=args.host,
        port=args.port,
        nch=nch_code,
        decimation_enabled=args.decimation_enabled,
        rec_on=args.rec_on,
        input_conf2_bytes=input_conf2_bytes,
    )
    meta = StreamMeta(channel_labels=channel_labels, config=config)
    return stream, meta


def _build_rebroadcast_stream(
    args: argparse.Namespace,
    channel_labels: dict[int, str],
    channel_scales: dict[int, float],
) -> tuple[RebroadcastStream, StreamMeta]:
    """Build a RebroadcastStream for connecting to a TCP stream (e.g., OT BioLab+)."""
    if args.host is None or args.port is None:
        raise SystemExit("--host and --port are required for --source=rebroadcast")

    nch_arg = args.n_channels
    rate_arg = args.sample_rate if args.sample_rate is not None else "auto"

    detect_nch = nch_arg is None or nch_arg == "auto"
    detect_rate = rate_arg == "auto"
    known_nch = nch_arg if isinstance(nch_arg, int) else None
    known_rate = rate_arg if isinstance(rate_arg, int) else None

    if detect_nch or detect_rate:
        detected = detect_stream_params(
            args.host,
            args.port,
            detect_nch=detect_nch,
            detect_rate=detect_rate,
            known_nch=known_nch,
            known_rate=known_rate,
        )
        n_channels = detected.n_channels
        sampling_rate_hz = detected.sampling_rate_hz
    else:
        n_channels = known_nch
        sampling_rate_hz = known_rate

    _validate_channel_indices(channel_labels, args.trigger_channel, n_channels)

    config = QuattrocentoConfig(
        sample_rate_hz=sampling_rate_hz,
        n_channels=n_channels,
        window_seconds=args.window_seconds,
        window_offset_seconds=args.window_offset_seconds,
        trigger_threshold=args.trigger_threshold,
        trigger_channel=args.trigger_channel,
        channel_scales=channel_scales,
    )
    stream = RebroadcastStream(config=config, host=args.host, port=args.port)
    meta = StreamMeta(channel_labels=channel_labels, config=config)
    return stream, meta


def _build_proxy_stream(
    args: argparse.Namespace,
    channel_labels: dict[int, str],
    channel_scales: dict[int, float],
) -> tuple[ProxyStream, StreamMeta]:
    """Build a ProxyStream to sit between an upstream client and a device."""
    if args.host is None or args.port is None:
        raise SystemExit("--host and --port are required for --source=proxy (the device)")

    try:
        stream = ProxyStream.listen_and_accept(
            listen_host=args.proxy_listen_host,
            listen_port=args.proxy_listen_port,
            device_host=args.host,
            device_port=args.port,
            window_seconds=args.window_seconds,
            window_offset_seconds=args.window_offset_seconds,
            trigger_threshold=args.trigger_threshold,
            trigger_channel=args.trigger_channel,
        )
    except (ConnectionError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        _validate_channel_indices(channel_labels, args.trigger_channel, stream.config.n_channels)
    except SystemExit:
        stream.close()
        raise

    meta = StreamMeta(channel_labels=channel_labels, config=stream.config)
    return stream, meta


def main(argv: list[str] | None = None) -> int:
    """Create and run the GUI application event loop."""
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        channel_labels, channel_scales, channel_kinds = load_channels(args.channels)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Failed to load channels file {args.channels!r}: {exc}") from exc

    emg_channels = sorted(
        (idx, channel_labels[idx])
        for idx, kind in channel_kinds.items()
        if kind == "emg"
    )
    finger_indices = sorted(
        idx for idx, kind in channel_kinds.items()
        if kind == "finger"
    )

    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        qt_app = QtWidgets.QApplication(sys.argv)

    if args.source == "real":
        stream, meta = _build_real_stream(args, channel_labels, channel_scales)
    elif args.source == "rebroadcast":
        stream, meta = _build_rebroadcast_stream(args, channel_labels, channel_scales)
    else:
        stream, meta = _build_proxy_stream(args, channel_labels, channel_scales)

    processor = TriggerWindowProcessor(stream.config)
    window = QuattrocentoMainWindow(
        channel_labels=channel_labels,
        trigger_channel=stream.config.trigger_channel,
        sample_rate_hz=stream.config.sample_rate_hz,
        trigger_threshold=args.trigger_threshold,
        emg_channels=emg_channels,
    )
    event_hooks = [CaptureLogger(args.log_dir)] if args.log_dir else []
    controller = QuattrocentoController(
        stream.config, stream, processor, window, meta,
        stream_hooks=[PassedTenPercentAnyFinger(finger_indices=finger_indices)],
        event_hooks=event_hooks,
    )
    controller.start()

    try:
        return qt_app.exec_()
    finally:
        stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
