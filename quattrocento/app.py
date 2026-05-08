from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("quattrocento.app")

from PyQt5 import QtCore, QtWidgets

from .capture_log import CaptureLogger
from .channels import load_channels
from .hooks import PassedTenPercentRightIndex
from .config import QuattrocentoConfig
from .controller import QuattrocentoController
from .device import QuattrocentoStream
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
from .ui import QuattrocentoMainWindow


def _parse_auto_or_int(raw: str) -> str | int:
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
        choices=("real", "rebroadcast"),
        required=True,
        help=(
            "Data source type. 'real' connects directly to a Quattrocento "
            "device; 'rebroadcast' connects to OT BioLab+ (or the bundled "
            "simulator: python -m quattrocento.simulator)."
        ),
    )
    common.add_argument(
        "--channels",
        required=True,
        metavar="FILE",
        help=(
            "TOML file defining channel labels and optionally trigger_channel. "
            "See examples/channels_default.toml for format."
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
        help="Trigger detection threshold.",
    )
    common.add_argument(
        "--sample-rate",
        type=_parse_auto_or_int,
        default=None,
        help=(
            f"Sample rate (Hz). One of {SUPPORTED_SAMPLE_RATES}. "
            "Rebroadcast also accepts 'auto' (default 'auto')."
        ),
    )
    common.add_argument("--host", type=str, default=None, help="Device/server host.")
    common.add_argument("--port", type=int, default=None, help="Device/server port.")
    common.add_argument(
        "--n-channels", dest="n_channels", type=_parse_auto_or_int, default=None,
        help=(
            "Number of channels. Real source: integer. Rebroadcast: integer or 'auto'."
        ),
    )
    common.add_argument(
        "--log-dir", type=str, default=None,
        help=(
            "If set, write each captured trigger window as JSON under "
            "<log-dir>/session_<timestamp>/event_NNNNN.json."
        ),
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

    return parser.parse_args(argv)


def _validate_channel_indices(
    channel_labels: dict[int, str],
    trigger_channel: int,
    n_channels: int,
) -> None:
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
) -> tuple[QuattrocentoStream, StreamMeta]:
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
    )
    stream = QuattrocentoStream(
        config,
        handshake_kind="real",
        host=args.host,
        port=args.port,
        nch=nch_code,
        decimation_enabled=args.decimation_enabled,
        rec_on=args.rec_on,
        input_conf2_bytes=input_conf2_bytes,
    )
    meta = StreamMeta(
        channel_labels=channel_labels,
        config=config,
    )
    return stream, meta


def _build_rebroadcast_stream(
    args: argparse.Namespace,
    channel_labels: dict[int, str],
) -> tuple[QuattrocentoStream, StreamMeta]:
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
    )
    stream = QuattrocentoStream(
        config=config,
        handshake_kind="rebroadcast",
        host=args.host,
        port=args.port,
    )
    meta = StreamMeta(
        channel_labels=channel_labels,
        config=config,
    )
    return stream, meta


def main(argv: list[str] | None = None) -> int:
    """Create and run the GUI application event loop."""
    args = parse_args(argv)

    try:
        channel_labels = load_channels(args.channels)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Failed to load channels file {args.channels!r}: {exc}") from exc

    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        qt_app = QtWidgets.QApplication(sys.argv)

    if args.source == "real":
        stream, meta = _build_real_stream(args, channel_labels)
    else:
        stream, meta = _build_rebroadcast_stream(args, channel_labels)

    processor = TriggerWindowProcessor(stream.config)
    window = QuattrocentoMainWindow(
        channel_labels=channel_labels,
        trigger_channel=stream.config.trigger_channel,
    )
    event_hooks = [CaptureLogger(args.log_dir)] if args.log_dir else []
    controller = QuattrocentoController(
        stream.config, stream, processor, window, meta,
        stream_hooks=[PassedTenPercentRightIndex()],
        event_hooks=event_hooks,
    )
    controller.start()

    try:
        return qt_app.exec_()
    finally:
        stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
