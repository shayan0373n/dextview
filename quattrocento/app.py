from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("quattrocento.app")

from PyQt5 import QtCore, QtWidgets

from .capture_log import CaptureLogger
from .hooks import PassedTenPercentRightIndex
from .config import QuattrocentoConfig
from .controller import QuattrocentoController
from .device import QuattrocentoStream
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


def _parse_channel_list(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        raise argparse.ArgumentTypeError("channel list cannot be empty")
    try:
        return tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid channel list {raw!r}: must be comma-separated integers"
        ) from exc


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
    # Validation for source-specific required args lives in _build_real_stream /
    # _build_rebroadcast_stream rather than here. If a third source is added, or
    # per-source arg lists diverge further, migrate to argparse subparsers instead.
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
            "Number of channels. Real source: integer (resolves to the "
            "smallest device configuration >= this value). Rebroadcast: "
            "integer or 'auto'."
        ),
    )
    common.add_argument(
        "--force-channels", type=_parse_channel_list, default=None,
        help=(
            "Comma-separated list of 10 force channel indices. Order matters: "
            "L Thumb, L Index, L Middle, L Ring, L Little, "
            "R Thumb, R Index, R Middle, R Ring, R Little. "
            "Required for --source=real. Rebroadcast defaults to channels 0-9."
        ),
    )
    common.add_argument(
        "--aux-in-channel", type=int, default=None,
        help=(
            "Index of the aux-in/trigger channel. "
            "Required for --source=real. Rebroadcast defaults to the last channel."
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


def _build_real_stream(args: argparse.Namespace) -> tuple[QuattrocentoConfig, QuattrocentoStream]:
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
    if args.force_channels is None or args.aux_in_channel is None:
        raise SystemExit(
            "--force-channels and --aux-in-channel are required for --source=real"
        )

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
        window_seconds=args.window_seconds,
        window_offset_seconds=args.window_offset_seconds,
        trigger_threshold=args.trigger_threshold,
    )
    stream = QuattrocentoStream(
        config,
        handshake_kind="real",
        host=args.host,
        port=args.port,
        n_channels=NCH_BITS_TO_NUM_CHANNELS[nch_code],
        force_channel_indices=tuple(args.force_channels),
        aux_in_channel_index=args.aux_in_channel,
        nch=nch_code,
        decimation_enabled=args.decimation_enabled,
        rec_on=args.rec_on,
        input_conf2_bytes=input_conf2_bytes,
    )
    return config, stream


def _build_rebroadcast_stream(
    args: argparse.Namespace,
) -> tuple[QuattrocentoConfig, QuattrocentoStream]:
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

    config = QuattrocentoConfig(
        sample_rate_hz=sampling_rate_hz,
        window_seconds=args.window_seconds,
        window_offset_seconds=args.window_offset_seconds,
        trigger_threshold=args.trigger_threshold,
    )

    force_channels = args.force_channels
    if force_channels is None:
        force_channels = tuple(range(config.sensor_count))
        logger.info("--force-channels not set; defaulting to channels %s", list(force_channels))

    aux_in_channel = args.aux_in_channel
    if aux_in_channel is None:
        aux_in_channel = n_channels - 1
        logger.info("--aux-in-channel not set; defaulting to channel %d (last)", aux_in_channel)

    stream = QuattrocentoStream(
        config=config,
        handshake_kind="rebroadcast",
        host=args.host,
        port=args.port,
        n_channels=n_channels,
        force_channel_indices=tuple(force_channels),
        aux_in_channel_index=aux_in_channel,
    )
    return config, stream


def main(argv: list[str] | None = None) -> int:
    """Create and run the GUI application event loop."""
    args = parse_args(argv)

    qt_app = QtWidgets.QApplication.instance()
    if qt_app is None:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        qt_app = QtWidgets.QApplication(sys.argv)

    if args.source == "real":
        config, stream = _build_real_stream(args)
    else:
        config, stream = _build_rebroadcast_stream(args)

    processor = TriggerWindowProcessor(config)
    window = QuattrocentoMainWindow(config.finger_labels)
    event_hooks = [CaptureLogger(args.log_dir)] if args.log_dir else []
    controller = QuattrocentoController(
        config, stream, processor, window,
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
