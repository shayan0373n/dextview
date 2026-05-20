from pathlib import Path
import tomllib

from .models import ChannelInfo, ChannelKind, Channels


def load_channels(
    path: str | Path,
) -> Channels:
    """Parse a TOML channels file.

    Returns an index-keyed map of ChannelInfo. Expects exactly one trigger channel.

    TOML format::

        [labels]
        "L Thumb"  = { index = 0, scale = 0.01667 }
        "EMG 1"    = { index = 11, scale = 1.0, kind = "emg" }
        "trigger"  = { index = 10, scale = 5.0, kind = "trigger" }
    """
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    raw_labels = payload.get("labels", {})
    if not isinstance(raw_labels, dict):
        raise ValueError("[labels] must be a TOML table")

    parsed: dict[int, ChannelInfo] = {}
    for label, conf in raw_labels.items():
        if not isinstance(label, str):
            raise ValueError(f"Label key must be a string, got {label!r}")

        if isinstance(conf, dict):
            idx = conf.get("index")
            if idx is None:
                raise ValueError(f"Label {label!r}: missing 'index'")
            scale = float(conf.get("scale", 1.0))
            kind_str = conf.get("kind", "finger")
        else:
            raise ValueError(f"Label {label!r}: invalid config {conf!r}")

        if not isinstance(idx, int):
            raise ValueError(
                f"Label {label!r}: index must be an integer, got {idx!r}"
            )
        if idx < 0:
            raise ValueError(f"Label {label!r}: index must be >= 0, got {idx}")
        if idx in parsed:
            existing = parsed[idx].label
            raise ValueError(
                f"Duplicate channel index {idx} (labels {existing!r} and {label!r})"
            )
        try:
            kind = ChannelKind(kind_str.lower().strip())
        except ValueError as exc:
            valid_kinds = [k.value for k in ChannelKind]
            raise ValueError(
                f"Label {label!r}: kind must be one of {valid_kinds}, got {kind_str!r}"
            ) from exc

        parsed[idx] = ChannelInfo(label=label, kind=kind, scale=scale)

    channels = Channels(parsed)

    # Validate that exactly one trigger channel exists
    trigger_indices = channels.by_kind(ChannelKind.TRIGGER).indices
    if len(trigger_indices) != 1:
        raise ValueError(
            f"Expected exactly one channel with kind='trigger', found {len(trigger_indices)}"
        )

    return channels

