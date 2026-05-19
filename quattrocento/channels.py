from pathlib import Path
import tomllib


_VALID_KINDS = ("finger", "emg", "trigger")


def load_channels(
    path: str | Path,
) -> tuple[dict[int, str], dict[int, float], dict[int, str]]:
    """Parse a TOML channels file.

    Returns three index-keyed maps: label, scale, and kind. ``kind`` is
    "finger" by default and can be set to "emg" or "trigger".

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

    channel_labels: dict[int, str] = {}
    channel_scales: dict[int, float] = {}
    channel_kinds: dict[int, str] = {}
    for label, conf in raw_labels.items():
        if not isinstance(label, str):
            raise ValueError(f"Label key must be a string, got {label!r}")

        if isinstance(conf, dict):
            idx = conf.get("index")
            if idx is None:
                raise ValueError(f"Label {label!r}: missing 'index'")
            scale = float(conf.get("scale", 1.0))
            kind = conf.get("kind", "finger")
        else:
            raise ValueError(f"Label {label!r}: invalid config {conf!r}")

        if not isinstance(idx, int):
            raise ValueError(
                f"Label {label!r}: index must be an integer, got {idx!r}"
            )
        if idx < 0:
            raise ValueError(f"Label {label!r}: index must be >= 0, got {idx}")
        if idx in channel_labels:
            existing = channel_labels[idx]
            raise ValueError(
                f"Duplicate channel index {idx} (labels {existing!r} and {label!r})"
            )
        if not isinstance(kind, str) or kind not in _VALID_KINDS:
            raise ValueError(
                f"Label {label!r}: kind must be one of {_VALID_KINDS}, got {kind!r}"
            )
        channel_labels[idx] = label
        channel_scales[idx] = scale
        channel_kinds[idx] = kind

    return channel_labels, channel_scales, channel_kinds
