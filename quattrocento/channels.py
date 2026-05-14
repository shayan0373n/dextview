from __future__ import annotations

from pathlib import Path
import tomllib


def load_channels(path: str | Path) -> tuple[dict[int, str], dict[int, float]]:
    """Parse a TOML channels file and return a mapping of channel index → label and index → scale.

    TOML format::

        [labels]
        "L Thumb"  = { index = 0, scale = 0.01667 }
        "L Index"  = { index = 1, scale = 0.01667 }
        "trigger"  = { index = 10, scale = 5.0 }
    """
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    raw_labels = payload.get("labels", {})
    if not isinstance(raw_labels, dict):
        raise ValueError("[labels] must be a TOML table")

    channel_labels: dict[int, str] = {}
    channel_scales: dict[int, float] = {}
    for label, conf in raw_labels.items():
        if not isinstance(label, str):
            raise ValueError(f"Label key must be a string, got {label!r}")

        if isinstance(conf, dict):
            idx = conf.get("index")
            if idx is None:
                raise ValueError(f"Label {label!r}: missing 'index'")
            scale = float(conf.get("scale", 1.0))
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
        channel_labels[idx] = label
        channel_scales[idx] = scale

    return channel_labels, channel_scales
