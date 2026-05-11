from __future__ import annotations

from pathlib import Path
import tomllib


def load_channels(path: str | Path) -> dict[int, str]:
    """Parse a TOML channels file and return a mapping of channel index → label.

    TOML format::

        [labels]
        "L Thumb"  = 0
        "L Index"  = 1
        "trigger"  = 10
    """
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)

    raw_labels = payload.get("labels", {})
    if not isinstance(raw_labels, dict):
        raise ValueError("[labels] must be a TOML table")

    channel_labels: dict[int, str] = {}
    for label, idx in raw_labels.items():
        if not isinstance(label, str):
            raise ValueError(f"Label key must be a string, got {label!r}")
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

    return channel_labels
