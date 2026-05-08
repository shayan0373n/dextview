from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import CapturedWindow


class CaptureLogger:
    """Write each CapturedWindow to its own JSON file under a session dir.

    Implements the `EventHook` contract: instances are callable as
    ``logger(window)`` and are fully self-contained — all context needed to
    interpret the capture travels inside the window itself.
    """

    name = "Capture Logger"
    ui_controls = False

    def __init__(self, base_dir: str | Path) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = Path(base_dir) / f"session_{timestamp}"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    def set_active(self, active: bool) -> None:
        pass

    def reset(self) -> None:
        pass

    def __call__(self, window: CapturedWindow) -> None:
        self._count += 1
        path = self._session_dir / f"event_{self._count:05d}.json"
        payload = {
            "trigger_sample": window.trigger_sample,
            "trigger_channel": window.meta.config.trigger_channel,
            "sample_rate_hz": window.meta.config.sample_rate_hz,
            "channel_labels": {
                str(k): v for k, v in window.meta.channel_labels.items()
            },
            "baseline": (
                window.meta.baseline.tolist()
                if window.meta.baseline is not None
                else None
            ),
            "peak": (
                window.meta.peak.tolist()
                if window.meta.peak is not None
                else None
            ),
            "timestamps": window.batch.timestamps.tolist(),
            "signals": window.batch.signals.tolist(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
