from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import CapturedWindow, StreamMeta


class CaptureLogger:
    """Write each CapturedWindow to its own JSON file under a session dir.

    Implements the `EventHook` contract: instances are callable as
    ``logger(captured, meta)`` and pull current calibration values from
    ``meta`` rather than from controller-private state.
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

    def __call__(self, captured: CapturedWindow, meta: StreamMeta) -> None:
        self._count += 1
        path = self._session_dir / f"event_{self._count:05d}.json"
        payload = {
            "trigger_index": int(captured.trigger_index),
            "is_scaled": bool(captured.is_scaled),
            "finger_labels": list(captured.finger_labels),
            "finger_ranges": captured.finger_ranges.tolist(),
            "rest_means": (
                meta.rest_means.tolist() if meta.rest_means is not None else None
            ),
            "mvc_maxs": (
                meta.mvc_maxs.tolist() if meta.mvc_maxs is not None else None
            ),
            "timestamps": captured.timestamps.tolist(),
            "finger_forces": captured.finger_forces.tolist(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
