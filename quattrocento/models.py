from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class DataBatch:
    """One contiguous sample block from the stream."""

    timestamps: NDArray[np.float64]
    forces: NDArray[np.float64]
    aux_in: NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class CapturedWindow:
    """Processed post-trigger window used for visualization."""

    timestamps: NDArray[np.float64]
    finger_forces: NDArray[np.float64]
    finger_ranges: NDArray[np.float64]
    finger_labels: tuple[str, ...]
