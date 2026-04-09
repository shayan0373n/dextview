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
    """Processed post-trigger window used for visualization.

    Windows may span multiple DataBatches — samples are copied into a
    fixed-length buffer until the post-trigger window is full.
    `is_scaled` records whether `finger_forces`/`finger_ranges` have
    been normalized to % MVC, so downstream display can label units
    correctly without tracking session state.
    """

    timestamps: NDArray[np.float64]
    finger_forces: NDArray[np.float64]
    finger_ranges: NDArray[np.float64]
    finger_labels: tuple[str, ...]
    is_scaled: bool = False
