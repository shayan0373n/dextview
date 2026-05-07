from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class DataBatch:
    """One contiguous sample block from the stream."""

    timestamps: NDArray[np.float64]
    forces: NDArray[np.float64]
    aux_in: NDArray[np.float64]


@dataclass(slots=True, frozen=True)
class StreamMeta:
    """Static + calibration context passed alongside raw data to hooks.

    Hooks receive raw `DataBatch` / `CapturedWindow` plus this object and
    are responsible for any derived computations (e.g. % MVC scaling).
    `rest_means` and `mvc_maxs` are per-finger, ordered to match
    `finger_labels`. They are `None` until calibration has completed.
    """

    finger_sensor_map: Mapping[str, int]
    finger_labels: tuple[str, ...]
    sample_rate_hz: int
    rest_means: NDArray[np.float64] | None = None
    mvc_maxs: NDArray[np.float64] | None = None


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
    trigger_index: int = 0


class _Hook(Protocol):
    """Shared interface for all hooks registered with the controller."""

    name: str
    ui_controls: bool  # True → controller creates a toggle + reset button pair

    def set_active(self, active: bool) -> None: ...
    def reset(self) -> None: ...


class StreamHook(_Hook, Protocol):
    """Hook called with every live data batch.

    Data Contract:
    - `batch.forces` contains raw 16-bit counts from the device.
    - `meta.rest_means` and `meta.mvc_maxs` are also in raw 16-bit counts.
    Hooks requiring %-MVC or physical units must perform their own scaling.
    """

    def __call__(self, batch: DataBatch, meta: StreamMeta) -> None: ...


class EventHook(_Hook, Protocol):
    """Hook called once per completed capture window."""

    def __call__(self, window: CapturedWindow, meta: StreamMeta) -> None: ...
