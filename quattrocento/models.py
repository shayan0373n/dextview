from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .config import QuattrocentoConfig

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True, frozen=True)
class DataBatch:
    """One contiguous sample block from the stream."""

    timestamps: NDArray[np.float64]   # (samples,)
    signals: NDArray[np.float64]      # (samples, n_channels)


@dataclass(slots=True, frozen=True)
class StreamMeta:
    """Session-level context shared by stream hooks and embedded in captured windows.

    `channel_labels` maps channel index → display label for a sparse subset of
    channels. `baseline` and `peak` are per-channel calibration references with
    the same width as `DataBatch.signals`; they are `None` until calibration
    has completed. Consumers that want %-normalised values compute
    `(signals - baseline) / (peak - baseline) * 100` themselves.
    """

    channel_labels: Mapping[int, str]
    config: QuattrocentoConfig
    baseline: NDArray[np.float64] | None = None  # (n_channels,)
    peak: NDArray[np.float64] | None = None       # (n_channels,)

    def index_of(self, label: str) -> int:
        """Return the channel index for a label; raises KeyError if absent."""
        for idx, lbl in self.channel_labels.items():
            if lbl == label:
                return idx
        raise KeyError(label)


@dataclass(slots=True, frozen=True)
class CapturedWindow:
    """A self-contained record of one trigger event.

    Composition of a DataBatch (the captured samples), the StreamMeta
    snapshot at capture time, and the sample index within the batch where the
    trigger edge fired. All context needed to interpret or re-process the event
    travels with the window — no external session state required.
    """

    batch: DataBatch
    meta: StreamMeta
    trigger_sample: int


class _Hook(Protocol):
    """Shared interface for all hooks registered with the controller."""

    name: str
    ui_controls: bool

    def set_active(self, active: bool) -> None: ...
    def reset(self) -> None: ...


class StreamHook(_Hook, Protocol):
    """Hook called with every live data batch.

    Data contract: `batch.signals` contains raw 16-bit counts for all device
    channels. `meta.baseline` and `meta.peak` are also in raw 16-bit counts.
    Hooks requiring %-normalised values must perform their own scaling.
    """

    def __call__(self, batch: DataBatch, meta: StreamMeta) -> None: ...


class EventHook(_Hook, Protocol):
    """Hook called once per completed capture window.

    The window is self-contained: it carries its own `StreamMeta` snapshot
    taken at capture time, so hooks do not need a separate meta argument.
    """

    def __call__(self, window: CapturedWindow) -> None: ...
