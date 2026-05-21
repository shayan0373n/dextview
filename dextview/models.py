from enum import StrEnum
from dataclasses import dataclass
from collections.abc import Iterator, Mapping
from typing import Protocol

from .config import DextViewConfig

import numpy as np
from numpy.typing import NDArray


class ChannelKind(StrEnum):
    """The type of data stream recorded on a channel."""

    FINGER = "finger"
    EMG = "emg"
    TRIGGER = "trigger"


@dataclass(slots=True, frozen=True)
class ChannelInfo:
    """Complete metadata for a single device channel."""

    label: str
    kind: ChannelKind
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class ChannelGroup:
    """Channels of a single kind, pre-sorted by index."""

    indices: tuple[int, ...]
    labels: tuple[str, ...]


_EMPTY_GROUP = ChannelGroup(indices=(), labels=())


class Channels(Mapping[int, ChannelInfo]):
    """A read-only mapping of channel index to ChannelInfo.

    Static after construction. Kind-based lookups are pre-computed once
    and served via ``by_kind()``. Trigger semantics are a consumer concern;
    constraints (e.g. "exactly one trigger") are enforced at construction
    sites such as ``load_channels``.
    """

    def __init__(self, channels_dict: dict[int, ChannelInfo]) -> None:
        """Initialize the mapping and pre-compute kind groupings."""
        self._data = dict(channels_dict)
        self._by_kind = self._build_groups(self._data)

    @staticmethod
    def _build_groups(data: dict[int, ChannelInfo]) -> dict[ChannelKind, ChannelGroup]:
        """Sort and group channels by kind into frozen ChannelGroups."""
        groups: dict[ChannelKind, list[tuple[int, ChannelInfo]]] = {}
        for idx, info in sorted(data.items()):
            groups.setdefault(info.kind, []).append((idx, info))

        return {
            kind: ChannelGroup(
                indices=tuple(idx for idx, _ in entries),
                labels=tuple(info.label for _, info in entries),
            )
            for kind, entries in groups.items()
        }

    def __getitem__(self, key: int) -> ChannelInfo:
        """Return the ChannelInfo for the given channel index."""
        return self._data[key]

    def __iter__(self) -> Iterator[int]:
        """Iterate over the channel indices."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of registered channels."""
        return len(self._data)

    def by_kind(self, kind: ChannelKind) -> ChannelGroup:
        """Return the pre-computed ChannelGroup for *kind*."""
        return self._by_kind.get(kind, _EMPTY_GROUP)


@dataclass(slots=True, frozen=True)
class DataBatch:
    """One contiguous sample block from the stream."""

    timestamps: NDArray[np.float64]   # (samples,)
    signals: NDArray[np.float64]      # (samples, n_channels)

    def __post_init__(self) -> None:
        if self.timestamps.shape[0] != self.signals.shape[0]:
            raise ValueError("Timestamps and signals must have the same number of samples")


@dataclass(slots=True, frozen=True)
class StreamMeta:
    """Session-level context shared by stream hooks and embedded in captured windows.

    `channels` maps channel index → ChannelInfo. `baseline` and `peak` are
    per-channel calibration references with the same width as `DataBatch.signals`;
    they are `None` until calibration has completed. Consumers that want
    %-normalised values compute `(signals - baseline) / (peak - baseline) * 100` themselves.
    """

    channels: Channels
    config: DextViewConfig
    baseline: NDArray[np.float64] | None = None  # (n_channels,)
    peak: NDArray[np.float64] | None = None       # (n_channels,)
    empty: NDArray[np.float64] | None = None      # (n_channels,) — no-contact reference, display only


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


class Stream(Protocol):
    """Structural interface satisfied by all stream types."""

    @property
    def config(self) -> DextViewConfig:
        """The runtime configuration of the stream."""
        ...

    def read_batch(self) -> DataBatch:
        """Read the latest chunk of data from the stream."""
        ...

    def close(self) -> None:
        """Gracefully shut down the stream."""
        ...


class _Hook(Protocol):
    """Shared interface for all hooks registered with the controller."""

    name: str

    def set_active(self, active: bool) -> None:
        """Enable or disable the hook."""
        ...

    def reset(self) -> None:
        """Reset the hook's internal state."""
        ...


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
