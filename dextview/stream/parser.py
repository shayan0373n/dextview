import numpy as np

from ..config import DextViewConfig
from ..models import DataBatch

_MAX_BUFFER_BYTES = 50 * 1024 * 1024
_INT16_FULL_SCALE = 32768.0


class FrameParser:
    """Stateful int16 frame parser: bytes in, DataBatch out.

    Owns the receive buffer and sample counter so callers only deal with
    raw bytes on one side and DataBatch on the other.
    """

    def __init__(self, config: DextViewConfig) -> None:
        self._config = config
        self._frame_bytes = 2 * config.n_channels
        self._byte_buffer = bytearray()
        self._sample_index = 0

        self._scale_vector = np.ones(config.n_channels, dtype=np.float64)
        if config.channel_scales:
            for idx, scale in config.channel_scales.items():
                self._scale_vector[idx] = scale

    def feed(self, raw: bytes) -> None:
        """Append bytes to the internal buffer; trim overflow if needed."""
        self._byte_buffer.extend(raw)
        if len(self._byte_buffer) > _MAX_BUFFER_BYTES:
            excess = len(self._byte_buffer) - _MAX_BUFFER_BYTES
            samples_to_drop = (excess + self._frame_bytes - 1) // self._frame_bytes
            bytes_to_drop = samples_to_drop * self._frame_bytes
            del self._byte_buffer[:bytes_to_drop]
            self._sample_index += samples_to_drop

    def drain(self) -> DataBatch:
        """Extract all complete frames from the buffer and return them as a batch."""
        sample_count = len(self._byte_buffer) // self._frame_bytes
        if sample_count == 0:
            return DataBatch(
                timestamps=np.empty(0, dtype=np.float64),
                signals=np.empty((0, self._config.n_channels), dtype=np.float64),
            )
        bytes_to_parse = sample_count * self._frame_bytes
        raw = bytes(self._byte_buffer[:bytes_to_parse])
        del self._byte_buffer[:bytes_to_parse]

        signals = (
            np.frombuffer(raw, dtype="<i2")
            .reshape(sample_count, self._config.n_channels)
            .astype(np.float64)
        )
        signals /= _INT16_FULL_SCALE
        signals *= self._scale_vector
        sample_indices = np.arange(
            self._sample_index, self._sample_index + sample_count, dtype=np.int64
        )
        timestamps = sample_indices.astype(np.float64) / self._config.sample_rate_hz
        self._sample_index += sample_count
        return DataBatch(timestamps=timestamps, signals=signals)
