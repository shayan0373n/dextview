from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


def _default_finger_sensor_map() -> Mapping[str, int]:
    return {
        "L Thumb": 0,
        "L Index": 1,
        "L Middle": 2,
        "L Ring": 3,
        "L Little": 4,
        "R Thumb": 5,
        "R Index": 6,
        "R Middle": 7,
        "R Ring": 8,
        "R Little": 9,
    }


@dataclass(slots=True)
class QuattrocentoConfig:
    """Runtime configuration for stream processing and GUI update cadence."""

    sample_rate_hz: int = 512
    window_seconds: float = 5.0
    window_offset_seconds: float = 0.0
    batch_duration_seconds: float = 0.05
    trigger_threshold: float = 0.5
    ui_refresh_ms: int = 30
    finger_sensor_map: Mapping[str, int] = field(
        default_factory=_default_finger_sensor_map
    )

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.window_offset_seconds > 0:
            raise ValueError("window_offset_seconds must be <= 0 (pre-trigger only)")
        if self.batch_duration_seconds <= 0:
            raise ValueError("batch_duration_seconds must be positive")
        if self.ui_refresh_ms <= 0:
            raise ValueError("ui_refresh_ms must be positive")
        self._validate_finger_sensor_map()

    @property
    def sensor_count(self) -> int:
        """Number of sensors, derived from the finger-sensor mapping."""
        return len(self.finger_sensor_map)

    @property
    def window_samples(self) -> int:
        """Number of samples to keep after each trigger edge."""
        return max(1, int(round(self.sample_rate_hz * self.window_seconds)))

    @property
    def pre_trigger_samples(self) -> int:
        """Number of samples to keep before each trigger edge."""
        return max(0, int(round(self.sample_rate_hz * -self.window_offset_seconds)))

    @property
    def total_window_samples(self) -> int:
        """Total captured window length: pre-trigger + post-trigger."""
        return self.pre_trigger_samples + self.window_samples

    @property
    def finger_labels(self) -> tuple[str, ...]:
        """Ordered finger labels used by plots and legends."""
        return tuple(self.finger_sensor_map.keys())

    def _validate_finger_sensor_map(self) -> None:
        if not self.finger_sensor_map:
            raise ValueError("finger_sensor_map cannot be empty")

        assigned_sensors: set[int] = set()
        for finger_name, sensor_idx in self.finger_sensor_map.items():
            if not isinstance(sensor_idx, int):
                raise ValueError(
                    f"sensor index for {finger_name!r} must be int, got {sensor_idx!r}"
                )
            if sensor_idx < 0 or sensor_idx >= self.sensor_count:
                raise ValueError(
                    f"sensor index {sensor_idx} for {finger_name!r} "
                    f"must be between 0 and {self.sensor_count - 1}"
                )
            if sensor_idx in assigned_sensors:
                raise ValueError(
                    f"sensor index {sensor_idx} is assigned to multiple fingers"
                )
            assigned_sensors.add(sensor_idx)
