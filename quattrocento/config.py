from dataclasses import dataclass


@dataclass(slots=True)
class QuattrocentoConfig:
    """Runtime configuration for stream processing and GUI update cadence."""

    sample_rate_hz: int = 512
    n_channels: int = 120
    window_seconds: float = 5.0
    window_offset_seconds: float = 0.0
    batch_duration_seconds: float = 0.05
    trigger_threshold: float = 0.5
    trigger_channel: int = 0
    ui_refresh_ms: int = 30
    channel_scales: dict[int, float] | None = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.window_offset_seconds > 0:
            raise ValueError("window_offset_seconds must be <= 0 (pre-trigger only)")
        if -self.window_offset_seconds >= self.window_seconds:
            raise ValueError(
                "abs(window_offset_seconds) must be less than window_seconds "
                "(pre-trigger cannot span the entire window; trigger sample requires at least one post-trigger slot)"
            )
        if self.batch_duration_seconds <= 0:
            raise ValueError("batch_duration_seconds must be positive")
        if self.ui_refresh_ms <= 0:
            raise ValueError("ui_refresh_ms must be positive")

    @property
    def total_window_samples(self) -> int:
        return max(1, int(round(self.sample_rate_hz * self.window_seconds)))

    @property
    def pre_trigger_samples(self) -> int:
        return max(0, int(round(self.sample_rate_hz * -self.window_offset_seconds)))

    @property
    def post_trigger_samples(self) -> int:
        return max(1, self.total_window_samples - self.pre_trigger_samples)
