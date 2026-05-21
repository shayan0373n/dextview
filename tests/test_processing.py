import unittest

import numpy as np

from dextview.config import DextViewConfig
from dextview.models import DataBatch, StreamMeta, ChannelInfo, ChannelKind, Channels
from dextview.processing import TriggerWindowProcessor

# 10 force channels + 1 trigger channel in every test.
N_CHANNELS = 11
TRIGGER_CH = 10


def _make_channels() -> Channels:
    """Build channels map for testing."""
    parsed = {}
    for i in range(N_CHANNELS):
        if i == TRIGGER_CH:
            parsed[i] = ChannelInfo(label=f"ch{i}", kind=ChannelKind.TRIGGER)
        else:
            parsed[i] = ChannelInfo(label=f"ch{i}", kind=ChannelKind.FINGER)
    return Channels(parsed)


def _meta(config: DextViewConfig) -> StreamMeta:
    """StreamMeta with unified channels dictionary."""
    return StreamMeta(
        channels=_make_channels(),
        config=config,
    )


def _processor(config: DextViewConfig) -> TriggerWindowProcessor:
    return TriggerWindowProcessor(config)


def _batch(
    timestamps: np.ndarray,
    signals: np.ndarray,
    trigger_col: np.ndarray,
) -> DataBatch:
    """Build a DataBatch where `signals` carries N_CHANNELS columns and
    trigger_col is inserted at TRIGGER_CH."""
    assert signals.shape[1] == N_CHANNELS - 1
    full = np.zeros((timestamps.shape[0], N_CHANNELS), dtype=np.float64)
    full[:, :signals.shape[1]] = signals
    full[:, TRIGGER_CH] = trigger_col
    return DataBatch(timestamps=timestamps, signals=full)


def _feed_warmup(
    proc: TriggerWindowProcessor,
    meta: StreamMeta,
    baseline: float = 0.0,
) -> None:
    """Advance proc past its warmup gate by feeding _warmup_samples flat-baseline samples."""
    n = proc._warmup_samples
    ts = np.arange(n, dtype=np.float64) / proc._sample_rate_hz
    aux = np.full(n, baseline)
    proc.process_batch(_batch(ts, np.zeros((n, N_CHANNELS - 1)), aux), meta)


class TriggerWindowProcessorTests(unittest.TestCase):
    def test_capture_collects_next_window_after_rising_edge(self) -> None:
        config = DextViewConfig(sample_rate_hz=4, n_channels=N_CHANNELS, window_seconds=1.0, trigger_channel=TRIGGER_CH)
        proc = _processor(config)
        meta = _meta(config)
        _feed_warmup(proc, meta)

        force_rows = np.array(
            [[row * 100.0 + sensor for sensor in range(10)] for row in range(7)],
            dtype=np.float64,
        )
        timestamps = np.arange(7, dtype=np.float64) / config.sample_rate_hz

        batch_1 = _batch(timestamps[:3], force_rows[:3], np.array([0.0, 1.0, 0.0]))
        batch_2 = _batch(timestamps[3:], force_rows[3:], np.zeros(4))

        self.assertEqual(len(proc.process_batch(batch_1, meta)), 0)
        captured_list = proc.process_batch(batch_2, meta)
        self.assertEqual(len(captured_list), 1)
        captured = captured_list[0]

        self.assertEqual(captured.batch.signals.shape, (4, N_CHANNELS))
        np.testing.assert_allclose(captured.batch.timestamps, timestamps[1:5])

    def test_multiple_captures_in_single_batch(self) -> None:
        config = DextViewConfig(sample_rate_hz=10, n_channels=N_CHANNELS, window_seconds=0.5, trigger_channel=TRIGGER_CH)
        proc = _processor(config)
        meta = _meta(config)
        _feed_warmup(proc, meta)

        n = 20
        timestamps = np.arange(n, dtype=np.float64) / 10.0
        signals = np.zeros((n, 10))
        aux = np.zeros(n)
        aux[2] = 5.0
        aux[10] = 5.0

        batch = _batch(timestamps, signals, aux)
        captured_list = proc.process_batch(batch, meta)

        self.assertEqual(len(captured_list), 2)
        self.assertEqual(captured_list[0].trigger_sample, 0)
        np.testing.assert_allclose(captured_list[0].batch.timestamps, timestamps[2:7])
        self.assertEqual(captured_list[1].trigger_sample, 0)
        np.testing.assert_allclose(captured_list[1].batch.timestamps, timestamps[10:15])

    def test_capture_detects_trigger_with_high_aux_baseline(self) -> None:
        config = DextViewConfig(sample_rate_hz=4, n_channels=N_CHANNELS, window_seconds=1.0, trigger_channel=TRIGGER_CH)
        proc = _processor(config)
        meta = _meta(config)
        _feed_warmup(proc, meta, baseline=8000.0)

        force_rows = np.array(
            [[row * 10.0 + sensor for sensor in range(10)] for row in range(8)],
            dtype=np.float64,
        )
        timestamps = np.arange(8, dtype=np.float64) / config.sample_rate_hz

        batch_1 = _batch(
            timestamps[:4], force_rows[:4],
            np.array([8000.0, 8000.0, 9000.0, 8000.0]),
        )
        batch_2 = _batch(
            timestamps[4:], force_rows[4:],
            np.full(4, 8000.0),
        )

        self.assertEqual(len(proc.process_batch(batch_1, meta)), 0)
        captured_list = proc.process_batch(batch_2, meta)
        self.assertEqual(len(captured_list), 1)

        np.testing.assert_allclose(captured_list[0].batch.timestamps, timestamps[2:6])
        self.assertEqual(captured_list[0].batch.signals.shape, (4, N_CHANNELS))

    def test_single_sample_pulse_triggers_capture(self) -> None:
        config = DextViewConfig(sample_rate_hz=4, n_channels=N_CHANNELS, window_seconds=1.0, trigger_channel=TRIGGER_CH)
        proc = _processor(config)
        meta = _meta(config)
        _feed_warmup(proc, meta, baseline=5000.0)

        force_rows = np.array(
            [[row * 10.0 + sensor for sensor in range(10)] for row in range(12)],
            dtype=np.float64,
        )
        timestamps = np.arange(12, dtype=np.float64) / config.sample_rate_hz

        warmup = _batch(timestamps[:4], force_rows[:4], np.full(4, 5000.0))
        pulse = _batch(timestamps[4:6], force_rows[4:6], np.array([6000.0, 5000.0]))
        tail = _batch(timestamps[6:10], force_rows[6:10], np.full(4, 5000.0))

        self.assertEqual(len(proc.process_batch(warmup, meta)), 0)
        self.assertEqual(len(proc.process_batch(pulse, meta)), 0)
        captured_list = proc.process_batch(tail, meta)
        self.assertEqual(len(captured_list), 1)

        np.testing.assert_allclose(captured_list[0].batch.timestamps, timestamps[4:8])
        self.assertEqual(captured_list[0].batch.signals.shape, (4, N_CHANNELS))

    def test_trigger_signal_preserved_in_capture(self) -> None:
        """The trigger channel is stored in signals, so loggers get it automatically."""
        config = DextViewConfig(sample_rate_hz=4, n_channels=N_CHANNELS, window_seconds=1.0, trigger_channel=TRIGGER_CH)
        proc = _processor(config)
        meta = _meta(config)
        _feed_warmup(proc, meta)

        timestamps = np.arange(8, dtype=np.float64) / config.sample_rate_hz
        force_rows = np.zeros((8, 10))
        # Single pulse at index 1.
        aux = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        batch_1 = _batch(timestamps[:3], force_rows[:3], aux[:3])
        batch_2 = _batch(timestamps[3:], force_rows[3:], aux[3:])

        proc.process_batch(batch_1, meta)
        captured_list = proc.process_batch(batch_2, meta)
        self.assertEqual(len(captured_list), 1)

        captured = captured_list[0]
        # Trigger column is present in signals.
        trigger_col = captured.batch.signals[:, TRIGGER_CH]
        # The pulse sample (global index 1) should appear in the capture.
        self.assertTrue(np.any(trigger_col > 0))


class WindowOffsetTests(unittest.TestCase):
    """Tests for negative window-offset (pre-trigger) capture."""

    RATE = 8
    WINDOW_SECONDS = 1.0
    OFFSET_SECONDS = -0.5
    N_FORCE = 10

    def _make_processor(self) -> TriggerWindowProcessor:
        config = DextViewConfig(
            sample_rate_hz=self.RATE,
            n_channels=N_CHANNELS,
            window_seconds=self.WINDOW_SECONDS,
            window_offset_seconds=self.OFFSET_SECONDS,
            trigger_channel=TRIGGER_CH,
        )
        return TriggerWindowProcessor(config)

    def _make_meta(self) -> StreamMeta:
        """Create StreamMeta with test channels."""
        return StreamMeta(
            channels=_make_channels(),
            config=DextViewConfig(sample_rate_hz=self.RATE, n_channels=N_CHANNELS, trigger_channel=TRIGGER_CH),
        )

    def _stream(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        timestamps = np.arange(n, dtype=np.float64) / self.RATE
        # Each row of force channels encodes the global sample index.
        forces = np.broadcast_to(
            np.arange(n, dtype=np.float64)[:, None], (n, self.N_FORCE)
        ).copy()
        return timestamps, forces

    def _feed(
        self,
        processor: TriggerWindowProcessor,
        meta: StreamMeta,
        timestamps: np.ndarray,
        forces: np.ndarray,
        aux: np.ndarray,
        batch_size: int,
    ) -> list:
        captures = []
        n = timestamps.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            out = processor.process_batch(
                _batch(timestamps[start:end], forces[start:end], aux[start:end]),
                meta,
            )
            captures.extend(out)
        return captures

    def _assert_contiguous(
        self,
        captured,
        edge_index: int,
        expected_pre: int,
        expected_post: int,
    ) -> None:
        period = 1.0 / self.RATE
        n = expected_pre + expected_post
        self.assertEqual(captured.batch.timestamps.shape[0], n)
        self.assertEqual(captured.batch.signals.shape, (n, N_CHANNELS))
        self.assertEqual(captured.trigger_sample, expected_pre)

        diffs = np.diff(captured.batch.timestamps)
        np.testing.assert_allclose(diffs, np.full(n - 1, period))

        first_global = edge_index - expected_pre
        expected_ts = (
            np.arange(first_global, first_global + n, dtype=np.float64) / self.RATE
        )
        np.testing.assert_allclose(captured.batch.timestamps, expected_ts)

        # Force channels 0-9 encode the global sample index.
        expected_forces = np.broadcast_to(
            np.arange(first_global, first_global + n, dtype=np.float64)[:, None],
            (n, self.N_FORCE),
        )
        np.testing.assert_allclose(
            captured.batch.signals[:, : self.N_FORCE], expected_forces
        )

        self.assertAlmostEqual(
            captured.batch.timestamps[captured.trigger_sample], edge_index * period
        )

    def test_edge_mid_batch_with_full_ring(self) -> None:
        proc = self._make_processor()
        meta = self._make_meta()
        n = 64 + 32
        edge = 64
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, meta, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=4, expected_post=4)

    def test_edge_at_first_sample_of_batch(self) -> None:
        proc = self._make_processor()
        meta = self._make_meta()
        n = 60 + 32
        edge = 60
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, meta, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=4, expected_post=4)

    def test_edge_at_last_sample_of_batch(self) -> None:
        proc = self._make_processor()
        meta = self._make_meta()
        n = 69 + 32
        edge = 69
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, meta, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=4, expected_post=4)

    def test_edge_before_ring_full_emits_truncated_window(self) -> None:
        proc = self._make_processor()
        meta = self._make_meta()
        # Advance past warmup, then clear the ring so the test scenario starts
        # with a fresh ring (simulating a sub-session where the ring was just reset).
        _feed_warmup(proc, meta)
        proc._ring_filled = 0
        proc._ring_pos = 0
        n = 2 + 16
        edge = 2
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, meta, ts, f, aux, batch_size=8)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=2, expected_post=4)

    def test_back_to_back_triggers_share_no_state(self) -> None:
        proc = self._make_processor()
        meta = self._make_meta()
        edge_a = 64
        edge_b = edge_a + 64
        n = edge_b + 16
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge_a] = 5.0
        aux[edge_b] = 5.0
        captures = self._feed(proc, meta, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 2)
        self._assert_contiguous(captures[0], edge_index=edge_a, expected_pre=4, expected_post=4)
        self._assert_contiguous(captures[1], edge_index=edge_b, expected_pre=4, expected_post=4)

    def test_zero_offset_no_preroll(self) -> None:
        config = DextViewConfig(
            sample_rate_hz=self.RATE,
            n_channels=N_CHANNELS,
            window_seconds=self.WINDOW_SECONDS,
            window_offset_seconds=0.0,
            trigger_channel=TRIGGER_CH,
        )
        proc = TriggerWindowProcessor(config)
        meta = StreamMeta(
            channels=_make_channels(),
            config=config,
        )
        edge = 64
        n = edge + self.RATE * 2
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, meta, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=0, expected_post=8)

    def test_trigger_as_last_sample(self) -> None:
        config = DextViewConfig(
            sample_rate_hz=self.RATE,
            n_channels=N_CHANNELS,
            window_seconds=self.WINDOW_SECONDS,
            window_offset_seconds=-(self.WINDOW_SECONDS - 1.0 / self.RATE),
            trigger_channel=TRIGGER_CH,
        )
        proc = TriggerWindowProcessor(config)
        meta = StreamMeta(
            channels=_make_channels(),
            config=config,
        )
        edge = 56
        n = edge + 8
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge] = 5.0
        captures = self._feed(proc, meta, ts, f, aux, batch_size=8)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=7, expected_post=1)


class WindowOffsetConfigTests(unittest.TestCase):
    def test_positive_offset_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DextViewConfig(window_offset_seconds=0.1)

    def test_offset_equal_to_window_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DextViewConfig(
                sample_rate_hz=8,
                window_seconds=1.0,
                window_offset_seconds=-1.0,
            )

    def test_post_trigger_samples_no_offset(self) -> None:
        c = DextViewConfig(sample_rate_hz=8, window_seconds=1.0)
        self.assertEqual(c.total_window_samples, 8)
        self.assertEqual(c.pre_trigger_samples, 0)
        self.assertEqual(c.post_trigger_samples, 8)

    def test_post_trigger_samples_with_offset(self) -> None:
        c = DextViewConfig(
            sample_rate_hz=8,
            window_seconds=1.0,
            window_offset_seconds=-0.5,
        )
        self.assertEqual(c.total_window_samples, 8)
        self.assertEqual(c.pre_trigger_samples, 4)
        self.assertEqual(c.post_trigger_samples, 4)

    def test_post_trigger_samples_minimum_one_sample(self) -> None:
        c = DextViewConfig(
            sample_rate_hz=8,
            window_seconds=1.0,
            window_offset_seconds=-7.0 / 8,
        )
        self.assertEqual(c.pre_trigger_samples, 7)
        self.assertEqual(c.post_trigger_samples, 1)

    def test_pre_trigger_samples_rounding(self) -> None:
        c1 = DextViewConfig(
            sample_rate_hz=8,
            window_seconds=1.0,
            window_offset_seconds=-1.0 / 8,
        )
        self.assertEqual(c1.pre_trigger_samples, 1)
        c2 = DextViewConfig(
            sample_rate_hz=8,
            window_seconds=1.0,
            window_offset_seconds=-0.4 / 8,
        )
        self.assertEqual(c2.pre_trigger_samples, 0)


if __name__ == "__main__":
    unittest.main()
