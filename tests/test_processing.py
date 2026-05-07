import unittest

import numpy as np

from quattrocento.config import QuattrocentoConfig
from quattrocento.models import DataBatch
from quattrocento.processing import TriggerWindowProcessor, aggregate_finger_forces


class TriggerWindowProcessorTests(unittest.TestCase):
    def test_capture_collects_next_window_after_rising_edge(self) -> None:
        config = QuattrocentoConfig(sample_rate_hz=4, window_seconds=1.0)
        processor = TriggerWindowProcessor(config)

        force_rows = np.array(
            [[row * 100.0 + sensor for sensor in range(10)] for row in range(7)],
            dtype=np.float64,
        )
        timestamps = np.arange(7, dtype=np.float64) / config.sample_rate_hz

        batch_1 = DataBatch(
            timestamps=timestamps[:3],
            forces=force_rows[:3],
            aux_in=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )
        batch_2 = DataBatch(
            timestamps=timestamps[3:],
            forces=force_rows[3:],
            aux_in=np.zeros(4, dtype=np.float64),
        )

        self.assertEqual(len(processor.process_batch(batch_1)), 0)
        captured_list = processor.process_batch(batch_2)
        self.assertEqual(len(captured_list), 1)
        captured = captured_list[0]

        self.assertEqual(captured.finger_forces.shape, (4, 10))
        np.testing.assert_allclose(captured.timestamps, timestamps[1:5])
        np.testing.assert_allclose(captured.finger_ranges, np.full(10, 300.0))

    def test_multiple_captures_in_single_batch(self) -> None:
        # 10 Hz rate, 0.5s window = 5 samples.
        config = QuattrocentoConfig(sample_rate_hz=10, window_seconds=0.5)
        processor = TriggerWindowProcessor(config)

        # 20 samples = 2 seconds.
        n = 20
        timestamps = np.arange(n, dtype=np.float64) / 10.0
        forces = np.zeros((n, 10))
        # Trigger pulses at index 2 and index 10.
        # Capture 1: starts at 2, ends at 2+5=7.
        # Capture 2: starts at 10, ends at 10+5=15.
        aux = np.zeros(n)
        aux[2] = 5.0
        aux[10] = 5.0

        batch = DataBatch(timestamps=timestamps, forces=forces, aux_in=aux)
        captured_list = processor.process_batch(batch)

        self.assertEqual(len(captured_list), 2)
        self.assertEqual(captured_list[0].trigger_index, 0)
        np.testing.assert_allclose(captured_list[0].timestamps, timestamps[2:7])
        self.assertEqual(captured_list[1].trigger_index, 0)
        np.testing.assert_allclose(captured_list[1].timestamps, timestamps[10:15])

    def test_aggregate_finger_forces_maps_each_finger_to_its_sensor(self) -> None:
        sensor_forces = np.array(
            [
                [1.0, 4.0, 7.0, 10.0],
                [2.0, 5.0, 8.0, 11.0],
            ],
            dtype=np.float64,
        )
        finger_map = {"F1": 0, "F2": 1, "F3": 2, "F4": 3}

        finger_forces, labels = aggregate_finger_forces(sensor_forces, finger_map)

        self.assertEqual(labels, ("F1", "F2", "F3", "F4"))
        np.testing.assert_allclose(finger_forces, sensor_forces)

    def test_capture_detects_trigger_with_high_aux_baseline(self) -> None:
        config = QuattrocentoConfig(sample_rate_hz=4, window_seconds=1.0)
        processor = TriggerWindowProcessor(config)

        force_rows = np.array(
            [[row * 10.0 + sensor for sensor in range(10)] for row in range(8)],
            dtype=np.float64,
        )
        timestamps = np.arange(8, dtype=np.float64) / config.sample_rate_hz

        # High DC offset with one short pulse above baseline.
        batch_1 = DataBatch(
            timestamps=timestamps[:4],
            forces=force_rows[:4],
            aux_in=np.array([8000.0, 8000.0, 9000.0, 8000.0], dtype=np.float64),
        )
        batch_2 = DataBatch(
            timestamps=timestamps[4:],
            forces=force_rows[4:],
            aux_in=np.full(4, 8000.0, dtype=np.float64),
        )

        self.assertEqual(len(processor.process_batch(batch_1)), 0)
        captured_list = processor.process_batch(batch_2)
        self.assertEqual(len(captured_list), 1)
        captured = captured_list[0]

        np.testing.assert_allclose(captured.timestamps, timestamps[2:6])
        self.assertEqual(captured.finger_forces.shape, (4, 10))


    def test_single_sample_pulse_triggers_capture(self) -> None:
        config = QuattrocentoConfig(sample_rate_hz=4, window_seconds=1.0)
        processor = TriggerWindowProcessor(config)

        force_rows = np.array(
            [[row * 10.0 + sensor for sensor in range(10)] for row in range(12)],
            dtype=np.float64,
        )
        timestamps = np.arange(12, dtype=np.float64) / config.sample_rate_hz

        # Warm up baseline with steady signal.
        warmup = DataBatch(
            timestamps=timestamps[:4],
            forces=force_rows[:4],
            aux_in=np.full(4, 5000.0, dtype=np.float64),
        )
        # Single-sample pulse at index 4, back to baseline at index 5.
        pulse = DataBatch(
            timestamps=timestamps[4:6],
            forces=force_rows[4:6],
            aux_in=np.array([6000.0, 5000.0], dtype=np.float64),
        )
        # Fill remaining capture window.
        tail = DataBatch(
            timestamps=timestamps[6:10],
            forces=force_rows[6:10],
            aux_in=np.full(4, 5000.0, dtype=np.float64),
        )

        self.assertEqual(len(processor.process_batch(warmup)), 0)
        self.assertEqual(len(processor.process_batch(pulse)), 0)
        captured_list = processor.process_batch(tail)
        self.assertEqual(len(captured_list), 1)
        captured = captured_list[0]

        # Capture starts at the rising-edge sample (index 4).
        np.testing.assert_allclose(captured.timestamps, timestamps[4:8])
        self.assertEqual(captured.finger_forces.shape, (4, 10))


class WindowOffsetTests(unittest.TestCase):
    """Tests for negative window-offset (pre-trigger) capture.

    All tests build a contiguous synthetic stream where forces[i, k] = i and
    timestamps[i] = i / rate, then feed it as fixed-size batches. After a
    trigger fires, the captured window must be a contiguous slice of that
    stream — no drops, no duplicates, no gap at the seam between pre-roll
    and post-trigger.
    """

    RATE = 8
    POST_SECONDS = 1.0  # post = 8 samples
    OFFSET_SECONDS = -0.5  # pre = 4 samples
    SENSORS = 10  # default finger_sensor_map size

    def _make_processor(self) -> TriggerWindowProcessor:
        config = QuattrocentoConfig(
            sample_rate_hz=self.RATE,
            window_seconds=self.POST_SECONDS,
            window_offset_seconds=self.OFFSET_SECONDS,
        )
        return TriggerWindowProcessor(config)

    def _stream(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Build (timestamps, forces) where each row encodes its global index."""
        timestamps = np.arange(n, dtype=np.float64) / self.RATE
        forces = np.broadcast_to(
            np.arange(n, dtype=np.float64)[:, None], (n, self.SENSORS)
        ).copy()
        return timestamps, forces

    def _feed(
        self,
        processor: TriggerWindowProcessor,
        timestamps: np.ndarray,
        forces: np.ndarray,
        aux: np.ndarray,
        batch_size: int,
    ) -> list:
        """Feed the stream to the processor in fixed-size batches."""
        captures = []
        n = timestamps.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            out = processor.process_batch(
                DataBatch(
                    timestamps=timestamps[start:end],
                    forces=forces[start:end],
                    aux_in=aux[start:end],
                )
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
        """The captured window must be a clean slice of the source stream:
        - timestamps strictly increase by exactly 1/RATE (no drops, no dupes),
        - forces[i, k] == global_index_of_sample_i (proves no reordering),
        - trigger_index points at the edge sample,
        - lengths match expected_pre + expected_post.
        """
        period = 1.0 / self.RATE
        n = expected_pre + expected_post
        self.assertEqual(captured.timestamps.shape[0], n)
        self.assertEqual(captured.finger_forces.shape, (n, self.SENSORS))
        self.assertEqual(captured.trigger_index, expected_pre)

        # Contiguity: every step is exactly one sample period.
        diffs = np.diff(captured.timestamps)
        np.testing.assert_allclose(diffs, np.full(n - 1, period))

        # The capture covers global indices [edge - expected_pre, edge + expected_post).
        first_global = edge_index - expected_pre
        expected_ts = (
            np.arange(first_global, first_global + n, dtype=np.float64) / self.RATE
        )
        np.testing.assert_allclose(captured.timestamps, expected_ts)

        # Forces encode the global index — confirms ring → buffer → emit
        # preserved sample identity (no swap, no overwrite).
        expected_forces = np.broadcast_to(
            np.arange(first_global, first_global + n, dtype=np.float64)[:, None],
            (n, self.SENSORS),
        )
        np.testing.assert_allclose(captured.finger_forces, expected_forces)

        # t=0 lands on the edge sample.
        self.assertAlmostEqual(
            captured.timestamps[captured.trigger_index], edge_index * period
        )

    # ---- individual scenarios -----------------------------------------------

    def test_edge_mid_batch_with_full_ring(self) -> None:
        """Edge fires mid-batch after enough quiet samples to fill the ring."""
        proc = self._make_processor()
        # 64 quiet samples (ring needs only 4) then AUX steps high and stays high.
        n = 64 + 32
        edge = 64
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=4, expected_post=8)

    def test_edge_at_first_sample_of_batch(self) -> None:
        """Pre-roll comes entirely from the ring; none from the current batch."""
        proc = self._make_processor()
        # Edge lands on a batch boundary (index 60, batch_size 10).
        n = 60 + 32
        edge = 60
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=4, expected_post=8)

    def test_edge_at_last_sample_of_batch(self) -> None:
        """Pre-roll spans ring + nearly-whole current batch; post comes from later batches."""
        proc = self._make_processor()
        # batch_size=10; place edge at index 69 (= last sample of the 7th batch).
        n = 69 + 32
        edge = 69
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 1)
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=4, expected_post=8)

    def test_edge_before_ring_full_emits_truncated_window(self) -> None:
        """Trigger fires before pre_samples have accumulated — pre-roll is short."""
        proc = self._make_processor()
        # Edge at global index 2 (only 2 quiet samples seen; ring wants 4).
        n = 2 + 16
        edge = 2
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = self._feed(proc, ts, f, aux, batch_size=8)
        self.assertEqual(len(captures), 1)
        # Only 2 pre-roll samples available; post is the usual 8.
        self._assert_contiguous(captures[0], edge_index=edge, expected_pre=2, expected_post=8)

    def test_back_to_back_triggers_share_no_state(self) -> None:
        """A second trigger must see a fresh, contiguous pre-roll — proves the
        ring keeps updating during and after capture."""
        proc = self._make_processor()
        edge_a = 64
        # Spacing: capture A's window (post=8) plus enough rest for the
        # adaptive noise estimate to decay below the second pulse amplitude.
        edge_b = edge_a + 64
        n = edge_b + 16
        ts, f = self._stream(n)
        aux = np.zeros(n)
        # Single-sample pulses at each edge — minimal disturbance to the
        # adaptive baseline/noise estimators between events.
        aux[edge_a] = 5.0
        aux[edge_b] = 5.0
        captures = self._feed(proc, ts, f, aux, batch_size=10)
        self.assertEqual(len(captures), 2)
        self._assert_contiguous(captures[0], edge_index=edge_a, expected_pre=4, expected_post=8)
        self._assert_contiguous(captures[1], edge_index=edge_b, expected_pre=4, expected_post=8)

    def test_zero_offset_matches_no_preroll(self) -> None:
        """Default (offset=0) behavior: zero pre-roll, trigger_index == 0."""
        config = QuattrocentoConfig(
            sample_rate_hz=self.RATE,
            window_seconds=self.POST_SECONDS,
            window_offset_seconds=0.0,
        )
        proc = TriggerWindowProcessor(config)
        n = 64 + 16
        edge = 64
        ts, f = self._stream(n)
        aux = np.zeros(n)
        aux[edge:] = 5.0
        captures = []
        for s in range(0, n, 10):
            e = min(s + 10, n)
            out = proc.process_batch(
                DataBatch(timestamps=ts[s:e], forces=f[s:e], aux_in=aux[s:e])
            )
            captures.extend(out)
        self.assertEqual(len(captures), 1)
        captured = captures[0]
        self.assertEqual(captured.trigger_index, 0)
        period = 1.0 / self.RATE
        np.testing.assert_allclose(np.diff(captured.timestamps), np.full(7, period))
        np.testing.assert_allclose(
            captured.timestamps,
            np.arange(edge, edge + 8, dtype=np.float64) / self.RATE,
        )

    def test_pre_trigger_samples_rounding(self) -> None:
        """Sub-sample offsets round to the nearest whole sample."""
        # 1/RATE exactly → 1 sample.
        c1 = QuattrocentoConfig(
            sample_rate_hz=self.RATE,
            window_seconds=1.0,
            window_offset_seconds=-1.0 / self.RATE,
        )
        self.assertEqual(c1.pre_trigger_samples, 1)
        # Slightly less than half a sample → rounds to 0.
        c2 = QuattrocentoConfig(
            sample_rate_hz=self.RATE,
            window_seconds=1.0,
            window_offset_seconds=-0.4 / self.RATE,
        )
        self.assertEqual(c2.pre_trigger_samples, 0)


class WindowOffsetConfigTests(unittest.TestCase):
    def test_positive_offset_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QuattrocentoConfig(window_offset_seconds=0.1)


if __name__ == "__main__":
    unittest.main()
