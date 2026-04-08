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

        self.assertIsNone(processor.process_batch(batch_1))
        captured = processor.process_batch(batch_2)
        self.assertIsNotNone(captured)
        assert captured is not None

        self.assertEqual(captured.finger_forces.shape, (4, 10))
        np.testing.assert_allclose(captured.timestamps, timestamps[2:6])
        np.testing.assert_allclose(captured.finger_ranges, np.full(10, 300.0))

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

        self.assertIsNone(processor.process_batch(batch_1))
        captured = processor.process_batch(batch_2)
        self.assertIsNotNone(captured)
        assert captured is not None

        np.testing.assert_allclose(captured.timestamps, timestamps[3:7])
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

        self.assertIsNone(processor.process_batch(warmup))
        self.assertIsNone(processor.process_batch(pulse))
        captured = processor.process_batch(tail)
        self.assertIsNotNone(captured)
        assert captured is not None

        # Capture starts one sample after the rising edge (index 5).
        np.testing.assert_allclose(captured.timestamps, timestamps[5:9])
        self.assertEqual(captured.finger_forces.shape, (4, 10))


if __name__ == "__main__":
    unittest.main()
