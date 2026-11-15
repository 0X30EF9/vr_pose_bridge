from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop_hand_ik import HandIkError, LatestHandIkWorker


class BlockingSidecar:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests = []
        self.closed = False

    def solve_replay_frame(self, frame):
        self.requests.append(frame["receive_time"])
        self.entered.set()
        if not self.release.wait(2.0):
            raise RuntimeError("test sidecar was not released")
        return SimpleNamespace(sequence=frame["receive_time"])

    def abort(self):
        self.release.set()

    def close(self):
        self.closed = True


class AsyncHandIkTest(unittest.TestCase):
    def wait_result(self, worker, sequence):
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            worker.check_health()
            completed = worker.poll()
            if completed is not None and completed.result.sequence == sequence:
                return completed
            time.sleep(0.001)
        self.fail(f"no asynchronous result for {sequence}")

    def test_blocked_sdk_does_not_block_submit_and_pending_is_latest_only(self):
        sidecar = BlockingSidecar()
        worker = LatestHandIkWorker(sidecar, rate_hz=1000)
        self.addCleanup(worker.close)
        worker.submit({"receive_time": 1}, 10.0)
        self.assertTrue(sidecar.entered.wait(1.0))
        for index in range(2, 101):
            self.assertTrue(worker.submit({"receive_time": index}, 10.0 + index / 1000.0))
        self.assertIsNone(worker.poll())
        self.assertEqual(sidecar.requests, [1])
        sidecar.release.set()
        completed = self.wait_result(worker, 100)
        self.assertEqual(sidecar.requests, [1, 100])
        self.assertEqual(completed.submission, 100)
        self.assertAlmostEqual(completed.received_at, 10.1)
        self.assertIsNone(worker.poll())

    def test_duplicate_and_out_of_order_inputs_are_not_resubmitted(self):
        sidecar = BlockingSidecar()
        worker = LatestHandIkWorker(sidecar)
        self.addCleanup(sidecar.release.set)
        self.addCleanup(worker.close)
        self.assertTrue(worker.submit({"receive_time": 1}, 10.0))
        self.assertFalse(worker.submit({"receive_time": 1}, 10.1))
        self.assertFalse(worker.submit({"receive_time": 2}, 9.9))
        self.assertFalse(worker.submit({"receive_time": 2}, 10.0))
        self.assertFalse(worker.submit({"receive_time": 0}, 10.2))

    def test_invalidating_session_discards_its_inflight_result(self):
        sidecar = BlockingSidecar()
        worker = LatestHandIkWorker(sidecar, rate_hz=1000)
        self.addCleanup(worker.close)
        worker.submit({"receive_time": 1}, 10.0)
        self.assertTrue(sidecar.entered.wait(1.0))
        worker.invalidate()
        worker.submit({"receive_time": 2}, 10.2)
        sidecar.release.set()
        completed = self.wait_result(worker, 2)
        self.assertEqual(completed.submission, 2)
        self.assertIsNone(worker.poll())

    def test_worker_failure_is_explicit_and_cannot_reuse_old_result(self):
        sidecar = BlockingSidecar()
        def fail(frame):
            raise OSError("fake SDK failure")
        sidecar.solve_replay_frame = fail
        worker = LatestHandIkWorker(sidecar)
        self.addCleanup(worker.close)
        worker.submit({"receive_time": 1}, 10.0)
        worker._thread.join(timeout=1.0)
        with self.assertRaisesRegex(HandIkError, "fake SDK failure"):
            worker.poll()
        with self.assertRaises(HandIkError):
            worker.submit({"receive_time": 2}, 10.1)

    def test_shutdown_interrupts_a_blocked_sidecar_and_clears_results(self):
        sidecar = BlockingSidecar()
        worker = LatestHandIkWorker(sidecar)
        worker.submit({"receive_time": 1}, 10.0)
        self.assertTrue(sidecar.entered.wait(1.0))
        worker.close()
        self.assertFalse(worker._thread.is_alive())
        self.assertTrue(sidecar.closed)
        self.assertIsNone(worker.poll())
        with self.assertRaises(HandIkError):
            worker.submit({"receive_time": 2}, 10.1)


if __name__ == "__main__":
    unittest.main()
