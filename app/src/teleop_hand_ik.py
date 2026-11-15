from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, TextIO

from teleop_joint_schema import HAND_JOINT_ORDER
from teleop_mocap_protocol import MocapFrame

_HAND_SOURCE_SUFFIX: Mapping[str, str] = {
    "hand": "hand",
    "thumb_near": "thumb_near",
    "thumb_middle": "thumb_mid",
    "thumb_far": "thumb_far",
    "index_near": "index_near",
    "index_middle": "index_mid",
    "index_far": "index_far",
    "middle_near": "middle_near",
    "middle_middle": "middle_mid",
    "middle_far": "middle_far",
    "ring_near": "ring_near",
    "ring_middle": "ring_mid",
    "ring_far": "ring_far",
    "little_near": "pinky_near",
    "little_middle": "pinky_mid",
    "little_far": "pinky_far",
}


class HandIkError(RuntimeError):
    pass


@dataclass(frozen=True)
class HandIkSidecarConfig:
    sdk_root: Path
    sdk_config: Path
    library: Path
    timeout_s: float = 2.0
    diagnostic_csv: Path | None = None
    python_executable: Path = Path(sys.executable)
    worker_path: Path = Path(__file__).resolve().parent / "revo2_hand_ik_worker.py"


@dataclass(frozen=True)
class HandIkFrameResult:
    sequence: int
    timestamp_ms: int
    joint_positions: tuple[float, ...]
    left_calibrated: bool
    right_calibrated: bool
    left_solve_time_ms: float
    right_solve_time_ms: float
    left_calibration_status: str
    right_calibration_status: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "joint_unit": "rad",
            "joint_order": list(HAND_JOINT_ORDER),
            "joint_positions": list(self.joint_positions),
            "left_calibrated": self.left_calibrated,
            "right_calibrated": self.right_calibrated,
            "left_solve_time_ms": self.left_solve_time_ms,
            "right_solve_time_ms": self.right_solve_time_ms,
            "left_calibration_status": self.left_calibration_status,
            "right_calibration_status": self.right_calibration_status,
            "transport": "local_subprocess_pipe",
            "published": False,
        }


def _hand_input(frame: MocapFrame, side: str) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    points: dict[str, tuple[float, ...]] = {}
    quaternions: dict[str, tuple[float, ...]] = {}
    for sdk_name, source_suffix in _HAND_SOURCE_SUFFIX.items():
        pose = frame.joint(f"{side}_{source_suffix}")
        if not pose.tracked:
            raise HandIkError(f"未跟踪手部节点: {side}_{source_suffix}")
        if not all(math.isfinite(value) for value in pose.position + pose.quaternion_xyzw):
            raise HandIkError(f"手部节点包含 NaN/Inf: {side}_{source_suffix}")
        qx, qy, qz, qw = pose.quaternion_xyzw
        points[sdk_name] = tuple(float(value) for value in pose.position)
        quaternions[sdk_name] = (float(qw), float(qx), float(qy), float(qz))
    return points, quaternions


def hand_ik_request(frame: MocapFrame, sequence: int) -> dict[str, Any]:
    left_points, left_quats = _hand_input(frame, "left")
    right_points, right_quats = _hand_input(frame, "right")
    return {
        "type": "solve",
        "sequence": int(sequence),
        "timestamp_ms": int(frame.timestamp_us // 1000),
        "left_points": left_points,
        "right_points": right_points,
        "left_quaternions_wxyz": left_quats,
        "right_quaternions_wxyz": right_quats,
    }


def mocap_frame_from_replay(frame: Mapping[str, Any]) -> MocapFrame:
    mocap_frame = frame.get("mocap_frame")
    if not isinstance(mocap_frame, MocapFrame):
        raise HandIkError("手部 IK 需要包含规范 mocap_frame 的 64 节点输入")
    return mocap_frame


def _debug_value(result: Mapping[str, Any], side: str, key: str, default: Any) -> Any:
    hand = result.get(side, {})
    debug = hand.get("debug", {}) if isinstance(hand, Mapping) else {}
    return debug.get(key, default) if isinstance(debug, Mapping) else default


def parse_hand_ik_result(payload: Mapping[str, Any], expected_sequence: int) -> HandIkFrameResult:
    if payload.get("type") == "error":
        raise HandIkError(str(payload.get("error", "Revo2 worker solve failed")))
    if payload.get("type") in {"fatal", "closed"}:
        raise HandIkError(str(payload.get("error", f"Revo2 worker unexpected response: {payload.get('type')}")))
    if payload.get("type") != "result" or int(payload.get("sequence", -1)) != expected_sequence:
        raise HandIkError(f"Revo2 worker 响应序号不匹配: {payload}")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise HandIkError("Revo2 worker 结果不是 object")
    order = tuple(str(value) for value in result.get("joint_order", ()))
    if order != HAND_JOINT_ORDER:
        raise HandIkError(f"Revo2 12 路输出顺序不匹配: {order}")
    positions = tuple(float(value) for value in result.get("joint_positions", ()))
    if len(positions) != len(HAND_JOINT_ORDER) or not all(math.isfinite(value) for value in positions):
        raise HandIkError("Revo2 输出必须是 12 维有限弧度角")
    return HandIkFrameResult(
        sequence=expected_sequence,
        timestamp_ms=int(result.get("timestamp_ms", 0)),
        joint_positions=positions,
        left_calibrated=bool(_debug_value(result, "left", "calibrated", False)),
        right_calibrated=bool(_debug_value(result, "right", "calibrated", False)),
        left_solve_time_ms=float(result.get("left_solve_time_ms", 0.0)),
        right_solve_time_ms=float(result.get("right_solve_time_ms", 0.0)),
        left_calibration_status=str(_debug_value(result, "left", "calibration_status", "unknown")),
        right_calibration_status=str(_debug_value(result, "right", "calibration_status", "unknown")),
    )


class HandIkDiagnosticWriter:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self.path.open("w", encoding="utf-8", newline="")
        fields = [
            "sequence", "timestamp_ms", "left_calibrated", "right_calibrated",
            "left_solve_time_ms", "right_solve_time_ms",
            "left_calibration_status", "right_calibration_status",
            *HAND_JOINT_ORDER,
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()

    def write(self, result: HandIkFrameResult) -> None:
        row = {
            "sequence": result.sequence,
            "timestamp_ms": result.timestamp_ms,
            "left_calibrated": int(result.left_calibrated),
            "right_calibrated": int(result.right_calibrated),
            "left_solve_time_ms": result.left_solve_time_ms,
            "right_solve_time_ms": result.right_solve_time_ms,
            "left_calibration_status": result.left_calibration_status,
            "right_calibration_status": result.right_calibration_status,
        }
        row.update({name: value for name, value in zip(HAND_JOINT_ORDER, result.joint_positions)})
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class Revo2HandIkSidecar:
    """仅通过本机匿名管道调用 Revo2 SDK；该类不包含任何网络发布接口。"""

    def __init__(self, config: HandIkSidecarConfig):
        if not math.isfinite(config.timeout_s) or config.timeout_s <= 0:
            raise ValueError("hand IK timeout_s 必须 > 0")
        self.config = config
        self._sequence = 0
        self.latest_result: HandIkFrameResult | None = None
        self._diagnostic = HandIkDiagnosticWriter(config.diagnostic_csv) if config.diagnostic_csv else None
        self._process: subprocess.Popen[str] | None = None
        self._response: TextIO | None = None
        try:
            self._start()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "Revo2HandIkSidecar":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _start(self) -> None:
        for path, label in (
            (self.config.sdk_root, "sdk_root"),
            (self.config.sdk_config, "sdk_config"),
            (self.config.library, "library"),
            (self.config.python_executable, "python_executable"),
            (self.config.worker_path, "worker_path"),
        ):
            if not path.expanduser().resolve().exists():
                raise FileNotFoundError(f"hand IK {label} 不存在: {path}")
        read_fd, write_fd = os.pipe()
        env = os.environ.copy()
        lib_dir = str(self.config.sdk_root.expanduser().resolve() / "lib")
        env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        command = [
            str(self.config.python_executable.expanduser().resolve()),
            str(self.config.worker_path.expanduser().resolve()),
            "--sdk-root", str(self.config.sdk_root.expanduser().resolve()),
            "--sdk-config", str(self.config.sdk_config.expanduser().resolve()),
            "--library", str(self.config.library.expanduser().resolve()),
            "--response-fd", str(write_fd),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self.config.sdk_root.expanduser().resolve(),
                env=env,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                pass_fds=(write_fd,),
            )
        except Exception:
            os.close(read_fd)
            raise
        finally:
            os.close(write_fd)
        self._response = os.fdopen(read_fd, "r", encoding="utf-8", buffering=1)
        ready = self._read_response()
        if ready.get("type") != "ready":
            self.close()
            raise HandIkError(f"Revo2 worker 启动失败: {ready}")

    def _read_response(self) -> dict[str, Any]:
        if self._response is None:
            raise HandIkError("Revo2 worker 响应管道未初始化")
        ready, _, _ = select.select([self._response], [], [], self.config.timeout_s)
        if not ready:
            raise HandIkError(f"Revo2 worker 响应超时: {self.config.timeout_s}s")
        line = self._response.readline()
        if not line:
            code = None if self._process is None else self._process.poll()
            raise HandIkError(f"Revo2 worker 已退出: returncode={code}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise HandIkError("Revo2 worker 响应必须是 JSON object")
        return payload

    def _write_request(self, payload: Mapping[str, Any]) -> None:
        if self._process is None or self._process.stdin is None or self._process.poll() is not None:
            raise HandIkError("Revo2 worker 不可用")
        self._process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def solve_mocap_frame(self, frame: MocapFrame) -> HandIkFrameResult:
        sequence = self._sequence
        self._sequence += 1
        self._write_request(hand_ik_request(frame, sequence))
        result = parse_hand_ik_result(self._read_response(), sequence)
        self.latest_result = result
        if self._diagnostic is not None:
            self._diagnostic.write(result)
        return result

    def solve_replay_frame(self, frame: Mapping[str, Any]) -> HandIkFrameResult:
        return self.solve_mocap_frame(mocap_frame_from_replay(frame))

    def reset(self) -> None:
        self._write_request({"type": "reset"})
        response = self._read_response()
        if response.get("type") != "reset":
            raise HandIkError(f"Revo2 worker reset 失败: {response}")
        self._sequence = 0
        self.latest_result = None

    def abort(self) -> None:
        """Interrupt a blocked SDK subprocess when the owning worker shuts down."""
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    def close(self) -> None:
        if self._process is not None:
            process = self._process
            try:
                if self._process.poll() is None:
                    self._write_request({"type": "close"})
                    try:
                        self._read_response()
                    except Exception:
                        pass
                if self._process.stdin is not None:
                    self._process.stdin.close()
                self._process.wait(timeout=max(0.2, self.config.timeout_s))
            except Exception:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=1.0)
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            self._process = None
        if self._response is not None:
            self._response.close()
            self._response = None
        if self._diagnostic is not None:
            self._diagnostic.close()
            self._diagnostic = None


@dataclass(frozen=True)
class AsyncHandIkResult:
    submission: int
    received_at: float
    result: HandIkFrameResult


class LatestHandIkWorker:
    """One running solve and one replaceable pending frame; never publishes.

    The worker owns the sidecar. Results retain the original local monotonic
    input receipt time and can be consumed only once. Invalidating a session
    also discards any result still being computed for that session.
    """

    def __init__(self, sidecar, rate_hz: float = 20.0):
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("hand control rate must be positive and finite")
        self.sidecar = sidecar
        self.rate_hz = rate_hz
        self._condition = threading.Condition()
        self._pending = None
        self._completed = None
        self._error = None
        self._stopped = False
        self._generation = 0
        self._submission = 0
        self._last_received = None
        self._last_key = None
        self._thread = threading.Thread(target=self._run, name="hand-ik-latest", daemon=True)
        self._thread.start()

    def submit(self, frame: Mapping[str, Any], received_at: float) -> bool:
        if not math.isfinite(received_at):
            raise HandIkError("hand input receipt time must be finite")
        mocap = frame.get("mocap_frame")
        key = getattr(mocap, "timestamp_us", frame.get("packet_timestamp_us", frame.get("receive_time")))
        if key is not None:
            try:
                key = float(key)
            except (TypeError, ValueError) as exc:
                raise HandIkError("hand frame timestamp must be numeric") from exc
            if not math.isfinite(key):
                raise HandIkError("hand frame timestamp must be finite")
        with self._condition:
            if self._stopped:
                raise HandIkError("hand worker is closed")
            if self._error is not None:
                raise HandIkError(f"hand worker failed: {self._error}")
            if self._last_received is not None and received_at <= self._last_received:
                return False
            if key is not None and self._last_key is not None and key <= self._last_key:
                return False
            self._submission += 1
            self._last_received, self._last_key = received_at, key
            self._pending = (self._generation, self._submission, dict(frame), received_at)
            self._condition.notify()
            return True

    def poll(self) -> AsyncHandIkResult | None:
        with self._condition:
            if self._error is not None:
                raise HandIkError(f"hand worker failed: {self._error}")
            result, self._completed = self._completed, None
            return result

    def check_health(self) -> None:
        """Inspect a latched SDK failure without consuming a completed result."""
        with self._condition:
            if self._error is not None:
                raise HandIkError(f"hand worker failed: {self._error}")
            if self._stopped:
                raise HandIkError("hand worker is closed")

    def invalidate(self) -> None:
        with self._condition:
            self._generation += 1
            self._pending = self._completed = None

    def _run(self) -> None:
        next_solve = 0.0
        try:
            while True:
                with self._condition:
                    while not self._stopped:
                        delay = max(0.0, next_solve - time.monotonic())
                        if self._pending is not None and delay <= 0:
                            break
                        self._condition.wait(delay if self._pending is not None else None)
                    if self._stopped:
                        return
                    generation, submission, frame, received_at = self._pending
                    self._pending = None
                next_solve = time.monotonic() + 1.0 / self.rate_hz
                result = self.sidecar.solve_replay_frame(frame)
                with self._condition:
                    if not self._stopped and generation == self._generation:
                        self._completed = AsyncHandIkResult(submission, received_at, result)
        except Exception as exc:
            with self._condition:
                if not self._stopped:
                    self._error = exc
                self._pending = self._completed = None
        finally:
            try:
                self.sidecar.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._condition:
            self._stopped = True
            self._pending = self._completed = None
            self._condition.notify_all()
        self._thread.join(timeout=0.2)
        if self._thread.is_alive():
            abort = getattr(self.sidecar, "abort", None)
            if abort is not None:
                abort()
            self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise HandIkError("hand worker did not stop within the shutdown deadline")


def hand_ik_config_from_args(args: argparse.Namespace) -> HandIkSidecarConfig:
    diagnostic_csv = Path(args.hand_ik_diagnostic_csv) if args.hand_ik_diagnostic_csv else None
    return HandIkSidecarConfig(
        sdk_root=Path(args.hand_ik_sdk_root),
        sdk_config=Path(args.hand_ik_sdk_config),
        library=Path(args.hand_ik_library),
        timeout_s=float(args.hand_ik_timeout),
        diagnostic_csv=diagnostic_csv,
    )


def add_hand_ik_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hand-ik-compute", choices=("true", "false"), default="false")
    parser.add_argument("--hand-ik-sdk-root", default="")
    parser.add_argument("--hand-ik-sdk-config", default="")
    parser.add_argument("--hand-ik-library", default="")
    parser.add_argument("--hand-ik-timeout", type=float, default=2.0)
    parser.add_argument("--hand-ik-diagnostic-csv", default="")


def hand_ik_enabled(args: argparse.Namespace) -> bool:
    return str(args.hand_ik_compute).strip().lower() == "true"
