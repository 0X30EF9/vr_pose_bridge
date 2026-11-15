"""ctypes binding for librevo2_ik_sdk.so.

The binding deliberately uses only the Python standard library.  Pinocchio and
all C++ code stay inside the pre-built shared library, so the target runtime
needs Python 3.10+ but no ROS, compiler, pybind11 or eigenpy.
"""
from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

MAX_OUTPUT_JOINTS = 16
MAX_JOINT_NAME_LEN = 64
DEBUG_POINT_COUNT = 15
DEBUG_FRAME_NAME_LEN = 64
DEBUG_LONG_FINGER_COUNT = 4
DEBUG_CALIBRATION_STATUS_LEN = 64
DEBUG_LONG_FINGER_NAMES = ("index", "middle", "ring", "little")

DEBUG_POINT_NAMES = (
    "thumb_near", "thumb_middle", "thumb_far",
    "index_near", "index_middle", "index_far",
    "middle_near", "middle_middle", "middle_far",
    "ring_near", "ring_middle", "ring_far",
    "little_near", "little_middle", "little_far",
)


class Revo2IkError(RuntimeError):
    """Native SDK reported an initialization or solve failure."""


class Point3(ct.Structure):
    _fields_ = [
        ("x", ct.c_double),
        ("y", ct.c_double),
        ("z", ct.c_double),
        ("valid", ct.c_int),
    ]


class Quaternion(ct.Structure):
    _fields_ = [
        ("w", ct.c_double),
        ("x", ct.c_double),
        ("y", ct.c_double),
        ("z", ct.c_double),
        ("valid", ct.c_int),
    ]


class HandMocapInput(ct.Structure):
    # Must remain byte-for-byte aligned with Revo2HandMocapInput in the public
    # C header.  The default C++ long-finger retarget uses palm-relative
    # positions; thumb flex additionally consumes near/far quaternions when
    # they are available.
    _fields_ = [
        ("hand", Point3),
        ("hand_quat", Quaternion),
        ("thumb_near", Point3),
        ("thumb_middle", Point3),
        ("thumb_far", Point3),
        ("thumb_near_quat", Quaternion),
        ("thumb_middle_quat", Quaternion),
        ("thumb_far_quat", Quaternion),
        ("index_near", Point3),
        ("index_middle", Point3),
        ("index_far", Point3),
        ("index_near_quat", Quaternion),
        ("index_middle_quat", Quaternion),
        ("index_far_quat", Quaternion),
        ("middle_near", Point3),
        ("middle_middle", Point3),
        ("middle_far", Point3),
        ("middle_near_quat", Quaternion),
        ("middle_middle_quat", Quaternion),
        ("middle_far_quat", Quaternion),
        ("ring_near", Point3),
        ("ring_middle", Point3),
        ("ring_far", Point3),
        ("ring_near_quat", Quaternion),
        ("ring_middle_quat", Quaternion),
        ("ring_far_quat", Quaternion),
        ("little_near", Point3),
        ("little_middle", Point3),
        ("little_far", Point3),
        ("little_near_quat", Quaternion),
        ("little_middle_quat", Quaternion),
        ("little_far_quat", Quaternion),
        ("sequence", ct.c_uint64),
        ("timestamp_ms", ct.c_uint32),
    ]


class IkOutput(ct.Structure):
    _fields_ = [
        ("joint_count", ct.c_int),
        ("joint_names", (ct.c_char * MAX_JOINT_NAME_LEN) * MAX_OUTPUT_JOINTS),
        ("position_norm_0_100", ct.c_double * MAX_OUTPUT_JOINTS),
        # Retained in the C ABI for compatibility. It has the same 0..100 values.
        ("position_norm_1_100", ct.c_double * MAX_OUTPUT_JOINTS),
        ("position_rad", ct.c_double * MAX_OUTPUT_JOINTS),
        ("solve_time_ms", ct.c_double),
    ]


class LongFingerDebug(ct.Structure):
    _fields_ = [
        ("target_middle", Point3),
        ("target_far", Point3),
        ("robot_middle", Point3),
        ("robot_far", Point3),
        ("valid", ct.c_int),
    ]


class DebugFrame(ct.Structure):
    _fields_ = [
        ("sequence", ct.c_uint64),
        ("timestamp_ms", ct.c_uint32),
        ("calibrated", ct.c_int),
        ("point_count", ct.c_int),
        ("frame_id", ct.c_char * DEBUG_FRAME_NAME_LEN),
        ("mapped_input_points", Point3 * DEBUG_POINT_COUNT),
        ("thumb_target_middle", Point3),
        ("thumb_target_far", Point3),
        ("thumb_robot_middle", Point3),
        ("thumb_robot_far", Point3),
        ("long_finger_tasks", LongFingerDebug * DEBUG_LONG_FINGER_COUNT),
        ("calibration_stage", ct.c_int),
        ("calibration_samples", ct.c_int),
        ("calibration_required", ct.c_int),
        ("calibration_warmup_remaining", ct.c_int),
        ("calibration_rejected", ct.c_int),
        ("calibration_quality_failures", ct.c_int),
        ("calibration_point_rms_m", ct.c_double),
        ("calibration_length_cv", ct.c_double),
        ("calibration_status", ct.c_char * DEBUG_CALIBRATION_STATUS_LEN),
    ]


def _decode_c_string(raw: Any) -> str:
    return bytes(raw).split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _default_library_candidates() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("REVO2_IK_SDK_LIBRARY")
    if env:
        candidates.append(Path(env).expanduser())

    here = Path(__file__).resolve()
    # Runtime package layout: <runtime>/python/revo2_hand_ik/native.py
    candidates.append(here.parents[2] / "lib" / "librevo2_ik_sdk.so")
    # Source-tree layout: <project>/python/revo2_hand_ik/native.py
    candidates.append(here.parents[2] / "lib" / "librevo2_ik_sdk.so")
    return candidates


def _load_native_library(path: str | os.PathLike[str] | None) -> ct.CDLL:
    if path is not None:
        selected = Path(path).expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"Revo2 SDK shared library not found: {selected}")
    else:
        selected = next((p for p in _default_library_candidates() if p.is_file()), None)
        if selected is None:
            searched = "\n  ".join(str(p) for p in _default_library_candidates())
            raise FileNotFoundError(
                "librevo2_ik_sdk.so was not found. Build the runtime first or set "
                "REVO2_IK_SDK_LIBRARY. Searched:\n  " + searched
            )

    # RTLD_GLOBAL lets libpinocchio_default/libpinocchio_parsers in the same
    # runtime lib directory resolve normally when run scripts set LD_LIBRARY_PATH.
    return ct.CDLL(str(selected), mode=ct.RTLD_GLOBAL)


class Revo2HandIkSdk:
    """Stateful Python wrapper around one left- or right-hand native solver."""

    def __init__(self, config_path: str | os.PathLike[str], library_path: str | os.PathLike[str] | None = None):
        self._lib = _load_native_library(library_path)
        self._configure_signatures()
        self._handle = self._lib.revo2_ik_create()
        if not self._handle:
            raise Revo2IkError("revo2_ik_create returned null")
        self.config_path = str(Path(config_path).expanduser().resolve())
        result = self._lib.revo2_ik_init(self._handle, self.config_path.encode("utf-8"))
        if result != 0:
            message = self.last_error()
            self.close()
            raise Revo2IkError(f"revo2_ik_init failed ({result}): {message}")

    def _configure_signatures(self) -> None:
        self._lib.revo2_ik_create.argtypes = []
        self._lib.revo2_ik_create.restype = ct.c_void_p
        self._lib.revo2_ik_destroy.argtypes = [ct.c_void_p]
        self._lib.revo2_ik_destroy.restype = None
        self._lib.revo2_ik_init.argtypes = [ct.c_void_p, ct.c_char_p]
        self._lib.revo2_ik_init.restype = ct.c_int
        self._lib.revo2_ik_solve.argtypes = [ct.c_void_p, ct.POINTER(HandMocapInput), ct.POINTER(IkOutput)]
        self._lib.revo2_ik_solve.restype = ct.c_int
        self._lib.revo2_ik_reset.argtypes = [ct.c_void_p]
        self._lib.revo2_ik_reset.restype = ct.c_int
        self._lib.revo2_ik_last_error.argtypes = [ct.c_void_p]
        self._lib.revo2_ik_last_error.restype = ct.c_char_p
        self._lib.revo2_ik_init_input.argtypes = [ct.POINTER(HandMocapInput)]
        self._lib.revo2_ik_init_input.restype = None
        self._lib.revo2_ik_init_output.argtypes = [ct.POINTER(IkOutput)]
        self._lib.revo2_ik_init_output.restype = None
        self._lib.revo2_ik_sizeof_hand_mocap_input.argtypes = []
        self._lib.revo2_ik_sizeof_hand_mocap_input.restype = ct.c_size_t
        self._lib.revo2_ik_sizeof_output.argtypes = []
        self._lib.revo2_ik_sizeof_output.restype = ct.c_size_t
        self._lib.revo2_ik_init_debug_frame.argtypes = [ct.POINTER(DebugFrame)]
        self._lib.revo2_ik_init_debug_frame.restype = None
        self._lib.revo2_ik_get_debug_frame.argtypes = [ct.c_void_p, ct.POINTER(DebugFrame)]
        self._lib.revo2_ik_get_debug_frame.restype = ct.c_int
        self._lib.revo2_ik_sizeof_debug_frame.argtypes = []
        self._lib.revo2_ik_sizeof_debug_frame.restype = ct.c_size_t
        native_input_size = int(self._lib.revo2_ik_sizeof_hand_mocap_input())
        native_output_size = int(self._lib.revo2_ik_sizeof_output())
        native_debug_size = int(self._lib.revo2_ik_sizeof_debug_frame())
        if (native_input_size != ct.sizeof(HandMocapInput) or
                native_output_size != ct.sizeof(IkOutput) or
                native_debug_size != ct.sizeof(DebugFrame)):
            raise Revo2IkError(
                "ctypes ABI size mismatch: "
                f"input python/native={ct.sizeof(HandMocapInput)}/{native_input_size}, "
                f"output python/native={ct.sizeof(IkOutput)}/{native_output_size}, "
                f"debug python/native={ct.sizeof(DebugFrame)}/{native_debug_size}"
            )

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.revo2_ik_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "Revo2HandIkSdk":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - destructor timing is interpreter-dependent.
        try:
            self.close()
        except Exception:
            pass

    def last_error(self) -> str:
        if not getattr(self, "_handle", None):
            return "SDK handle is closed"
        raw = self._lib.revo2_ik_last_error(self._handle)
        return raw.decode("utf-8", errors="replace") if raw else ""

    def reset(self) -> None:
        result = self._lib.revo2_ik_reset(self._handle)
        if result != 0:
            raise Revo2IkError(f"revo2_ik_reset failed ({result}): {self.last_error()}")

    def make_input(self) -> HandMocapInput:
        value = HandMocapInput()
        self._lib.revo2_ik_init_input(ct.byref(value))
        return value

    @staticmethod
    def _set_point(dst: Point3, value: Any) -> None:
        if value is None:
            dst.valid = 0
            return
        if len(value) != 3:
            raise ValueError(f"Point must contain exactly 3 values, got {value!r}")
        dst.x, dst.y, dst.z = (float(value[0]), float(value[1]), float(value[2]))
        dst.valid = 1

    @staticmethod
    def _set_quat(dst: Quaternion, value: Any) -> None:
        if value is None:
            dst.valid = 0
            return
        if len(value) != 4:
            raise ValueError(f"Quaternion must contain exactly 4 values in w,x,y,z order, got {value!r}")
        dst.w, dst.x, dst.y, dst.z = (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        dst.valid = 1

    def input_from_mapping(self, points: Mapping[str, Any], quaternions_wxyz: Mapping[str, Any] | None = None,
                           *, sequence: int = 0, timestamp_ms: int = 0) -> HandMocapInput:
        """Create a native input from hand-only point dictionaries.

        Required point keys: thumb_near/middle/far, index_near/middle/far,
        middle_near/middle/far, ring_near/middle/far, little_near/middle/far.
        ``hand`` is optional.  FSS quaternions are forwarded for hand and all
        finger nodes.  The default C++ long-finger path uses positions; thumb
        flex uses near/far orientation when available and has a position
        fallback for sources that do not publish orientation.
        """
        inp = self.make_input()
        for key in (
            "hand", "thumb_near", "thumb_middle", "thumb_far",
            "index_near", "index_middle", "index_far",
            "middle_near", "middle_middle", "middle_far",
            "ring_near", "ring_middle", "ring_far",
            "little_near", "little_middle", "little_far",
        ):
            self._set_point(getattr(inp, key), points.get(key))
        quaternions_wxyz = quaternions_wxyz or {}
        for key in (
            "hand",
            "thumb_near", "thumb_middle", "thumb_far",
            "index_near", "index_middle", "index_far",
            "middle_near", "middle_middle", "middle_far",
            "ring_near", "ring_middle", "ring_far",
            "little_near", "little_middle", "little_far",
        ):
            self._set_quat(getattr(inp, f"{key}_quat"), quaternions_wxyz.get(key))
        inp.sequence = int(sequence)
        inp.timestamp_ms = int(timestamp_ms)
        return inp

    @staticmethod
    def _point_to_dict(point: Point3) -> dict[str, Any]:
        return {
            "x": float(point.x),
            "y": float(point.y),
            "z": float(point.z),
            "valid": bool(point.valid),
        }

    def get_debug_frame(self) -> dict[str, Any]:
        """Return source-overlay, thumb-task and long-finger curve-task diagnostics."""
        debug = DebugFrame()
        self._lib.revo2_ik_init_debug_frame(ct.byref(debug))
        result = self._lib.revo2_ik_get_debug_frame(self._handle, ct.byref(debug))
        if result != 0:
            raise Revo2IkError(f"revo2_ik_get_debug_frame failed ({result}): {self.last_error()}")
        count = max(0, min(int(debug.point_count), DEBUG_POINT_COUNT))
        points = {
            DEBUG_POINT_NAMES[index]: self._point_to_dict(debug.mapped_input_points[index])
            for index in range(count)
        }
        long_finger_tasks = []
        for index, name in enumerate(DEBUG_LONG_FINGER_NAMES):
            task = debug.long_finger_tasks[index]
            long_finger_tasks.append({
                "name": name,
                "valid": bool(task.valid),
                "target_middle": self._point_to_dict(task.target_middle),
                "target_far": self._point_to_dict(task.target_far),
                "robot_middle": self._point_to_dict(task.robot_middle),
                "robot_far": self._point_to_dict(task.robot_far),
            })
        return {
            "sequence": int(debug.sequence),
            "timestamp_ms": int(debug.timestamp_ms),
            "calibrated": bool(debug.calibrated),
            "calibration_stage": int(debug.calibration_stage),
            "calibration_samples": int(debug.calibration_samples),
            "calibration_required": int(debug.calibration_required),
            "calibration_warmup_remaining": int(debug.calibration_warmup_remaining),
            "calibration_rejected": int(debug.calibration_rejected),
            "calibration_quality_failures": int(debug.calibration_quality_failures),
            "calibration_point_rms_m": float(debug.calibration_point_rms_m),
            "calibration_length_cv": float(debug.calibration_length_cv),
            "calibration_status": _decode_c_string(debug.calibration_status),
            "frame_id": _decode_c_string(debug.frame_id),
            "points": points,
            "thumb_target_middle": self._point_to_dict(debug.thumb_target_middle),
            "thumb_target_far": self._point_to_dict(debug.thumb_target_far),
            "thumb_robot_middle": self._point_to_dict(debug.thumb_robot_middle),
            "thumb_robot_far": self._point_to_dict(debug.thumb_robot_far),
            "long_finger_tasks": long_finger_tasks,
        }

    def solve_native(self, inp: HandMocapInput) -> dict[str, Any]:
        out = IkOutput()
        self._lib.revo2_ik_init_output(ct.byref(out))
        started = perf_counter()
        result = self._lib.revo2_ik_solve(self._handle, ct.byref(inp), ct.byref(out))
        wall_ms = (perf_counter() - started) * 1000.0
        if result != 0:
            raise Revo2IkError(f"revo2_ik_solve failed ({result}): {self.last_error()}")
        count = max(0, min(int(out.joint_count), MAX_OUTPUT_JOINTS))
        joints = []
        for i in range(count):
            joints.append({
                "name": _decode_c_string(out.joint_names[i]),
                "norm_0_100": float(out.position_norm_0_100[i]),
                "rad": float(out.position_rad[i]),
            })
        return {
            "joint_count": count,
            "joints": joints,
            "solve_time_ms": float(out.solve_time_ms),
            "python_call_time_ms": wall_ms,
            "debug": self.get_debug_frame(),
        }

    def solve(self, points: Mapping[str, Any], quaternions_wxyz: Mapping[str, Any] | None = None,
              *, sequence: int = 0, timestamp_ms: int = 0) -> dict[str, Any]:
        return self.solve_native(self.input_from_mapping(
            points, quaternions_wxyz, sequence=sequence, timestamp_ms=timestamp_ms
        ))
