"""Unity/OpenXR controller UDP to Cartesian IK, with optional rosbridge output.

The default is an ideal-feedback offline preview. ROS and native vendor SDKs
are not imported by the command gate or by preview mode.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import socket
import time

import numpy as np

from teleop_control_safety import command_time_step, joint_velocity_limits_from_urdf, require_fresh_feedback
from teleop_ik_types import IKCommand
from teleop_joint_schema import ARM_JOINT_NAMES, ARM_MOTOR_IDS, arm_joint_limits_from_urdf

URDF = Path(__file__).resolve().parent / "models/walker_tienkung_ei_description/walker_tienkung_ei.urdf"


def side_indices(side):
    if side not in {"left", "right", "both"}:
        raise ValueError("control_side must be left, right or both")
    return np.arange(0 if side != "right" else 7, 14 if side != "left" else 7)


class ControllerCommandGate:
    """Publication-boundary guard with independent, committed per-arm history.

    prepare() is pure with respect to command history. Call commit() only after
    a successful send, and never replace the inactive arm's committed target
    with measured feedback. A clutch changes controller anchors, not baseline.
    """

    def __init__(self, baseline_q, *, control_rate=100., max_joint_velocity=.48,
                 max_step=.06, max_from_baseline=1., feedback_timeout=.5,
                 input_timeout=.3, control_side="both"):
        self.control_rate = float(control_rate)
        self.max_step = float(max_step)
        self.max_from_baseline = float(max_from_baseline)
        self.feedback_timeout, self.input_timeout = float(feedback_timeout), float(input_timeout)
        self.control_side = control_side
        side_indices(control_side)
        for value in (self.control_rate, max_joint_velocity, self.max_step,
                      self.max_from_baseline, self.feedback_timeout, self.input_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("controller rates, timeouts and motion bounds must be positive and finite")
        self.limits = np.asarray(arm_joint_limits_from_urdf(URDF))
        self.velocities = np.minimum(float(max_joint_velocity), joint_velocity_limits_from_urdf(URDF, ARM_JOINT_NAMES))
        self.baseline = self._q(baseline_q)
        self.baseline.setflags(write=False)
        self._last = self.baseline.copy()
        self._sent = {"left": False, "right": False}
        self._times = {"left": None, "right": None}

    def _q(self, q):
        if np.asarray(q).dtype == bool:
            raise ValueError("arm joint positions must be numbers, not booleans")
        q = np.asarray(q, dtype=float)
        if q.shape != (14,) or not np.all(np.isfinite(q)):
            raise ValueError("arm feedback/command must contain 14 finite joint positions")
        if np.any(q < self.limits[:, 0] - 1e-9) or np.any(q > self.limits[:, 1] + 1e-9):
            raise ValueError("arm feedback/command is outside URDF limits")
        return q.copy()

    def reference(self, current_q):
        result = self._q(current_q)
        for side in ("left", "right"):
            if self._sent[side]:
                indices = side_indices(side)
                result[indices] = self._last[indices]
        return result

    def _command(self, command):
        if (tuple(command.joint_names) != ARM_JOINT_NAMES or tuple(command.motor_ids) != ARM_MOTOR_IDS
                or command.waist_position is not None):
            raise ValueError("controller output must use the arm-only 14-axis schema")
        indices = side_indices(command.control_side)
        if self.control_side != "both" and command.control_side != self.control_side:
            raise ValueError("candidate enables an arm outside the configured control side")
        return self._q(command.positions), indices

    def prepare(self, command, current_q, feedback_received_at, input_received_at, now):
        require_fresh_feedback(feedback_received_at, now, self.feedback_timeout, "arm/status")
        require_fresh_feedback(input_received_at, now, self.input_timeout, "VR controller input")
        candidate, indices = self._command(command)
        current = self._q(current_q)
        measured_step = np.minimum(self.max_step, self.velocities / self.control_rate)
        lower = np.maximum.reduce((self.limits[:, 0], self.baseline - self.max_from_baseline, current - measured_step))
        upper = np.minimum.reduce((self.limits[:, 1], self.baseline + self.max_from_baseline, current + measured_step))
        active_sides = ("left", "right") if command.control_side == "both" else (command.control_side,)
        for side in active_sides:
            if self._sent[side]:
                arm = side_indices(side)
                step = np.minimum(self.max_step, self.velocities[arm] * command_time_step(self.control_rate, now, self._times[side]))
                lower[arm] = np.maximum(lower[arm], self._last[arm] - step)
                upper[arm] = np.minimum(upper[arm], self._last[arm] + step)
        if np.any(lower[indices] > upper[indices] + 1e-12):
            raise RuntimeError("measured feedback and the committed command envelope are incompatible")
        result = current.copy()
        result[indices] = np.clip(candidate[indices], lower[indices], upper[indices])
        return replace(command, positions=result, meta={**command.meta,
                       "publication_guard_limited": bool(np.any(np.abs(result[indices]-candidate[indices]) > 1e-12)),
                       "residual_scope": "solver output before publication guard"})

    def commit(self, command, now):
        candidate, indices = self._command(command)
        if not math.isfinite(now):
            raise ValueError("commit time must be finite")
        sides = ("left", "right") if command.control_side == "both" else (command.control_side,)
        if any(self._times[side] is not None and now < self._times[side] for side in sides):
            raise ValueError("commit time cannot move backwards")
        self._last[indices] = candidate[indices]
        for side in sides:
            self._sent[side], self._times[side] = True, now


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def publish_controller_command(gate, command, input_received_at, snapshot, publish, clock=None):
    """Recheck measured state at the send boundary; failed sends never commit."""
    from teleop_ik_adapters import build_rosbridge_motor_message
    from mocap_replay_to_robot import make_motor_templates

    clock = time.monotonic if clock is None else clock
    received, current = snapshot()
    checked = gate.prepare(command, current, received, input_received_at, clock())
    templates = make_motor_templates(checked.control_side, .7, 2.)
    build_rosbridge_motor_message(checked, templates)
    received, current = snapshot()
    checked = gate.prepare(checked, current, received, input_received_at, clock())
    message = build_rosbridge_motor_message(checked, templates)
    now = clock()
    require_fresh_feedback(received, now, gate.feedback_timeout, "arm/status")
    require_fresh_feedback(input_received_at, now, gate.input_timeout, "VR controller input")
    publish(message)
    gate.commit(checked, clock())
    return checked


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=15010)
    parser.add_argument("--control-rate", type=float, default=100.)
    parser.add_argument("--control-side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--position-scale", type=float, default=1.)
    parser.add_argument("--orientation-scale", type=float, default=1.)
    parser.add_argument("--ik-task", choices=("pose", "position_only"), default="pose")
    parser.add_argument("--max-joint-velocity", type=float, default=.48)
    parser.add_argument("--max-joint-delta-per-step", type=float, default=.06)
    parser.add_argument("--max-command-delta-from-baseline", type=float, default=1.)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--connect-timeout", type=float, default=8.)
    parser.add_argument("--status-timeout", type=float, default=10.)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Read robot feedback and publish arm commands")
    modes.add_argument("--preview-robot", action="store_true", help="Read robot feedback without publishing")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--log-rate", type=float, default=1.)
    parser.add_argument("--initial-q", type=float, nargs=14, default=None, help="Offline preview initial pose only")
    args = parser.parse_args(argv)
    for name in ("control_rate", "max_joint_velocity", "max_joint_delta_per_step", "max_command_delta_from_baseline",
                 "connect_timeout", "status_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be positive and finite")
    if not math.isfinite(args.log_rate) or args.log_rate < 0:
        parser.error("log_rate must be finite and nonnegative")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be positive and finite")
    if not 1 <= args.udp_port <= 65535 or not 1 <= args.port <= 65535:
        parser.error("ports must be in 1..65535")
    if args.initial_q is not None and (args.execute or args.preview_robot):
        parser.error("initial-q is only for offline preview; robot modes require measured feedback")
    return args


def run(args):
    from teleop_vr_controller import VrControllerConfig, VrControllerRetargeter, parse_controller_packet

    mapper = VrControllerRetargeter(VrControllerConfig(
        control_side=args.control_side, control_rate_hz=args.control_rate,
        max_joint_velocity=args.max_joint_velocity, max_joint_delta_per_step=args.max_joint_delta_per_step,
        max_command_delta_from_baseline=args.max_command_delta_from_baseline,
        position_scale=args.position_scale, orientation_scale=args.orientation_scale,
        task_mode=args.ik_task))
    client = subscriber = publisher = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    preview_q = np.zeros(14) if args.initial_q is None else np.asarray(args.initial_q, dtype=float)
    feedback = None
    try:
        if args.execute or args.preview_robot:
            import roslibpy
            from mocap_replay_to_robot import ARM_CMD_TOPIC, ARM_CMD_TYPE, ARM_STATUS_TOPIC, ARM_STATUS_TYPE, ArmStatusBuffer
            client = roslibpy.Ros(host=args.host, port=args.port)
            client.run(timeout=args.connect_timeout)
            if not client.is_connected:
                raise RuntimeError("rosbridge is not connected")
            feedback = ArmStatusBuffer()
            subscriber = roslibpy.Topic(client, ARM_STATUS_TOPIC, ARM_STATUS_TYPE)
            subscriber.subscribe(feedback.callback)
            sample = feedback.wait_complete(args.status_timeout)
            if sample is None:
                raise RuntimeError("no fresh, complete arm/status feedback")
            preview_q = sample[1]
            if args.execute:
                publisher = roslibpy.Topic(client, ARM_CMD_TOPIC, ARM_CMD_TYPE)
                publisher.advertise()
        gate = ControllerCommandGate(preview_q, control_rate=args.control_rate,
            max_joint_velocity=args.max_joint_velocity, max_step=args.max_joint_delta_per_step,
            max_from_baseline=args.max_command_delta_from_baseline, control_side=args.control_side)
        sock.bind((args.bind, args.udp_port))
        sock.setblocking(False)
        mode = "robot_execute" if args.execute else "robot_feedback_preview" if args.preview_robot else "ideal_feedback_preview"
        print(json.dumps({"event": "controller_ready", "mode": mode, "bind": args.bind, "udp_port": args.udp_port,
                          "control_rate_hz": args.control_rate, "task": args.ik_task}), flush=True)
        began = deadline = time.monotonic()
        latest = None
        source = None
        last_log = float("-inf")
        accepted = computed = submitted = invalid = 0
        while args.duration is None or time.monotonic() - began < args.duration:
            if latest is not None and time.monotonic() - latest.received_at > .3:
                mapper.invalidate("stale_input")
                latest = None
            for _ in range(256):
                try:
                    packet, address = sock.recvfrom(16385)
                except BlockingIOError:
                    break
                if source is not None and address != source:
                    continue
                try:
                    frame = parse_controller_packet(packet, received_at=time.monotonic())
                    if mapper.ingest(frame, now=time.monotonic()):
                        source = address
                        latest = frame
                        accepted += 1
                except (ValueError, TypeError) as exc:
                    latest = None
                    mapper.invalidate("invalid_packet")
                    invalid += 1
                    print(json.dumps({"event": "controller_input_rejected", "reason": str(exc)}), flush=True)
            if latest is not None:
                now = time.monotonic()
                if now - latest.received_at > .3:
                    mapper.invalidate("stale_input")
                    latest = None
                    continue
                if feedback is not None:
                    sample = feedback.latest_complete()
                    if sample is None:
                        raise RuntimeError("arm/status is invalid")
                    received, current = sample
                    require_fresh_feedback(received, now, .5, "arm/status")
                    if not client.is_connected:
                        raise RuntimeError("rosbridge disconnected")
                else:
                    received, current = now, preview_q.copy()
                command = mapper.update(latest, current, last_q_cmd=gate.reference(current), now=now)
                if command is not None:
                    computed += 1
                    if feedback is not None:
                        sample = feedback.latest_complete()
                        if sample is None or not client.is_connected:
                            raise RuntimeError("robot feedback/connection became invalid")
                        received, current = sample
                    if publisher is not None:
                        def snapshot():
                            sample = feedback.latest_complete()
                            if sample is None or not client.is_connected:
                                raise RuntimeError("robot feedback/connection became invalid before publication")
                            return sample
                        checked = publish_controller_command(gate, command, latest.received_at, snapshot, publisher.publish)
                        submitted += 1
                    else:
                        checked = gate.prepare(command, current, received, latest.received_at, time.monotonic())
                        gate.commit(checked, time.monotonic())
                    if feedback is None:
                        preview_q = checked.positions.copy()
                    if args.log_rate and now - last_log >= 1. / args.log_rate:
                        print(json.dumps(json_value({"event": "controller_ik", "mode": mode, "sequence": latest.sequence,
                            "side": checked.control_side, "positions": checked.positions, "meta": checked.meta})), flush=True)
                        last_log = now
            deadline += 1. / args.control_rate
            now = time.monotonic()
            if deadline < now:
                deadline += (math.floor((now-deadline)*args.control_rate)+1) / args.control_rate
            if deadline > now:
                time.sleep(deadline-now)
        print(json.dumps({"event": "controller_summary", "accepted": accepted, "computed": computed,
                          "robot_submitted": submitted, "invalid": invalid}), flush=True)
        return 0
    finally:
        sock.close()
        if subscriber is not None:
            subscriber.unsubscribe()
        if publisher is not None:
            publisher.unadvertise()
        if client is not None:
            client.terminate()


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"event": "controller_stopped", "reason": str(exc)}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
