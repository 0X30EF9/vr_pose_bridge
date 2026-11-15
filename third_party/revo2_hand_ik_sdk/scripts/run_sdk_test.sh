#!/usr/bin/env bash
# SDK 测试运行：Formotion G2 UDP 输入 -> Python SDK 接口 -> RViz。默认启动 RViz。
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREFIX="${PINOCCHIO_PREFIX:-$ROOT_DIR/thirdparty/install/pinocchio-4.0.0}"
export LD_LIBRARY_PATH="$ROOT_DIR/lib:$PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT_DIR/python:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export REVO2_IK_SDK_LIBRARY="${REVO2_IK_SDK_LIBRARY:-$ROOT_DIR/lib/librevo2_ik_sdk.so}"
[[ -f "$REVO2_IK_SDK_LIBRARY" ]] || {
  echo "[sdk-test][ERROR] SDK library not found: $REVO2_IK_SDK_LIBRARY" >&2
  echo "[sdk-test][HINT] run: ./scripts/build_sdk.sh" >&2
  exit 1
}

ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP_FILE="${ROS_SETUP_FILE:-/opt/ros/${ROS_DISTRO}/setup.bash}"
[[ -f "$ROS_SETUP_FILE" ]] || { echo "[sdk-test][ERROR] ROS2 setup missing: $ROS_SETUP_FILE" >&2; exit 1; }
set +u
source "$ROS_SETUP_FILE"
set -u
source "$ROOT_DIR/tools/ros_local_rviz_env.sh"
command -v rviz2 >/dev/null 2>&1 || { echo "[sdk-test][ERROR] rviz2 not found." >&2; exit 1; }
command -v ros2 >/dev/null 2>&1 || { echo "[sdk-test][ERROR] ros2 not found." >&2; exit 1; }

CONFIG="${REVO2_SDK_CONFIG:-$ROOT_DIR/bin/conf/mocap_ik/mocap_ik_revo2_dual_hand_sdk.json}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  shift
fi
[[ -f "$CONFIG" ]] || { echo "[sdk-test][ERROR] config not found: $CONFIG" >&2; exit 1; }

FSS_LISTEN_IP="${FSS_LISTEN_IP:-0.0.0.0}"
FSS_LISTEN_PORT="${FSS_LISTEN_PORT:-15000}"
FSS_SOURCE_IP="${FSS_SOURCE_IP:-}"
CONTROL_MIRROR_IP="${CONTROL_MIRROR_IP:-127.0.0.1}"
CONTROL_MIRROR_PORT="${CONTROL_MIRROR_PORT:-16124}"
DEBUG_OVERLAY_IP="${DEBUG_OVERLAY_IP:-127.0.0.1}"
DEBUG_OVERLAY_PORT="${DEBUG_OVERLAY_PORT:-16125}"

LEFT_URDF="$ROOT_DIR/bin/conf/mocap_ik/revo2/urdf/revo2_left_hand.urdf"
RIGHT_URDF="$ROOT_DIR/bin/conf/mocap_ik/revo2/urdf/revo2_right_hand.urdf"
DUAL_URDF="$ROOT_DIR/bin/generated/revo2_dual_hand_rviz.urdf"
RVIZ_CFG="$ROOT_DIR/rviz/revo2_dual_hand.rviz"
LOG_DIR="$ROOT_DIR/bin/log/SDK测试运行"
JOINT_TOPIC="/joint_states"
mkdir -p "$LOG_DIR" "$(dirname "$DUAL_URDF")"

RUNTIME_TAG="sdk-test"
RUNTIME_LOG_DIR="$LOG_DIR"
source "$ROOT_DIR/tools/runtime_session.sh"
runtime_session_init
runtime_install_signal_traps

python3 "$ROOT_DIR/python/generate_dual_hand_urdf.py" \
  --left "$LEFT_URDF" --right "$RIGHT_URDF" --output "$DUAL_URDF"

runtime_start "control_udp_to_rviz" "$LOG_DIR/control_udp_to_rviz.log" \
  python3 "$ROOT_DIR/tools/control_udp_to_rviz.py" \
    --listen-ip "$CONTROL_MIRROR_IP" --listen-port "$CONTROL_MIRROR_PORT" \
    --debug-listen-ip "$DEBUG_OVERLAY_IP" --debug-listen-port "$DEBUG_OVERLAY_PORT" \
    --joint-topic "$JOINT_TOPIC"
MIRROR_PID="$RUNTIME_LAST_PID"

runtime_start "robot_state_publisher" "$LOG_DIR/robot_state_publisher.log" \
  ros2 run robot_state_publisher robot_state_publisher "$DUAL_URDF" \
    --ros-args \
    -r __node:=revo2_python_sdk_robot_state_publisher \
    -p publish_frequency:=100.0
RSP_PID="$RUNTIME_LAST_PID"

UDP_SOURCE_ARGS=()
if [[ -n "$FSS_SOURCE_IP" ]]; then
  UDP_SOURCE_ARGS+=(--source-ip "$FSS_SOURCE_IP")
fi

runtime_start "udp_to_sdk_test" "$LOG_DIR/udp_to_sdk_test.log" \
  python3 -u "$ROOT_DIR/python/examples/udp_to_sdk_test.py" \
    --sdk-config "$CONFIG" \
    --library "$REVO2_IK_SDK_LIBRARY" \
    --listen-ip "$FSS_LISTEN_IP" \
    --listen-port "$FSS_LISTEN_PORT" \
    "${UDP_SOURCE_ARGS[@]}" \
    --control-target-ip "$CONTROL_MIRROR_IP" \
    --control-target-port "$CONTROL_MIRROR_PORT" \
    --overlay-target-ip "$DEBUG_OVERLAY_IP" \
    --overlay-target-port "$DEBUG_OVERLAY_PORT" \
    "$@"
SDK_TEST_PID="$RUNTIME_LAST_PID"

sleep 1
for item in \
  "control_udp_to_rviz:$MIRROR_PID:$LOG_DIR/control_udp_to_rviz.log" \
  "robot_state_publisher:$RSP_PID:$LOG_DIR/robot_state_publisher.log" \
  "udp_to_sdk_test:$SDK_TEST_PID:$LOG_DIR/udp_to_sdk_test.log"; do
  IFS=: read -r name pid log <<< "$item"
  if ! runtime_pid_is_alive "$pid"; then
    echo "[sdk-test][ERROR] $name exited. Last log lines:" >&2
    tail -n 80 "$log" >&2 || true
    exit 1
  fi
done

runtime_start "rviz2" "$LOG_DIR/rviz2.log" rviz2 -d "$RVIZ_CFG"
RVIZ_PID="$RUNTIME_LAST_PID"

echo "[sdk-test][INFO] Formotion G2 UDP input=$FSS_LISTEN_IP:$FSS_LISTEN_PORT source=${FSS_SOURCE_IP:-any}"
echo "[sdk-test][INFO] Python SDK output unit=rad, no 0..100 normalization."
echo "[sdk-test][INFO] Calibration: keep four fingers fully extended and still for about 2 seconds; wait for calibrated=L1/R1."
echo "[sdk-test][INFO] RViz mirror=$CONTROL_MIRROR_IP:$CONTROL_MIRROR_PORT overlay=$DEBUG_OVERLAY_IP:$DEBUG_OVERLAY_PORT"
echo "[sdk-test][INFO] logs=$LOG_DIR. Close RViz or press Ctrl-C to stop."
echo "[sdk-test][INFO] UDP receive log: tail -f '$LOG_DIR/udp_to_sdk_test.log'"
echo "[sdk-test][INFO] Verify destination packets: sudo tcpdump -ni any -c 5 -vv 'udp dst port $FSS_LISTEN_PORT'"

while runtime_pid_is_alive "$RVIZ_PID"; do
  for item in \
    "control_udp_to_rviz:$MIRROR_PID:$LOG_DIR/control_udp_to_rviz.log" \
    "robot_state_publisher:$RSP_PID:$LOG_DIR/robot_state_publisher.log" \
    "udp_to_sdk_test:$SDK_TEST_PID:$LOG_DIR/udp_to_sdk_test.log"; do
    IFS=: read -r name pid log <<< "$item"
    if ! runtime_pid_is_alive "$pid"; then
      echo "[sdk-test][ERROR] $name exited. Last log lines:" >&2
      tail -n 80 "$log" >&2 || true
      exit 1
    fi
  done
  sleep 0.5
done

wait "$RVIZ_PID" 2>/dev/null || true
