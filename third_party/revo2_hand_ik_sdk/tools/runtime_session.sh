#!/usr/bin/env bash
# Shared process/session supervision for the integrated runtime launchers.
#
# Each child is started in an independent POSIX session through `setsid`.
# This prevents Ctrl-Z from stopping the bridge/RViz helpers together with the
# launcher shell. A stale PID file is reclaimed automatically on the next run.

# This file is intended to be sourced by a launcher that has `set -euo pipefail`.

runtime_require_tools() {
  command -v setsid >/dev/null 2>&1 || {
    echo "[${RUNTIME_TAG:-runtime}][ERROR] 'setsid' is required but was not found." >&2
    return 1
  }
  command -v ps >/dev/null 2>&1 || {
    echo "[${RUNTIME_TAG:-runtime}][ERROR] 'ps' is required but was not found." >&2
    return 1
  }
}

runtime_pid_is_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null
}

runtime_pgid_of() {
  local pid="$1"
  ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]'
}

runtime_own_pgid() {
  runtime_pgid_of "$$"
}

runtime_signal_record() {
  local pid="$1"
  local pgid="$2"
  local signal="$3"
  local own_pgid
  own_pgid="$(runtime_own_pgid || true)"

  # A child started through setsid is its own process-group leader. For old
  # session.pid files we may only know a child PID; derive its current PGID.
  if [[ ! "$pgid" =~ ^[1-9][0-9]*$ ]]; then
    pgid="$(runtime_pgid_of "$pid" || true)"
  fi

  # Never signal the current launcher's own process group. That would include
  # the user's interactive shell in some terminal/job-control configurations.
  if [[ "$pgid" =~ ^[1-9][0-9]*$ && "$pgid" != "$own_pgid" ]]; then
    kill "-$signal" -- "-$pgid" 2>/dev/null || true
  elif runtime_pid_is_alive "$pid"; then
    kill "-$signal" "$pid" 2>/dev/null || true
  fi
}

runtime_wait_for_exit() {
  local pid="$1"
  local timeout_s="$2"
  local deadline=$((SECONDS + timeout_s))

  while runtime_pid_is_alive "$pid"; do
    if (( SECONDS >= deadline )); then
      return 1
    fi
    sleep 0.1
  done
  return 0
}

runtime_stop_record() {
  local pid="$1"
  local pgid="$2"
  local name="$3"

  runtime_pid_is_alive "$pid" || return 0

  # Ctrl-Z leaves processes in state T. Resume first; otherwise an ordinary
  # TERM may remain pending and the stale process continues to occupy ports.
  runtime_signal_record "$pid" "$pgid" CONT
  runtime_signal_record "$pid" "$pgid" TERM

  if ! runtime_wait_for_exit "$pid" 3; then
    echo "[${RUNTIME_TAG}][WARN] force-stopping stale ${name} (pid=${pid})." >&2
    runtime_signal_record "$pid" "$pgid" KILL
    runtime_wait_for_exit "$pid" 2 || true
  fi
}

runtime_reclaim_previous_session() {
  local line pid pgid name derived_pgid

  if [[ -f "$RUNTIME_PID_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ -n "$line" ]] || continue

      # New format: pid|pgid|name. Old releases stored one bare PID per line.
      if [[ "$line" == *"|"* ]]; then
        IFS='|' read -r pid pgid name <<< "$line"
      else
        pid="$line"
        pgid=""
        name="legacy-session-child"
      fi

      [[ "$pid" =~ ^[1-9][0-9]*$ ]] || continue
      if runtime_pid_is_alive "$pid"; then
        derived_pgid="${pgid:-$(runtime_pgid_of "$pid" || true)}"
        echo "[${RUNTIME_TAG}][WARN] reclaiming stale ${name} (pid=${pid}, pgid=${derived_pgid:-unknown})." >&2
        runtime_stop_record "$pid" "$derived_pgid" "$name"
      fi
    done < "$RUNTIME_PID_FILE"
  fi

  # A killed terminal or power interruption can leave only the lock directory.
  # Once any recorded children above are reclaimed, that orphaned lock is safe
  # to remove before a new launcher claims it.
  rm -f "$RUNTIME_PID_FILE"
  rm -rf "$RUNTIME_LOCK_DIR"
}

runtime_session_init() {
  : "${RUNTIME_TAG:?RUNTIME_TAG must be set before runtime_session_init}"
  : "${RUNTIME_LOG_DIR:?RUNTIME_LOG_DIR must be set before runtime_session_init}"

  runtime_require_tools
  mkdir -p "$RUNTIME_LOG_DIR"
  RUNTIME_PID_FILE="$RUNTIME_LOG_DIR/session.pid"
  RUNTIME_LOCK_DIR="$RUNTIME_LOG_DIR/session.lock"
  RUNTIME_CHILDREN=()
  RUNTIME_CLEANED=0

  runtime_reclaim_previous_session

  if ! mkdir "$RUNTIME_LOCK_DIR" 2>/dev/null; then
    # A lock without a PID file can result from an interrupted shell before the
    # original launcher installed its EXIT trap. Reclaim once more, then refuse
    # only if another live launcher recreated the lock concurrently.
    runtime_reclaim_previous_session
    mkdir "$RUNTIME_LOCK_DIR" 2>/dev/null || {
      echo "[${RUNTIME_TAG}][ERROR] another launcher is initializing: $RUNTIME_LOCK_DIR" >&2
      return 1
    }
  fi

  : > "$RUNTIME_PID_FILE"
}

runtime_start() {
  local name="$1"
  local log_file="$2"
  shift 2

  # Do not use '--fork': the launcher must retain the actual child PID in order
  # to supervise it. A background shell child is not a process-group leader, so
  # GNU setsid can create the new session without an intermediate fork.
  setsid "$@" >"$log_file" 2>&1 &
  RUNTIME_LAST_PID=$!
  local pgid
  pgid="$(runtime_pgid_of "$RUNTIME_LAST_PID" || true)"
  [[ "$pgid" =~ ^[1-9][0-9]*$ ]] || pgid="$RUNTIME_LAST_PID"

  printf '%s|%s|%s\n' "$RUNTIME_LAST_PID" "$pgid" "$name" >> "$RUNTIME_PID_FILE"
  RUNTIME_CHILDREN+=("${RUNTIME_LAST_PID}|${pgid}|${name}")
}

runtime_cleanup() {
  [[ "${RUNTIME_CLEANED:-0}" == "1" ]] && return 0
  RUNTIME_CLEANED=1

  local record pid pgid name
  # Stop in reverse creation order: RViz first, then producers/ROS helpers.
  for ((idx=${#RUNTIME_CHILDREN[@]} - 1; idx>=0; --idx)); do
    record="${RUNTIME_CHILDREN[$idx]}"
    IFS='|' read -r pid pgid name <<< "$record"
    runtime_stop_record "$pid" "$pgid" "$name"
  done

  rm -f "${RUNTIME_PID_FILE:-}" 2>/dev/null || true
  rmdir "${RUNTIME_LOCK_DIR:-}" 2>/dev/null || rm -rf "${RUNTIME_LOCK_DIR:-}" 2>/dev/null || true
}

runtime_install_signal_traps() {
  trap 'runtime_cleanup' EXIT
  trap 'echo "[${RUNTIME_TAG}][INFO] interrupt received; stopping all runtime processes." >&2; exit 130' INT
  trap 'echo "[${RUNTIME_TAG}][INFO] termination received; stopping all runtime processes." >&2; exit 143' TERM HUP
  trap 'echo "[${RUNTIME_TAG}][INFO] Ctrl-Z received; stopping all runtime processes instead of leaving a suspended session." >&2; exit 148' TSTP
}
