#!/usr/bin/env bash
# Source this file after sourcing /opt/ros/<distro>/setup.bash.
# It isolates the local direct-control/RViz debug graph from the LAN.
#
# Optional override:
#   REVO2_RVIZ_ROS_DOMAIN_ID=124  # default: 124, valid ROS 2 range is 0..232
#
# Do not execute this script directly; source it from a shell or launcher.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Usage: source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

_REVO2_DOMAIN="${REVO2_RVIZ_ROS_DOMAIN_ID:-124}"
if ! [[ "${_REVO2_DOMAIN}" =~ ^[0-9]+$ ]] || (( _REVO2_DOMAIN < 0 || _REVO2_DOMAIN > 232 )); then
  echo "[revo2-rviz][ERROR] REVO2_RVIZ_ROS_DOMAIN_ID must be an integer in 0..232; got '${_REVO2_DOMAIN}'." >&2
  return 2
fi

# ROS 2 supported graph-isolation switches. Every process started by the local
# launcher inherits them, so discovery and data traffic stay on loopback.
export ROS_DOMAIN_ID="${_REVO2_DOMAIN}"
export ROS_LOCALHOST_ONLY=1
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# Do not inherit a company discovery server or a vendor-specific DDS profile.
# Either may reconnect this RViz session to the intranet despite a private domain.
unset ROS_DISCOVERY_SERVER
unset FASTDDS_DISCOVERY_SERVER
unset FASTRTPS_DEFAULT_PROFILES_FILE
unset FASTDDS_DEFAULT_PROFILES_FILE
unset CYCLONEDDS_URI

export REVO2_RVIZ_LOCAL_ONLY=1
printf '[revo2-rviz][INFO] local-only ROS graph: domain=%s rmw=%s localhost_only=%s discovery=%s\n' \
  "${ROS_DOMAIN_ID}" "${RMW_IMPLEMENTATION:-system-default}" "${ROS_LOCALHOST_ONLY}" "${ROS_AUTOMATIC_DISCOVERY_RANGE}"
unset _REVO2_DOMAIN
