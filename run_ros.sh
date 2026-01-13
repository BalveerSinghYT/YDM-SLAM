#!/usr/bin/env bash
set -euo pipefail

# Run the ROS1 all-in-one visualization stack (OctoMap + move_base demo + RViz).
# This launch expects YDM-SLAM to be running in another terminal and publishing:
#   /ydm/static_cloud (PointCloud2) + TF (map -> camera)
#
# Usage:
#   ./run_ros.sh
#   FRAME_ID=map ROBOT_BASE_FRAME=camera ./run_ros.sh
#   ./run_ros.sh frame_id:=map robot_base_frame:=camera

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[run_ros] $*"; }

die() {
  echo "[run_ros][ERROR] $*" >&2
  exit 1
}

# 1) Source ROS
if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f /opt/ros/noetic/setup.bash ]]; then
    export ROS_DISTRO=noetic
  elif [[ -f /opt/ros/melodic/setup.bash ]]; then
    export ROS_DISTRO=melodic
  else
    die "ROS_DISTRO is not set and no /opt/ros/noetic or /opt/ros/melodic found. Please: source /opt/ros/<distro>/setup.bash"
  fi
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
[[ -f "$ROS_SETUP" ]] || die "ROS setup not found: $ROS_SETUP"
# shellcheck disable=SC1090
source "$ROS_SETUP"

command -v roscore >/dev/null 2>&1 || die "roscore not found (did you install ROS1 and source it?)"
command -v roslaunch >/dev/null 2>&1 || die "roslaunch not found (did you install ROS1 and source it?)"

# 2) Source catkin workspace overlay (provides ydm_octomap_bridge / ydm_move_base_demo packages)
CATKIN_SETUP="$SCRIPT_DIR/ros_ws/devel/setup.bash"
if [[ -f "$CATKIN_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$CATKIN_SETUP"
else
  die "Catkin overlay not found: ros_ws/devel/setup.bash. Build it first: cd ros_ws && catkin_make"
fi

# 3) Launch
FRAME_ID="${FRAME_ID:-map}"
ROBOT_BASE_FRAME="${ROBOT_BASE_FRAME:-camera}"

# Optional: wait for SLAM to start publishing TF/PointCloud2 before launching costmaps.
WAIT_FOR_TF="${WAIT_FOR_TF:-1}"
WAIT_SECS="${WAIT_SECS:-15}"

log "ROS_DISTRO=${ROS_DISTRO}"
log "Launching: ydm_octomap_bridge/ydm_all_in_one.launch (frame_id=${FRAME_ID}, robot_base_frame=${ROBOT_BASE_FRAME})"
log "Hint: run SLAM in another terminal with: YDM_ROS=1 ./run_dynamic.sh"

if [[ "$WAIT_FOR_TF" == "1" ]]; then
  log "Waiting (up to ${WAIT_SECS}s) for SLAM TF (${FRAME_ID} -> ${ROBOT_BASE_FRAME}) and /ydm/static_cloud ..."
  start_ts=$(date +%s)
  while true; do
    now_ts=$(date +%s)
    if (( now_ts - start_ts >= WAIT_SECS )); then
      echo "[run_ros][WARN] Timed out waiting for TF/cloud. move_base costmaps may warn until SLAM publishes TF." >&2
      echo "[run_ros][WARN] Quick checks:" >&2
      echo "  - Is SLAM running with YDM_ROS=1 ?" >&2
      echo "  - Was YDM built with ROS: YDM_WITH_ROS=1 ./build.sh ?" >&2
      echo "  - rostopic list | grep -E 'ydm|tf'" >&2
      break
    fi

    have_cloud=0
    if rostopic list 2>/dev/null | grep -q '^/ydm/static_cloud$'; then
      have_cloud=1
    fi

    have_tf=0
    if command -v timeout >/dev/null 2>&1; then
      if timeout 0.8s rosrun tf tf_echo "$FRAME_ID" "$ROBOT_BASE_FRAME" >/dev/null 2>&1; then
        have_tf=1
      fi
    else
      # Fallback: only check /tf topic exists (less strict).
      if rostopic list 2>/dev/null | grep -q '^/tf$'; then
        have_tf=1
      fi
    fi

    if [[ "$have_cloud" == "1" && "$have_tf" == "1" ]]; then
      log "Detected /ydm/static_cloud and TF (${FRAME_ID} -> ${ROBOT_BASE_FRAME})."
      break
    fi

    sleep 0.2
  done
fi

exec roslaunch ydm_octomap_bridge ydm_all_in_one.launch \
  frame_id:="$FRAME_ID" \
  robot_base_frame:="$ROBOT_BASE_FRAME" \
  "$@"
