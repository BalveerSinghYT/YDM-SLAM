#!/usr/bin/env bash
set -euo pipefail

# Run from repo root so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage: ./run_evo.sh [gt] [est] [kf] [out_dir] [format]

Defaults:
  gt      Txt/groundtruth.txt
  est     Txt/CameraTrajectory.txt
  kf      Txt/KeyFrameTrajectory.txt
  out_dir evaluation/evo
  format  tum

Optional env vars:
  EVO_PLOT_MODE=xyz   # xyz|xy|xz|yz
  EVO_T_MAX_DIFF=0.02
  EVO_T_OFFSET=0.0
  EVO_SYNC=1          # 1 to sync to --ref (gt), 0 to skip
  EVO_SHOW=0          # 1 to open plot windows
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

GT="${1:-Txt/groundtruth.txt}"
EST="${2:-Txt/CameraTrajectory.txt}"
KF="${3:-Txt/KeyFrameTrajectory.txt}"
OUT_DIR="${4:-evaluation/evo}"
FORMAT="${5:-tum}"

if ! command -v evo_ape >/dev/null 2>&1; then
  echo "ERROR: evo_ape not found in PATH." >&2
  exit 1
fi
if ! command -v evo_traj >/dev/null 2>&1; then
  echo "ERROR: evo_traj not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$GT" ]]; then
  echo "ERROR: groundtruth not found: $GT" >&2
  exit 1
fi
if [[ ! -f "$EST" ]]; then
  echo "ERROR: estimate not found: $EST" >&2
  exit 1
fi

PLOT_MODE="${EVO_PLOT_MODE:-xyz}"
T_MAX_DIFF="${EVO_T_MAX_DIFF:-}"
T_OFFSET="${EVO_T_OFFSET:-}"
SYNC="${EVO_SYNC:-1}"
SHOW="${EVO_SHOW:-0}"

mkdir -p "$OUT_DIR"

ape_common=(-a --plot_mode "$PLOT_MODE")
if [[ "$SHOW" == "1" ]]; then
  ape_common+=(--plot)
fi
if [[ -n "$T_MAX_DIFF" ]]; then
  ape_common+=(--t_max_diff "$T_MAX_DIFF")
fi
if [[ -n "$T_OFFSET" ]]; then
  ape_common+=(--t_offset "$T_OFFSET")
fi

evo_ape "$FORMAT" "$GT" "$EST" \
  "${ape_common[@]}" \
  --save_plot "$OUT_DIR/ape_camera.png" \
  --save_results "$OUT_DIR/ape_camera.zip"

if [[ -f "$KF" ]]; then
  evo_ape "$FORMAT" "$GT" "$KF" \
    "${ape_common[@]}" \
    --save_plot "$OUT_DIR/ape_keyframe.png" \
    --save_results "$OUT_DIR/ape_keyframe.zip"
fi

traj_files=("$GT" "$EST")
if [[ -f "$KF" ]]; then
  traj_files+=("$KF")
fi

traj_common=(-a --plot_mode "$PLOT_MODE")
if [[ "$SHOW" == "1" ]]; then
  traj_common+=(--plot)
fi
if [[ "$SYNC" == "1" ]]; then
  traj_common+=(--ref "$GT" --sync)
fi
if [[ -n "$T_MAX_DIFF" ]]; then
  traj_common+=(--t_max_diff "$T_MAX_DIFF")
fi
if [[ -n "$T_OFFSET" ]]; then
  traj_common+=(--t_offset "$T_OFFSET")
fi

evo_traj "$FORMAT" "${traj_files[@]}" \
  "${traj_common[@]}" \
  --save_plot "$OUT_DIR/traj_${PLOT_MODE}.png"

# Also export a 2D top-down plot for quick inspection.
traj_xy=(-a --plot_mode xy)
if [[ "$SHOW" == "1" ]]; then
  traj_xy+=(--plot)
fi
if [[ "$SYNC" == "1" ]]; then
  traj_xy+=(--ref "$GT" --sync)
fi
if [[ -n "$T_MAX_DIFF" ]]; then
  traj_xy+=(--t_max_diff "$T_MAX_DIFF")
fi
if [[ -n "$T_OFFSET" ]]; then
  traj_xy+=(--t_offset "$T_OFFSET")
fi

evo_traj "$FORMAT" "${traj_files[@]}" \
  "${traj_xy[@]}" \
  --save_plot "$OUT_DIR/traj_xy.png"
