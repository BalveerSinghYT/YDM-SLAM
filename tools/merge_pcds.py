#!/usr/bin/env python3

import argparse
import glob
import os
import re
from typing import List


def _natural_key(path: str):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def _resolve_inputs(inp: str, pattern: str | None) -> List[str]:
    if pattern:
        paths = glob.glob(pattern)
        return sorted(paths, key=_natural_key)

    if os.path.isdir(inp):
        paths = glob.glob(os.path.join(inp, "*.pcd"))
        return sorted(paths, key=_natural_key)

    paths = glob.glob(inp)
    return sorted(paths, key=_natural_key)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Merge many world-frame PCDs (pcd/kf_*.pcd) into one global point cloud.\n"
        "This is an offline alternative to the in-run global merge (which may be unstable on some PCL builds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", default="pcd", help="Folder with PCDs, or a glob like 'pcd/kf_*.pcd'")
    ap.add_argument("--glob", default=None, help="Explicit glob pattern (overrides --input)")
    ap.add_argument("--output", default="merged.pcd", help="Output point cloud file (.pcd or .ply)")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth file")
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of input files (0 = no limit)")
    ap.add_argument("--voxel", type=float, default=0.03, help="Voxel size in meters (0 disables)")

    args = ap.parse_args()

    try:
        import open3d as o3d
    except Exception as e:
        print("ERROR: open3d not installed.")
        print("Install with: pip install open3d")
        print("Details:", repr(e))
        return 2

    paths = _resolve_inputs(args.input, args.glob)
    if not paths:
        print("ERROR: No PCD files found.")
        return 2

    step = max(1, int(args.stride))
    selected = paths[::step]
    if args.max_files and args.max_files > 0:
        selected = selected[: args.max_files]

    merged = o3d.geometry.PointCloud()
    used = 0
    for i, p in enumerate(selected):
        pc = o3d.io.read_point_cloud(p)
        if pc.is_empty():
            continue
        merged += pc
        used += 1
        if (i + 1) % 50 == 0:
            print(f"Loaded {i+1}/{len(selected)}")

    if used == 0 or merged.is_empty():
        print("ERROR: Merged cloud is empty.")
        return 2

    if args.voxel and args.voxel > 0:
        merged = merged.voxel_down_sample(float(args.voxel))

    ok = o3d.io.write_point_cloud(args.output, merged, write_ascii=False, compressed=True)
    if not ok:
        print("ERROR: Failed to write output.")
        return 2

    print(f"Done: wrote {args.output} (inputs_used={used}, points={len(merged.points)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
