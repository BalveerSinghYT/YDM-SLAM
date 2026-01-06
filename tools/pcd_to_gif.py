#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
from typing import List, Optional, Tuple


def _natural_key(path: str):
    # Sort like kf_000020.pcd, kf_000100.pcd
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def _resolve_inputs(inp: str, pattern: Optional[str]) -> List[str]:
    if pattern:
        paths = glob.glob(pattern)
        return sorted(paths, key=_natural_key)

    if os.path.isdir(inp):
        paths = glob.glob(os.path.join(inp, "*.pcd"))
        return sorted(paths, key=_natural_key)

    # Treat as glob
    paths = glob.glob(inp)
    return sorted(paths, key=_natural_key)


def _compute_global_bbox(o3d, paths: List[str], stride: int, max_frames_for_bbox: int) -> Tuple["o3d.geometry.AxisAlignedBoundingBox", int]:
    import numpy as np

    bbox = None
    used = 0

    step = max(1, stride)
    for i, p in enumerate(paths[::step]):
        if used >= max_frames_for_bbox:
            break
        pc = o3d.io.read_point_cloud(p)
        if pc.is_empty():
            continue
        b = pc.get_axis_aligned_bounding_box()
        if bbox is None:
            bbox = b
        else:
            # NOTE: Open3D CUDA AxisAlignedBoundingBox does not support operator+.
            # Compute union manually so this works for both CPU and CUDA builds.
            bb_min = np.asarray(bbox.get_min_bound())
            bb_max = np.asarray(bbox.get_max_bound())
            b_min = np.asarray(b.get_min_bound())
            b_max = np.asarray(b.get_max_bound())
            minb = np.minimum(bb_min, b_min)
            maxb = np.maximum(bb_max, b_max)
            try:
                bbox = type(bbox)(minb, maxb)
            except Exception:
                bbox = o3d.geometry.AxisAlignedBoundingBox(minb, maxb)
        used += 1

    if bbox is None:
        bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])

    return bbox, used


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render many .pcd files into an animated GIF (or a PNG sequence).\n"
        "Default behavior assumes pcd/kf_*.pcd saved by this repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input",
        default="pcd",
        help="Input folder with .pcd, or a glob like 'pcd/kf_*.pcd'",
    )
    ap.add_argument(
        "--glob",
        default=None,
        help="Explicit glob pattern (overrides --input)",
    )
    ap.add_argument("--output", default="pcd.gif", help="Output GIF path")
    ap.add_argument("--fps", type=float, default=10.0, help="GIF frames per second")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth PCD file")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--voxel", type=float, default=0.0, help="Voxel downsample size in meters (0 disables)")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit number of frames rendered (0 = no limit)",
    )
    ap.add_argument(
        "--write-frames",
        default=None,
        help="If set, also write PNG frames to this directory (useful for ffmpeg)",
    )
    ap.add_argument(
        "--max-frames-for-bbox",
        type=int,
        default=60,
        help="How many frames to sample (after stride) to estimate global bounding box",
    )

    args = ap.parse_args()

    try:
        import numpy as np
        import imageio.v2 as imageio
        import open3d as o3d
        from open3d.visualization import rendering
    except Exception as e:
        print("ERROR: Missing Python dependencies.")
        print("Install with: pip install open3d imageio numpy")
        print("Details:", repr(e))
        return 2

    paths = _resolve_inputs(args.input, args.glob)
    if not paths:
        print("ERROR: No .pcd files found.")
        print(f"Tried input={args.input!r} glob={args.glob!r}")
        return 2

    step = max(1, int(args.stride))
    selected = paths[::step]
    if args.max_frames and args.max_frames > 0:
        selected = selected[: args.max_frames]

    bbox, used_bbox = _compute_global_bbox(o3d, selected, stride=1, max_frames_for_bbox=args.max_frames_for_bbox)
    center = bbox.get_center()
    extent = bbox.get_extent()
    diag = float(np.linalg.norm(extent))
    if diag <= 1e-6:
        diag = 2.0

    # Camera: fixed viewpoint based on global bbox
    # eye at some offset so the cloud is visible and stable frame-to-frame.
    eye = center + np.array([0.0, -diag * 1.6, diag * 0.8])
    up = np.array([0.0, 0.0, 1.0])

    if args.write_frames:
        os.makedirs(args.write_frames, exist_ok=True)

    renderer = rendering.OffscreenRenderer(args.width, args.height)
    try:
        renderer.scene.set_background([0, 0, 0, 1])
    except Exception:
        pass

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(args.point_size)

    # Reasonable near/far based on bbox
    near = max(0.01, diag * 0.01)
    far = diag * 10.0
    fov = 60.0
    renderer.scene.camera.set_projection(fov, args.width / args.height, near, far, rendering.Camera.FovType.Vertical)
    renderer.scene.camera.look_at(center, eye, up)

    frames = []
    for idx, p in enumerate(selected):
        pc = o3d.io.read_point_cloud(p)
        if pc.is_empty():
            continue

        if args.voxel and args.voxel > 0:
            pc = pc.voxel_down_sample(voxel_size=float(args.voxel))

        # Replace geometry per-frame
        name = "pcd"
        try:
            renderer.scene.remove_geometry(name)
        except Exception:
            pass
        renderer.scene.add_geometry(name, pc, mat)

        img = renderer.render_to_image()
        arr = np.asarray(img)

        if args.write_frames:
            out_png = os.path.join(args.write_frames, f"frame_{idx:05d}.png")
            imageio.imwrite(out_png, arr)

        frames.append(arr)

        # Basic progress
        if (idx + 1) % 25 == 0:
            print(f"Rendered {idx+1}/{len(selected)}")

    if not frames:
        print("ERROR: Rendered 0 frames (empty point clouds?)")
        return 2

    duration = 1.0 / float(args.fps) if args.fps > 0 else 0.1
    imageio.mimsave(args.output, frames, duration=duration)

    print(f"Done: wrote {args.output} ({len(frames)} frames, fps={args.fps}, stride={args.stride}, bbox_samples={used_bbox})")

    if args.write_frames:
        print("Tip (MP4): ffmpeg -y -framerate 10 -i %s/frame_%%05d.png -pix_fmt yuv420p out.mp4" % args.write_frames)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
