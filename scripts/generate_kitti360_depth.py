#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
from plyfile import PlyData
import cv2
from tqdm import tqdm


def load_perspective_calib(calib_path):
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            data[key.strip()] = np.array([float(v) for v in vals.strip().split(" ") if v], dtype=np.float64)
    P_rect_00 = data["P_rect_00"].reshape(3, 4)
    S_rect_00 = data["S_rect_00"].astype(int)
    K = P_rect_00[:3, :3]
    width, height = int(S_rect_00[0]), int(S_rect_00[1])
    return K, width, height


def load_cam0_to_world(path):
    poses = {}
    with open(path, "r") as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            if len(vals) != 17:
                continue
            frame_id = int(vals[0])
            T = np.array(vals[1:]).reshape(4, 4)
            poses[frame_id] = T
    return poses


def load_static_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"]
    x = np.asarray(v["x"]); y = np.asarray(v["y"]); z = np.asarray(v["z"])
    sem = np.asarray(v["semantic"]) if "semantic" in v.data.dtype.names else None
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    return pts, sem


def project_points_to_frame(points_world, T_cam0_to_world, K, width, height):
    T_world_to_cam0 = np.linalg.inv(T_cam0_to_world)
    R = T_world_to_cam0[:3, :3]
    t = T_world_to_cam0[:3, 3]
    pts_cam = (R @ points_world.T).T + t
    z = pts_cam[:, 2]
    valid = z > 1e-3
    pts_cam = pts_cam[valid]
    z = z[valid]
    uv = (K @ pts_cam.T).T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_bounds]; v = v[in_bounds]; z = z[in_bounds]
    return u, v, z, valid, in_bounds


def rasterize_depth(u, v, z, width, height, scale=1000):
    depth = np.zeros((height, width), dtype=np.float64)
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    order = np.argsort(-z)
    ui, vi, z_sorted = ui[order], vi[order], z[order]
    depth[vi, ui] = z_sorted
    depth_mm = np.clip(depth * scale, 0, 65535).astype(np.uint16)
    return depth_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti360-root", required=True)
    ap.add_argument("--sequence", required=True, help="e.g. 2013_05_28_drive_0000_sync")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scale", type=int, default=1000)
    args = ap.parse_args()

    calib_path = os.path.join(args.kitti360_root, "calibration", "perspective.txt")
    K, width, height = load_perspective_calib(calib_path)

    poses_path = os.path.join(args.kitti360_root, "data_poses", args.sequence, "cam0_to_world.txt")
    poses = load_cam0_to_world(poses_path)

    ply_glob = os.path.join(args.kitti360_root, "data_3d_semantics", "train", args.sequence, "static", "*.ply")
    ply_files = sorted(glob.glob(ply_glob))
    if not ply_files:
        raise RuntimeError(f"No static PLY files found at {ply_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_points = []
    all_sem = []
    for pf in tqdm(ply_files, desc="loading static point clouds"):
        pts, sem = load_static_ply(pf)
        all_points.append(pts)
        if sem is not None:
            all_sem.append(sem)
    all_points = np.concatenate(all_points, axis=0)
    all_sem = np.concatenate(all_sem, axis=0) if all_sem else None

    for frame_id, T in tqdm(sorted(poses.items()), desc="projecting frames"):
        u, v, z, valid, in_bounds = project_points_to_frame(all_points, T, K, width, height)
        depth_mm = rasterize_depth(u, v, z, width, height, scale=args.scale)
        out_path = os.path.join(args.out_dir, f"depth{frame_id:010d}.png")
        cv2.imwrite(out_path, depth_mm)


if __name__ == "__main__":
    main()
