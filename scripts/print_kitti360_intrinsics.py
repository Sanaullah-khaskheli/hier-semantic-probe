#!/usr/bin/env python3
import argparse
import numpy as np


def load_perspective_calib(calib_path):
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, vals = line.split(":", 1)
            data[key.strip()] = np.array([float(v) for v in vals.strip().split(" ") if v], dtype=np.float64)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-path", required=True, help="path to calibration/perspective.txt")
    ap.add_argument("--cam", default="00", choices=["00", "01"])
    args = ap.parse_args()

    data = load_perspective_calib(args.calib_path)
    P = data[f"P_rect_{args.cam}"].reshape(3, 4)
    S = data[f"S_rect_{args.cam}"].astype(int)
    fx, fy = P[0, 0], P[1, 1]
    cx, cy = P[0, 2], P[1, 2]
    width, height = int(S[0]), int(S[1])

    print(f"image_width: {width}")
    print(f"image_height: {height}")
    print(f"fx: {fx}")
    print(f"fy: {fy}")
    print(f"cx: {cx}")
    print(f"cy: {cy}")
    print()
    print("Paste these four numbers into configs/data/kitti360_semantic.yaml,")
    print("replacing the placeholder fx/fy/cx/cy values.")


if __name__ == "__main__":
    main()
