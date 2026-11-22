#!/usr/bin/env python3
import re
import sys
import os


def patch_file(path):
    with open(path, "r") as f:
        src = f.read()
    orig = src

    src = re.sub(
        r'AT_ERROR\(\s*"([^"]*)"\s*\)',
        r'TORCH_CHECK(false, "\1")',
        src,
    )

    src = re.sub(
        r'\.data<([a-zA-Z_0-9:]+)>\(\)',
        r'.data_ptr<\1>()',
        src,
    )

    if src != orig:
        with open(path, "w") as f:
            f.write(src)
        print(f"patched: {path}")
    else:
        print(f"no change needed: {path}")


def main():
    if len(sys.argv) < 2:
        print("usage: patch_cuda_extension.py <path-to-hierslam-diff-gaussian-rasterization-w-depth>")
        sys.exit(1)
    root = sys.argv[1]
    targets = [
        os.path.join(root, "rasterize_points.cu"),
        os.path.join(root, "cuda_rasterizer", "forward.cu"),
        os.path.join(root, "cuda_rasterizer", "backward.cu"),
        os.path.join(root, "cuda_rasterizer", "rasterizer_impl.cu"),
    ]
    for t in targets:
        if os.path.exists(t):
            patch_file(t)
        else:
            print(f"skip (not found): {t}")


if __name__ == "__main__":
    main()
