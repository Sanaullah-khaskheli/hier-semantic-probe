HierSLAM on KITTI-360 — port kit (unverified, no GPU/data in build sandbox)
============================================================================

What this is
------------
A set of files meant to drop into a clone of the real Hier-SLAM repo
(https://github.com/LeeBY68/Hier-SLAM) so it runs on:
  1. a newer PyTorch/CUDA stack (2.1.2 / cu121) instead of the pinned
     1.12.1 / cudatoolkit 11.6.2, and
  2. KITTI-360 instead of Replica/ScanNet.

This was built with no GPU and no real KITTI-360 data available in the
build sandbox. Nothing here has been run end-to-end against the real
repo. Treat every file as a first draft that needs to be checked once
you have both a GPU and real data.

How to use this kit
--------------------
1. `git clone https://github.com/LeeBY68/Hier-SLAM.git`
2. Copy this kit's files into the clone, matching the paths:
   - `datasets/gradslam_datasets/kitti360.py` -> same path in the clone
   - `configs/data/kitti360_semantic.yaml` -> same path
   - `configs/kitti360/hierslam_semantic_run.py` -> same path
   - `scripts/generate_kitti360_depth.py` -> same path (new file)
   - `scripts/print_kitti360_intrinsics.py` -> same path (new file)
3. Register `KITTI360Dataset` / `KITTI360Dataset_semantic` in whatever
   dataset-name dispatch the real repo uses (I did not find and have not
   patched that dispatch point — see "Known gaps" below).
4. Run `scripts/print_kitti360_intrinsics.py --calib-path <your KITTI-360>/calibration/perspective.txt`
   and paste the real fx/fy/cx/cy into `configs/data/kitti360_semantic.yaml`
   — the shipped values are placeholders and are almost certainly wrong
   for your actual rectified images.
5. Run `scripts/generate_kitti360_depth.py --kitti360-root <root> --sequence <seq> --out-dir <root>/data_2d_depth_proj/<seq>/image_00`
   once per camera/sequence to build the depth maps `kitti360.py` expects
   under `data_2d_depth_proj/...` (KITTI-360 has no RGB-D sensor; depth is
   synthesized by projecting the accumulated 3D semantic point cloud into
   each pose, z-buffered, nearest-wins).
6. Patch the CUDA rasterizer extension for the newer PyTorch:
   `python patches/patch_cuda_extension.py <clone>/hierslam-diff-gaussian-rasterization-w-depth`
   This rewrites `AT_ERROR(...)` -> `TORCH_CHECK(false, ...)` and
   `.data<T>()` -> `.data_ptr<T>()`. It only touches the 4 files it looks
   for; if the real repo has other `.cu`/`.cpp` files using those same
   deprecated APIs, grep for `AT_ERROR` and `.data<` yourself before
   building.
7. Replace the repo's `environment.yml`/`requirements.txt` with this
   kit's versions (torch 2.1.2+cu121 instead of 1.12.1), rebuild the CUDA
   extension (`pip install -e .` inside the rasterizer directory) with
   a CUDA 12.1-compatible nvcc/gcc, and run
   `python scripts/run_kitti360.py` — i.e. whatever entry point the real
   repo uses with `configs/kitti360/hierslam_semantic_run.py` as the config
   argument (I have not located and copied that entry-point script itself;
   use the same one the repo's own `configs/replica/*.py` configs are run
   with).

Known gaps / what is NOT done
------------------------------
- The dataset-name dispatch (wherever the repo maps a string like
  "replica" to `ReplicaDataset_semantic`) has not been found/patched to
  also recognize "kitti360". You need to find that switch yourself and
  add a branch for `KITTI360Dataset_semantic`.
- The compact hierarchical embedding width (`NUM_SEMANTIC` in
  `cuda_rasterizer/config.h`) is hardcoded per-dataset in the real repo
  (26 for Replica's 5-level tree, 16/74 for ScanNet). For a 2-level
  KITTI-360 tree built in `kitti360.py` the correct channel count has NOT
  been derived — I have not located the actual Gaussian/scene-model code
  that consumes this embedding, so I can't tell you what number to put
  there. You will need to find that file in the real repo, read how the
  embedding width is computed from the tree, and set `NUM_SEMANTIC`
  accordingly before the CUDA extension will produce correctly-shaped
  tensors for KITTI-360.
- `scripts/generate_kitti360_depth.py` is untested — there was no real
  KITTI-360 PLY/pose data in the build sandbox to run it against. Sanity
  check the projected depth maps visually before trusting them.
- `datasets/gradslam_datasets/kitti360.py` is untested against a real
  KITTI-360 directory tree; file-glob patterns for color/depth/semantic
  paths are my best reconstruction from the documented KITTI-360 folder
  layout, not verified against an actual download.
- The CUDA patch script only does two mechanical substitutions; it does
  not attempt to fix any other PyTorch-2.x ABI breaks (e.g. changed
  `at::Tensor` constructor signatures, changed extension-building flags)
  that may surface only at compile time on the newer stack.
- Camera intrinsics in `configs/data/kitti360_semantic.yaml` are
  placeholders — see step 4 above.

In short: this gets you most of the mechanical plumbing (dataset class,
run config, dependency pins, a depth-synthesis script, a CUDA API patch)
but the two pieces that need the real repo's source in front of you —
the dataset-name dispatch switch and the `NUM_SEMANTIC` embedding-width
for a KITTI-360 tree — are flagged, not solved, because I could not find
that code in what I fetched of the repo.
