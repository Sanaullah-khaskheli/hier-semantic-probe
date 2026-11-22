import glob
import os
from typing import Optional

import cv2
import imageio
import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset, as_intrinsics_matrix
from . import datautils

KITTI360_LABELS = [
    (0,  "unlabeled",       "void"),
    (1,  "ego vehicle",     "void"),
    (2,  "rectification border", "void"),
    (3,  "out of roi",      "void"),
    (4,  "static",          "void"),
    (5,  "dynamic",         "void"),
    (6,  "ground",          "flat"),
    (7,  "road",            "flat"),
    (8,  "sidewalk",        "flat"),
    (9,  "parking",         "flat"),
    (10, "rail track",      "flat"),
    (11, "building",        "construction"),
    (12, "wall",            "construction"),
    (13, "fence",           "construction"),
    (14, "guard rail",      "construction"),
    (15, "bridge",          "construction"),
    (16, "tunnel",          "construction"),
    (17, "pole",            "object"),
    (18, "polegroup",       "object"),
    (19, "traffic light",   "object"),
    (20, "traffic sign",    "object"),
    (21, "vegetation",      "nature"),
    (22, "terrain",         "nature"),
    (23, "sky",             "sky"),
    (24, "person",          "human"),
    (25, "rider",           "human"),
    (26, "car",             "vehicle"),
    (27, "truck",           "vehicle"),
    (28, "bus",             "vehicle"),
    (29, "caravan",         "vehicle"),
    (30, "trailer",         "vehicle"),
    (31, "train",           "vehicle"),
    (32, "motorcycle",      "vehicle"),
    (33, "bicycle",         "vehicle"),
    (34, "garage",          "construction"),
    (35, "gate",            "construction"),
    (36, "stop",            "construction"),
    (37, "smallpole",       "object"),
    (38, "lamp",            "object"),
    (39, "trash bin",       "object"),
    (40, "vending machine", "object"),
    (41, "box",             "object"),
    (42, "unknown construction", "construction"),
    (43, "unknown vehicle", "vehicle"),
    (44, "unknown object",  "object"),
]
IGNORE_IDS = {0, 1, 2, 3, 4, 5}


def build_kitti360_tree(tree_level=2):
    rows = [(i, n, c) for (i, n, c) in KITTI360_LABELS if i not in IGNORE_IDS]
    categories = sorted(set(c for _, _, c in rows))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    leaf_ids_sorted = sorted(i for i, _, _ in rows)
    leaf_id_to_idx = {i: idx for idx, i in enumerate(leaf_ids_sorted)}

    label_mapping_tree = {}
    for i, name, cat in rows:
        if tree_level == 2:
            label_mapping_tree[i] = (cat_to_idx[cat], leaf_id_to_idx[i])
        else:
            raise NotImplementedError("Only tree_level=2 (category, leaf) is implemented for KITTI-360.")

    tree_id_classes_map = [
        {idx: cat for cat, idx in cat_to_idx.items()},
        {idx: KITTI360_LABELS[[r[0] for r in rows].index(i)][1] for i, idx in leaf_id_to_idx.items()},
    ]
    num_level_list = [len(categories), len(leaf_ids_sorted)]
    return label_mapping_tree, tree_id_classes_map, num_level_list, leaf_id_to_idx


class KITTI360Dataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 376,
        desired_width: Optional[int] = 1408,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        camera: str = "image_00",
        **kwargs,
    ):
        self.sequence = sequence
        self.camera = camera
        self.input_folder_rgb = os.path.join(basedir, "data_2d_raw", sequence, camera, "data_rect")
        self.input_folder_depth = os.path.join(basedir, "data_2d_depth_proj", sequence, camera)
        self.pose_path = os.path.join(basedir, "data_poses", sequence, "cam0_to_world.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder_rgb}/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder_depth}/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder_rgb}/../{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        pose_by_frame = {}
        with open(self.pose_path, "r") as f:
            for line in f:
                vals = [float(v) for v in line.strip().split()]
                if len(vals) != 17:
                    continue
                frame_id = int(vals[0])
                c2w = np.array(vals[1:]).reshape(4, 4)
                pose_by_frame[frame_id] = torch.from_numpy(c2w).float()

        poses = []
        for p in self.color_paths:
            frame_id = int(os.path.splitext(os.path.basename(p))[0])
            if frame_id not in pose_by_frame:
                raise KeyError(
                    f"No pose for frame {frame_id} in {self.pose_path}. "
                    f"KITTI-360 poses are only given for a subset of frames; "
                    f"filter color_paths to frames present in cam0_to_world.txt before calling this loader."
                )
            poses.append(pose_by_frame[frame_id])
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)


class KITTI360Dataset_semantic(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 376,
        desired_width: Optional[int] = 1408,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        camera: str = "image_00",
        **kwargs,
    ):
        self.sem_mode = config_dict["sem_mode"]
        self.num_tree_level = config_dict.get("num_tree_level", 2)
        self.camera = camera
        self.sequence = sequence

        self.input_folder_rgb = os.path.join(basedir, "data_2d_raw", sequence, camera, "data_rect")
        self.input_folder_depth = os.path.join(basedir, "data_2d_depth_proj", sequence, camera)
        self.input_folder_sem = os.path.join(basedir, "data_2d_semantics", "train", sequence, camera, "semantic")
        self.pose_path = os.path.join(basedir, "data_poses", sequence, "cam0_to_world.txt")

        self.semantic_paths = natsorted(glob.glob(f"{self.input_folder_sem}/*.png"))

        if self.sem_mode == "original":
            all_ids = sorted(set(i for i, _, _ in KITTI360_LABELS) - IGNORE_IDS)
            self.num_semantic = len(all_ids)
            self.num_semantic_class = self.num_semantic
            self.id_remap = {old: new for new, old in enumerate(all_ids)}
        elif "tree" in self.sem_mode:
            (self.label_mapping_tree, self.tree_id_classes_map,
             num_level_list, self.leaf_id_to_idx) = build_kitti360_tree(self.num_tree_level)
            self.num_semantic = num_level_list
            self.num_semantic_class = num_level_list[-1]
        else:
            raise ValueError(f"Unknown sem_mode {self.sem_mode}")

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

        sem_by_frame = {
            int(os.path.splitext(os.path.basename(p))[0]): p for p in self.semantic_paths
        }
        self.semantic_paths = []
        for p in self.color_paths:
            frame_id = int(os.path.splitext(os.path.basename(p))[0])
            if frame_id not in sem_by_frame:
                raise KeyError(
                    f"No 2D semantic label for frame {frame_id}. "
                    f"KITTI-360's data_2d_semantics only covers a subset of frames "
                    f"(see data_2d_semantics/train/*_train_frames.txt); "
                    f"filter color_paths/poses to that subset before using this loader."
                )
            self.semantic_paths.append(sem_by_frame[frame_id])

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder_rgb}/*.png"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder_depth}/depth*.png"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder_rgb}/../{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        pose_by_frame = {}
        with open(self.pose_path, "r") as f:
            for line in f:
                vals = [float(v) for v in line.strip().split()]
                if len(vals) != 17:
                    continue
                frame_id = int(vals[0])
                c2w = np.array(vals[1:]).reshape(4, 4)
                pose_by_frame[frame_id] = torch.from_numpy(c2w).float()

        poses = []
        for p in self.color_paths:
            frame_id = int(os.path.splitext(os.path.basename(p))[0])
            if frame_id not in pose_by_frame:
                raise KeyError(f"No pose for frame {frame_id} in {self.pose_path}")
            poses.append(pose_by_frame[frame_id])
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)

    def _preprocess_semantic_label(self, label: np.ndarray):
        label = cv2.resize(label, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)
        return label

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)

        sem_raw = np.asarray(imageio.imread(self.semantic_paths[index]), dtype=np.int64)
        sem_raw = self._preprocess_semantic_label(sem_raw)

        if self.sem_mode == "original":
            remap = np.vectorize(lambda v: self.id_remap.get(v, 0))
            semantic = remap(sem_raw).astype(np.int64)
            semantic = np.expand_dims(semantic, -1)
        else:
            level0 = np.zeros_like(sem_raw)
            level1 = np.zeros_like(sem_raw)
            for base_id, (cat_idx, leaf_idx) in self.label_mapping_tree.items():
                mask = sem_raw == base_id
                level0[mask] = cat_idx
                level1[mask] = leaf_idx
            semantic = np.stack([level0, level1], axis=0)

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        color = torch.from_numpy(color)
        K = torch.from_numpy(K)
        semantic = torch.from_numpy(semantic)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]

        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return (
                color.to(self.device).type(self.dtype),
                depth.to(self.device).type(self.dtype),
                intrinsics.to(self.device).type(self.dtype),
                pose.to(self.device).type(self.dtype),
                semantic.to(self.device),
                embedding.to(self.device),
            )

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            semantic.to(self.device),
        )
