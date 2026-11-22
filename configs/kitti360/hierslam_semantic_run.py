scene_name = "2013_05_28_drive_0000_sync"
seed = 0
group_name = "KITTI360_semantic"
run_name = f"{scene_name}_{seed}"

data = dict(
    basedir="/path/to/KITTI-360",
    sequence=scene_name,
    gradslam_data_cfg="./configs/data/kitti360_semantic.yaml",
    sem_mode="tree",
    num_tree_level=2,
    use_pyramid=False,
    pyramid_level=4,
    desired_image_height=376,
    desired_image_width=1408,
    start=0,
    end=-1,
    stride=1,
    num_frames=-1,
)

model = dict(
    flag_use_embedding=1,
    eval_gt_transfer=False,
)

mapping = dict(
    lrs=dict(semantic=0.05),
    loss_weights=dict(sem=0.2),
    map_every=8,
    keyframe_every=5,
    mapping_window_size=24,
    mapping_iters=60,
)

tracking = dict(
    lrs=dict(semantic=0.0),
    tracking_iters=40,
)

results_dir = f"./results/kitti360/{run_name}"
