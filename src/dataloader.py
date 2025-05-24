from nuscenes.nuscenes import NuScenes
import random
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from depth_pro.inference import estimateDepth
from edgemask import generate_edge_mask

DATA_ROOT = './calibration_data/imgs'

nusc = NuScenes(version='v1.0-mini', dataroot='./calibration_data/sets/v1.0-mini', verbose=True)

def random_scene_sample():
    random_scene = random.choice(nusc.scene)
    print(f"Random Scene Name: {random_scene['name']}")
    sample_token = random_scene['first_sample_token']
    return sample_token, random_scene['name']

def copy_data(sample_token, scene_name):
    camera_tokens = [
        'CAM_FRONT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT'
    ]
    scene_output_dir = os.path.join(DATA_ROOT, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    sample_id = 0
    while sample_token:
        sample = nusc.get('sample', sample_token)
        sample_output_dir = os.path.join(scene_output_dir, f"sample_{sample_id}")
        os.makedirs(sample_output_dir, exist_ok=True)
        sample_depth_dir = os.path.join(sample_output_dir, 'depth_imgs')
        os.makedirs(sample_depth_dir, exist_ok=True)

        lidar_token = sample['data']['LIDAR_TOP']
        filename = os.path.join(nusc.dataroot, nusc.get('sample_data', lidar_token)['filename'])
        shutil.copy(filename, os.path.join(scene_output_dir, os.path.basename(filename)))

        for i, channel in enumerate(camera_tokens):
            token = sample['data'][channel]
            record = nusc.get('sample_data', token)
            src_fp = os.path.join(nusc.dataroot, record['filename'])
            dst_fp = os.path.join(sample_output_dir, os.path.basename(record['filename']))
            print(f"Copying {src_fp} → {dst_fp}")
            shutil.copy(src_fp, dst_fp)

            depth, img = estimateDepth(dst_fp, DATA_ROOT = sample_depth_dir)
            sample_edge_dir = os.path.join(sample_output_dir, 'edgemasks')
            os.makedirs(sample_edge_dir, exist_ok=True)
            generate_edge_mask(img, kernel_size=10, show=False, save_path=os.path.join(sample_edge_dir, os.path.basename(dst_fp)))
        sample_token = None
        sample_id += 1

def plot_random_scene_sample(scene_name, sample_id):
    fig, axes = plt.subplots(3, 6, figsize=(36, 12))  # 3 rows: original, depth, edge mask
    sample_dir = os.path.join(DATA_ROOT, scene_name, f"sample_{sample_id}")
    depth_dir = os.path.join(sample_dir, 'depth_imgs')
    edgemask_dir = os.path.join(sample_dir, 'edgemasks')

    if not os.path.exists(sample_dir) or not os.path.exists(depth_dir) or not os.path.exists(edgemask_dir):
        print(f"One or more required directories are missing:\n"
              f" - Sample: {sample_dir}\n - Depth: {depth_dir}\n - Edgemasks: {edgemask_dir}")
        return

    original_imgs = sorted([f for f in os.listdir(sample_dir) if f.endswith('.jpg') and '__' in f])
    
    for i, img_file in enumerate(original_imgs[:6]):
        orig_path = os.path.join(sample_dir, img_file)
        depth_path = os.path.join(depth_dir, img_file)
        edge_path = os.path.join(edgemask_dir, img_file)

        # Extract clean camera channel name (e.g., CAM_FRONT)
        channel = img_file.split('__')[1].split('.')[0]  # Removes extension too

        # Row 0: Original
        img_orig = mpimg.imread(orig_path)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f"{channel} (Original)", fontsize=10)
        axes[0, i].axis('off')

        # Row 1: Depth
        if os.path.exists(depth_path):
            img_depth = mpimg.imread(depth_path)
            axes[1, i].imshow(img_depth, cmap='plasma')
            axes[1, i].set_title("Depth", fontsize=10)
        else:
            axes[1, i].text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=12)
        axes[1, i].axis('off')

        # Row 2: Edge Mask
        if os.path.exists(edge_path):
            img_edge = mpimg.imread(edge_path)
            axes[2, i].imshow(img_edge, cmap='gray')
            axes[2, i].set_title("Edge Mask", fontsize=10)
        else:
            axes[2, i].text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=12)
        axes[2, i].axis('off')

    plt.suptitle(f"Sample {sample_id} - Scene {scene_name}", fontsize=18)
    plt.tight_layout()
    plt.show()

    
sample_token, scene_name = random_scene_sample()
copy_data(sample_token, scene_name)
plot_random_scene_sample(scene_name, 0)

