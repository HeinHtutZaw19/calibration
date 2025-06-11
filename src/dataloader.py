from nuscenes.nuscenes import NuScenes
import random
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from depth_pro.inference import estimateDepth
from edgemask import generate_edge_mask
import numpy as np
from plot import *

DATA_ROOT = './calibration_data/imgs'

nusc = NuScenes(version='v1.0-mini', dataroot='./calibration_data/sets/v1.0-mini', verbose=True)

def random_scene_sample():
    random_scene = random.choice(nusc.scene)

    print(f"Random Scene Name: {random_scene['token']}")
    sample_token = random_scene['first_sample_token']
    return sample_token, random_scene['token']

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


def process_scene(scene, init_id):
    scene_name = scene['token']
    print(f"Processing Scene: {scene_name} with Sample ID: {init_id}")
    scene_output_dir = os.path.join(DATA_ROOT, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    sample_id = 0
    sample_token = scene['first_sample_token']
    
    while sample_token and sample_id < init_id:
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']
        sample_id += 1

    while sample_token:
        sample = nusc.get('sample', sample_token)
        sample_output_dir = os.path.join(scene_output_dir, f"sample_{sample_id}")
        os.makedirs(sample_output_dir, exist_ok=True)

        # Copy LIDAR_TOP data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_record = nusc.get('sample_data', lidar_token)
        lidar_src = os.path.join(nusc.dataroot, lidar_record['filename'])
        shutil.copy(lidar_src, os.path.join(scene_output_dir, os.path.basename(lidar_src)))

        # Create directories for depth and edges
        sample_depth_dir = os.path.join(sample_output_dir, 'depth_imgs')
        sample_edge_dir = os.path.join(sample_output_dir, 'edgemasks')
        os.makedirs(sample_depth_dir, exist_ok=True)
        os.makedirs(sample_edge_dir, exist_ok=True)
        camera_tokens = [
                'CAM_FRONT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT'
            ]
        # Process all 6 camera images
        for channel in camera_tokens:
            token = sample['data'][channel]
            record = nusc.get('sample_data', token)
            src_fp = os.path.join(nusc.dataroot, record['filename'])
            dst_fp = os.path.join(sample_output_dir, os.path.basename(record['filename']))

            print(f"Copying {src_fp} → {dst_fp}")
            shutil.copy(src_fp, dst_fp)

            # Estimate depth and generate edge mask
            try:
                depth, img = estimateDepth(dst_fp, DATA_ROOT=sample_depth_dir)
                generate_edge_mask(img, kernel_size=10, show=False,
                                   save_path=os.path.join(sample_edge_dir, os.path.basename(dst_fp)))
            except Exception as e:
                print(f"[Warning] Failed on {dst_fp}: {e}")

        sample_token = sample['next']
        sample_id += 1
        if sample_id > 10:
            break

# Main execution: loop over all scenes
for scene in nusc.scene:
    scene_output_dir = os.path.join(DATA_ROOT, scene['token'])
    if scene_output_dir and os.path.exists(scene_output_dir):
        samples = os.listdir(scene_output_dir)
        if len(samples) > 5:
            print(f"Skipping existing scene directory: {scene_output_dir}")
            continue
    print(f"\nProcessing Scene: {scene['name']} ({scene['token']})")
    process_scene(scene, len(samples))

print("All scenes and samples processed.")

    
# sample_token, scene_name = random_scene_sample()
# copy_data(sample_token, scene_name)
# plot_random_scene_sample(scene_name, 0)

