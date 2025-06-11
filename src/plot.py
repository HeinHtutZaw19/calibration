from nuscenes.nuscenes import NuScenes
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


DATA_ROOT = './calibration_data/imgs'
nusc = NuScenes(version='v1.0-mini', dataroot='./calibration_data/sets/v1.0-mini', verbose=True)

def plot_random_scene_sample(scene_name, sample_id):
    fig, axes = plt.subplots(3, 6, figsize=(36, 12))  # 3 rows: original, depth, edge mask
    sample_dir = os.path.join(DATA_ROOT, scene_name, f"sample_{sample_id}")
    depth_dir = os.path.join(sample_dir, 'depth_imgs')
    edgemask_dir = os.path.join(sample_dir, 'edgemasks')

    if not os.path.exists(sample_dir) or not os.path.exists(depth_dir) or not os.path.exists(edgemask_dir):
        print(f"One or more required directories are missing:\n"
              f" - Sample: {sample_dir}\n - Depth: {depth_dir}\n - Edgemasks: {edgemask_dir}")
        return
    
    depths = []

    original_imgs = sorted([f for f in os.listdir(sample_dir) if f.endswith('.jpg') and '__' in f])
    
    for i, img_file in enumerate(original_imgs[:6]):
        orig_path = os.path.join(sample_dir, img_file)
        depth_path = os.path.join(depth_dir, img_file)
        depth_npz_path = os.path.join(depth_dir, img_file.split('.')[0] + '.npz')
        edge_path = os.path.join(edgemask_dir, img_file)

        # Extract clean camera channel name (e.g., CAM_FRONT)
        channel = img_file.split('__')[1].split('.')[0]  # Removes extension too

        # Row 0: Original
        img_orig = mpimg.imread(orig_path)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f"{channel} (Original)", fontsize=10)
        axes[0, i].axis('off')

        # Row 1: Depth
        if os.path.exists(depth_npz_path):
            depth_npz = np.load(depth_npz_path)
            depth_img = depth_npz['depth'].astype(float)
            depth_img[depth_img<0] = 0
            depth_img[depth_img>11] = 0
            depth_img[depth_img==255] = 0
            axes[1, i].imshow(depth_img, cmap='plasma')
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

def histogram_plot(rot_err_with_obj, rot_err_no_obj, trans_err_with_obj, trans_err_no_obj):
    # rot_err_with_obj = list(np.random.normal(loc=12, scale=3, size=40)) + [40, 45]
    # rot_err_no_obj   = list(np.random.normal(loc=3, scale=0.5, size=40))

    # trans_err_with_obj = list(np.random.normal(loc=6, scale=2, size=40)) + [18, 20]
    # trans_err_no_obj   = list(np.random.normal(loc=1.2, scale=0.2, size=40))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(rot_err_with_obj, bins=15, alpha=0.7, label='With Objects', color='red')
    plt.hist(rot_err_no_obj, bins=15, alpha=0.7, label='No Objects', color='blue')
    plt.title('Rotation Error Distribution')
    plt.xlabel('Rotation Error (°)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(trans_err_with_obj, bins=15, alpha=0.7, label='With Objects', color='orange')
    plt.hist(trans_err_no_obj, bins=15, alpha=0.7, label='No Objects', color='green')
    plt.title('Translation Error Distribution')
    plt.xlabel('Translation Error (m)')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()


# plot_random_scene_sample("d25718445d89453381c659b9c8734939", 0)
if __name__ == "__main__":
    # Example usage
    rot_err_with_obj = list(np.random.normal(loc=12, scale=3, size=40)) + [40, 45]
    rot_err_no_obj   = list(np.random.normal(loc=3, scale=0.5, size=40))

    trans_err_with_obj = list(np.random.normal(loc=6, scale=2, size=40)) + [18, 20]
    trans_err_no_obj   = list(np.random.normal(loc=1.2, scale=0.2, size=40))
    histogram_plot(rot_err_with_obj, rot_err_no_obj, trans_err_with_obj, trans_err_no_obj)