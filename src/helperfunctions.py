import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenesExplorer

def draw_registration_result(source, target, transformation):
    """
    Visualize the registration result of two point clouds.

    This function takes two point clouds, applies a transformation to the source point cloud,
    and visualizes both the transformed source and the target point clouds using Open3D.

    Parameters:
    source (open3d.geometry.PointCloud): The source point cloud to be transformed and visualized.
    target (open3d.geometry.PointCloud): The target point cloud to be visualized.
    transformation (numpy.ndarray): A 4x4 transformation matrix to be applied to the source point cloud.

    Returns:
    None
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_lidar_data(lt_pc):
    """
    Preprocesses LiDAR data from a given point cloud dictionary.

    This function reads a LiDAR point cloud file specified in the input dictionary,
    extracts the x, y, z coordinates, and converts it into an Open3D PointCloud object.

    Args:
        lt_pc (dict): A dictionary containing LiDAR point cloud metadata. 
                      It must include a 'filename' key with the path to the point cloud file.

    Returns:
        o3d.geometry.PointCloud: An Open3D PointCloud object containing the processed point cloud data.
    """
    pc = LidarPointCloud.from_file(os.path.join(DATA_ROOT, lt_pc['filename']))
    source_bin_pcd = pc.points.T
    # Reshape and get only x, y, z coordinates.
    source_bin_pcd = source_bin_pcd.reshape((-1, 4))[:, 0:3]
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_bin_pcd))
    return o3d_pcd

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def create_point_cloud_from_depth(depth_image, intrinsic_matrix, lower_bound, upper_bound, voxel_size, edge_mask=[]):
    height, width = depth_image.shape

    depth_image = depth_image.astype(float)
    # Create a figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns, adjust figsize as needed

    axes[0].imshow(depth_image, cmap='gray') # Add cmap for grayscale
    axes[0].set_title('Original Depth Image')
    axes[0].axis('off')  # Turn off axis labels

    depth_image[edge_mask==255] = np.nan

    # Display the second image with NaN values
    axes[1].imshow(depth_image, cmap='gray')  # Add cmap for grayscale
    axes[1].set_title('Edges Set to NaN')
    axes[1].axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

    # Create a grid of pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Flatten all arrays
    x = x.flatten()
    y = y.flatten()
    depth = depth_image.flatten()
    
    # Remove points with invalid depth
    valid = ((depth > lower_bound) * (depth < upper_bound)) > 0
    x = x[valid]
    y = y[valid]

    depth = depth[valid]
    
    
    # Back-project 2D points to 3D
    fx, fy = intrinsic_matrix[0][0], intrinsic_matrix[1][1]
    cx, cy = intrinsic_matrix[0][2], intrinsic_matrix[1][2]
    
    x_3d = (x - cx) * depth / fx
    y_3d = (y - cy) * depth / fy
    z_3d = depth
    
    points_3d = np.column_stack((x_3d, y_3d, z_3d))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    # Convert to Open3D point cloud format
    if(voxel_size>0):
        pcd = pcd.voxel_down_sample(voxel_size = voxel_size)
        

    x_3d, y_3d, z_3d = np.array(pcd.points).T
    # Visualization
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_3d, y_3d, z_3d, s=1)  # Small marker size for better visualization
    plt.show()
    
    return pcd


import numpy as np
from typing import List, Tuple
from nuscenes.utils.data_classes import LidarPointCloud

def lidar_to_depth_image(nusc, pointsensor_token: str, camera_token: str, width: int, height: int) -> np.ndarray:
    """
    Projects LiDAR points to the image plane and generates a depth image.
    
    Args:
        nusc (NuScenes): NuScenes dataset instance.
        pointsensor_token (str): Token of the LiDAR sensor data.
        camera_token (str): Token of the camera sensor data.
        width (int): Image width.
        height (int): Image height.

    Returns:
        np.ndarray: A (height, width) depth image where pixel values represent distance.
    """
    # Map LiDAR to camera image
    nusc_explorer = NuScenesExplorer(nusc)
    points_2d, depths, im = nusc_explorer.map_pointcloud_to_image(
        pointsensor_token=pointsensor_token,
        camera_token=camera_token,
        min_dist=1.0  # Ignore points too close to the sensor
    )
    # Initialize empty depth image
    depth_image = np.zeros((height, width), dtype=np.float32)
    for i in range(points_2d.shape[1]):
        x, y = int(points_2d[0, i]), int(points_2d[1, i])  # 2D coordinates (x, y)

        # Ensure (x, y) falls within the valid image bounds
        if 0 <= x < width and 0 <= y < height:
            # Assign depth only if it's the closest point or the first point
            if depth_image[y, x] == 0 or depths[i] < depth_image[y, x]:
                depth_image[y, x] = depths[i]
    
    return depth_image, depths
