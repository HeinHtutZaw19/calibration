import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenesExplorer

DATA_ROOT = '../../calibration_data/sets/v1.0-mini'
def align_planes_using_normals(plane_model_lidar, plane_model_camera, lidar_points, camera_points):
    
    #Normals
    #ax + by + cz + d = 0 (d offset, a,b,c normal)
    n_lidar = np.array(plane_model_lidar[:3])
    n_camera = np.array(plane_model_camera[:3])

    # Normalize the normals
    n_lidar /= np.linalg.norm(n_lidar)
    n_camera /= np.linalg.norm(n_camera)

    # Calculate the rotation matrix using Rodrigues' rotation formula
    # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    v = np.cross(n_camera, n_lidar)
    s = np.linalg.norm(v) # sine
    c = np.dot(n_camera, n_lidar) # cosine

    if s < 1e-6:
        R = np.eye(3)  
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))

    centroid_lidar = np.mean(lidar_points, axis=0)
    centroid_camera = np.mean(camera_points, axis=0)

    rotated_centroid_camera = R @ centroid_camera
    # t = centroid_lidar - rotated_centroid_camera
    t = [0, 0, 0]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def draw_registration_result(source, target, registration):
    target.paint_uniform_color([0, 1, 0])
    pcds = []

    for src, reg in zip(source, registration):
        source_copy = copy.deepcopy(src)
        source_copy = source_copy.transform(np.linalg.inv(reg.transformation))
        source_copy.paint_uniform_color([1, 0.0, 0.0])
        pcds.append(source_copy)

    pcds.append(target)
    o3d.visualization.draw_geometries(pcds, point_show_normal=False)

def detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    
    a, b, c, d = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red for the Ground Plane
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1.0, 0]) # Green for the objects
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return plane_model, inliers

import time

def animate_registration(source, target, transformations, wait=0.5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ICP Progress")
    # make a copy so we always have the original to re‐transform
    src_original = source.clone()
    vis.add_geometry(src_original)
    vis.add_geometry(target)
    for T in transformations:
        src_original.transform(np.linalg.inv(T))
        vis.update_geometry(src_original)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(wait)
    vis.run()      # keep window alive at the end
    vis.destroy_window()

def preprocess_lidar_data(filename):
    pc = LidarPointCloud.from_file(filename)
    #pc.points = [[x1, x2, ...],[y1, y2, ...], [z1, z2, ...], [a1, a2, ...]]
    source_bin_pcd = pc.points.T
    # Reshape and get only x, y, z coordinates.
    source_bin_pcd = source_bin_pcd.reshape((-1, 4))[:, 0:3]
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_bin_pcd))
    return o3d_pcd

def create_point_cloud_from_depth(depth_image, intrinsic_matrix, lower_bound, upper_bound, voxel_size, edge_mask=[]):
    height, width = depth_image.shape
    depth_image = depth_image.astype(float)
    # Create a figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns, adjust figsize as needed


    depth_image[depth_image>upper_bound] = 0
    # depth_image = depth_image/depth_image.max() * 255
    axes[0].imshow(depth_image, cmap='gray') # Add cmap for grayscale
    axes[0].set_title('Original Depth Image')
    axes[0].axis('off')  # Turn off axis labels

    depth_image[edge_mask==255] = 0

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
    return pcd


import numpy as np
from typing import List, Tuple
from nuscenes.utils.data_classes import LidarPointCloud

#TODO : Read all the raw lidar points -> crop in some range -> trans_init referential -> lidar to camera calibration
#     : Proper data loader -> all automated testing given one image 
#     : Downsample the image point clouds -> what is the effect
#     : Can we project lidar points to image, and associate 3d points to pixel? (lose normal estimation, have focal length as 1 param)

def lidar_to_depth_image(nusc, pointsensor_token: str, camera_token: str, width: int, height: int) -> np.ndarray:
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
