import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenesExplorer
import matplotlib.image as mpimg
from trans_init import coarse_translation_search
from custom_icp import icp_1dof

DATA_ROOT = '../../calibration_data/sets/v1.0-mini'

def cluster_objects(pcd, eps=0.5, min_points=50):
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )
    clusters = []
    for lbl in np.unique(labels):
        if lbl < 0: 
            continue  # noise
        idx = np.where(labels == lbl)[0]
        cluster = pcd.select_by_index(idx)
        clusters.append(cluster)
    return clusters


def crop_lidar_within_box(pcd_lidar, center=[0, 0, 0], extent=[25, 25, 5]):
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array(center) - np.array(extent) / 2,
        max_bound=np.array(center) + np.array(extent) / 2
    )
    pcd_cropped = pcd_lidar.crop(bbox)
    return pcd_cropped

def align_planes_using_normals(plane_model_lidar, plane_model_camera, pcd_lidar, pcd_camera, pcd_objs_lidar, pcd_objs_camera, visualize=None):
    
    n_lidar = np.array(plane_model_lidar[:3])
    n_camera = np.array(plane_model_camera[:3])

    # Normalize the normals
    n_lidar /= np.linalg.norm(n_lidar)
    n_camera /= np.linalg.norm(n_camera)

    # Calculate the rotation matrix using Rodrigues' rotation formula
    # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    v = np.cross(n_camera, n_lidar) #rotation axis
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

    # t = centroid_lidar - rotated_centroid_camera
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    T_rot[:3, 3] = [0, 0, 0]

    reg = o3d.pipelines.registration.registration_icp(
            source=pcd_lidar,
            target=pcd_camera,
            max_correspondence_distance=1.1,
            init=T_rot,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-2,
                relative_rmse=1e-2,
                max_iteration=1000
            )
        )
    
    T_init = reg.transformation

    pcd_objs_camera = copy.deepcopy(pcd_objs_camera).transform(np.linalg.inv(T_init))  # rotate camera to LiDAR frame
    pcd_objs_camera.paint_uniform_color([1, 0, 0])  # Red for camera points
    pcd_objs_lidar.paint_uniform_color([0, 1, 0])  # Green for LiDAR points
    if visualize:
        o3d.visualization.draw_geometries([pcd_objs_camera, pcd_objs_lidar], point_show_normal=False)

    center = [0, 12, 0]  
    extent = [14, 14, 0.1]
    bbox = visualize_search_range(center=center, extent=extent)
    if visualize:
        o3d.visualization.draw_geometries([
            pcd_objs_camera, pcd_objs_lidar, bbox
        ])

    translation, _ = coarse_translation_search(
        pcd_objs_camera, pcd_objs_lidar,
        x_range=(-1, 1), y_range=(-1, 1), z_range=[-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2, 0],
        step=0.1, center = [0,12,0], visualize=visualize
    )
    original_center = pcd_objs_camera.get_center()  
    shifted_center = original_center - np.array(translation)
    pcd_objs_camera_aligned = copy.deepcopy(pcd_objs_camera).translate(translation, relative=False)
    pcd_objs_camera_aligned.paint_uniform_color([0, 0, 1])
    if visualize:
        o3d.visualization.draw_geometries([
            pcd_objs_camera_aligned, pcd_objs_lidar
        ])
    applied_translation = shifted_center
    T_final_init = np.eye(4)
    T_final_init[:3, 3] = applied_translation
    T_final_initialization = np.linalg.inv(T_final_init) @ np.linalg.inv(T_init)
    pcd_objs_camera_aligned = copy.deepcopy(pcd_camera).transform(T_final_initialization)  # apply final transformation
    pcd_objs_camera_aligned.paint_uniform_color([0, 0, 1])  # Red for camera points
    pcd_lidar.paint_uniform_color([0, 1, 0])  # Green for LiDAR points
    # o3d.visualization.draw_geometries([pcd_objs_camera_aligned, pcd_lidar], point_show_normal=False)

    transformation, _= icp_1dof(np.asarray(pcd_lidar.points), target=np.asarray(pcd_camera.points), init=T_final_initialization, max_iterations=50, tolerance=1e-5, axis='z')
    # reg = o3d.pipelines.registration.registration_icp(
    #         source=pcd_lidar,
    #         target=pcd_camera,
    #         max_correspondence_distance=1.1,
    #         init=T_final_initialization,
    #         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
    #             relative_fitness=1e-3,
    #             relative_rmse=1e-3,
    #             max_iteration=1000
    #         )
    #     )
    pcd_camera_final = copy.deepcopy(pcd_camera).transform(transformation)
    pcd_camera_final.paint_uniform_color([1, 0, 0])  
    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Final ICP')
        vis.add_geometry(pcd_camera_final)
        vis.add_geometry(pcd_lidar)
        vis.run()
        vis.destroy_window()
    return  np.linalg.inv(transformation)


    # pc_camera_rotated.paint_uniform_color([1, 0, 0])  # Red for camera points
    # pcd_lidar.paint_uniform_color([0, 1, 0])  # Green for LiDAR points
    # o3d.visualization.draw_geometries([pc_camera_rotated, pcd_lidar], point_show_normal=False)
    return T_rot
def visualize_search_range(center=[0, 0, 0], extent=[1000, 100, 1]):
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array(center) - np.array(extent) / 2,
        max_bound=np.array(center) + np.array(extent) / 2
    )
    bbox.color = (0, 0, 1)  # Blue bounding box
    return bbox

# def align_planes_using_normals(plane_model_lidar, plane_model_camera, lidar_points, camera_points, pcd_lidar, pcd_camera):
    
#     #Normals
#     #ax + by + cz + d = 0 (d offset, a,b,c normal)
#     n_lidar = np.array(plane_model_lidar[:3])
#     n_camera = np.array(plane_model_camera[:3])

#     # Normalize the normals
#     n_lidar /= np.linalg.norm(n_lidar)
#     n_camera /= np.linalg.norm(n_camera)

#     # Calculate the rotation matrix using Rodrigues' rotation formula
#     # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
#     v = np.cross(n_camera, n_lidar) #rotation axis
#     s = np.linalg.norm(v) # sine
#     c = np.dot(n_camera, n_lidar) # cosine

#     if s < 1e-6:
#         R = np.eye(3)  
#     else:
#         vx = np.array([
#             [0, -v[2], v[1]],
#             [v[2], 0, -v[0]],
#             [-v[1], v[0], 0]
#         ])
#         R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s ** 2))

#     # t = centroid_lidar - rotated_centroid_camera
#     T_rot = np.eye(4)
#     T_rot[:3, :3] = R
#     T_rot[:3, 3] = [0, 0, 0]

#     pc_camera_rotated = copy.deepcopy(pcd_camera).transform(np.linalg.inv(T_rot))  # rotate camera to LiDAR frame
#     pc_camera_rotated.paint_uniform_color([1, 0, 0])  # Red for camera points
#     pcd_lidar.paint_uniform_color([0, 1, 0])  # Green for LiDAR points
#     visualize_with_title([pc_camera_rotated, pcd_lidar], title="Camera vs LiDAR Final Rotation Alignment")

    
#     best_translation, _ = coarse_translation_search(
#         pc_camera_rotated, pcd_camera,
#         x_range=(-5, 5), y_range=(-5, 5), z_range=[0], step=1.0
#     )
#     pc_camera_aligned = copy.deepcopy(pc_camera_rotated).translate(best_translation, relative=True)
#     pc_camera_aligned.paint_uniform_color([1, 0, 0])  # Red for camera points
#     visualize_with_title([pc_camera_aligned, pcd_lidar], title="Camera vs LiDAR Final Translation Alignment")

#     T_final = np.eye(4)
#     T_final[:3, :3] = np.linalg.inv(T_rot)[:3, :3]
#     T_final[:3, 3] = best_translation

#     pc_camera_final = copy.deepcopy(pcd_camera).transform(np.linalg.inv(T_final))  # apply final transformation
#     pc_camera_final.paint_uniform_color([1, 0, 0])  # Red for camera points
#     visualize_with_title([pc_camera_final, pcd_lidar], title="Camera vs LiDAR Final Trans Init Alignment")

#     return T_rot

# import open3d as o3d

def draw_registration_result(source, target, transformation):
    target.paint_uniform_color([0, 1, 0])
    pcds = []

    for src, reg in zip(source, transformation):
        source_copy = copy.deepcopy(src)
        source_copy = source_copy.transform(np.linalg.inv(reg))
        source_copy.paint_uniform_color([1, 0.0, 0.0])
        pcds.append(source_copy)

    pcds.append(target)
    o3d.visualization.draw_geometries([pcds[0], pcds[-1]], point_show_normal=False)

def detect_ground_plane(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000, depth_input = None, visualize=None):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    
    a, b, c, d = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1, 0, 0])  
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1, 0])
    if visualize:
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
    # if depth_input is not None:
    #     depth = depth_input.astype(float)
    #     plt.figure("Depth Map")
    #     plt.imshow(depth, cmap="plasma")
    #     plt.title("Depth Map")
    #     plt.axis("off")
    #     plt.show()

    return plane_model, inliers, outlier_cloud

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

import numpy as np
import matplotlib.pyplot as plt

def plot_depth_maps(depth_images, lower_bound, upper_bound, edge_mask, titles):
    num_images = len(depth_images)
    cols = 3
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, (title, depth_img) in enumerate(zip(titles, depth_images)):
        depth_img = depth_img.astype(float)
        depth_img[depth_img > upper_bound] = 0
        depth_img[depth_img < lower_bound] = 0
        
        if isinstance(edge_mask, list) or isinstance(edge_mask, tuple):
            current_mask = edge_mask[i]
        else:
            current_mask = edge_mask

        if current_mask is not None and current_mask.shape == depth_img.shape:
            depth_img[current_mask == 255] = 0

        plt.subplot(rows, cols, i + 1)
        plt.imshow(depth_img, cmap='plasma')
        plt.title(title)
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def create_point_cloud_from_depth(depth_image, intrinsic_matrix, lower_bound, upper_bound, voxel_size, edge_mask=[], camera_token = None):
    height, width = depth_image.shape
    depth_image = depth_image.astype(float)
   

    depth_image[depth_image>upper_bound] = 0
    depth_image[depth_image<lower_bound] = 0
    depth_image[edge_mask==255] = 0
    # depth_image = depth_image/depth_image.max() * 255
    
    # if camera_token:
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(depth_image, cmap='plasma')
    #     plt.title(camera_token)
    #     plt.axis('off')
    #     plt.show()

    # Create a grid of pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Flatten all arrays
    x = x.flatten()
    y = y.flatten()
    depth = depth_image.flatten()
    
    # Remove points with invalid depth
    valid = ((depth > lower_bound) & (depth < upper_bound)) 
    # print(lower_bound, upper_bound)
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
    # print(f"MIN :{np.asarray(pcd.points).min(axis=0)}")
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
