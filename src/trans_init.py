import numpy as np
import open3d as o3d

def nearest_neighbor_distance_sum(pc_source, pc_target_kdtree, pc_target):
    distances = []
    for pt in np.asarray(pc_source.points):
        [_, idx, _] = pc_target_kdtree.search_knn_vector_3d(pt, 1)
        closest_pt = np.asarray(pc_target.points)[idx[0]]
        dist = np.linalg.norm(pt - closest_pt)
        distances.append(dist)
    return np.sum(distances)

def crop_lidar_within_bbox(pcd_lidar, reference_pcd, margin=[1.0, 1.0, 1.0]):
    ref_bbox = reference_pcd.get_axis_aligned_bounding_box()
    
    min_bound = ref_bbox.min_bound - np.array(margin)
    max_bound = ref_bbox.max_bound + np.array(margin)
    
    adjusted_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    return pcd_lidar.crop(adjusted_bbox)

import copy
def coarse_translation_search(pc_camera, pc_lidar, 
                              x_range, y_range, z_range=[0], 
                              step=1.0, center = [0,0,0], rotation=np.eye(4), visualize=None):

    pc_camera_down = copy.deepcopy(pc_camera).voxel_down_sample(voxel_size=0.5)
    pc_camera_down = copy.deepcopy(pc_camera).voxel_down_sample(voxel_size=0.5)
    
    # cropped_lidar = crop_lidar_within_bbox(pc_lidar, pc_camera_down, margin= np.array([5, 5, 5]))
    cropped_lidar = pc_lidar
    cropped_lidar.paint_uniform_color([0, 0, 1])  # Green for lidar points
    pc_camera_down.paint_uniform_color([1, 0, 0])  # Red for camera points
    if visualize:
        o3d.visualization.draw_geometries([cropped_lidar, pc_camera_down], point_show_normal=False)
    pc_lidar_kdtree = o3d.geometry.KDTreeFlann(cropped_lidar)
    # print(f"Original lidar points: {np.asarray(pc_lidar.points).shape}")
    # print(f"Cropped lidar points: {np.asarray(cropped_lidar.points).shape}")

    best_translation = pc_camera_down.get_center()
    min_distance_sum = nearest_neighbor_distance_sum(pc_camera_down, pc_lidar_kdtree, cropped_lidar)
    # print(min_distance_sum)
    xs = np.arange(x_range[0], x_range[1], step)
    ys = np.arange(y_range[0], y_range[1], step)
    zs = np.array(z_range)
    
    # pc_camera_down.paint_uniform_color([1, 0, 0])  # Red for camera points
    # pc_lidar.paint_uniform_color([0, 1, 0])  # Green for lidar points
    # # pc_cam_trans = pc_camera_down.translate(shift_rotated, relative=False)

    # o3d.visualization.draw_geometries([pc_lidar, pc_camera_down], point_show_normal=False)
    # T = np.eye(4)
    # T[:3, :3] = rotation[:3, :3]
    # T[:3, 3] = [0,0,0]
    # pc_camera_down = pc_camera_down.transform(np.linalg.inv(T))
    if visualize:
        o3d.visualization.draw_geometries([pc_camera_down, pc_lidar], point_show_normal=False)
    
    for x_shift in xs:
        for y_shift in ys:
            for z_shift in zs:
                shift = np.array([x_shift, y_shift, z_shift]) + np.array(center)
                # shift_rotated = rotation @ shift
                pc_cam_trans = pc_camera_down.translate(shift, relative=False)                
                dist_sum = nearest_neighbor_distance_sum(pc_cam_trans, pc_lidar_kdtree, cropped_lidar)
                if dist_sum < min_distance_sum:
                    min_distance_sum = dist_sum
                    best_translation = shift

                pc_camera_down.translate(-shift, relative=False)

    # print(f"Best translation (x,y,z): {best_translation} with distance sum: {min_distance_sum}")

    return best_translation, min_distance_sum



# best_trans, best_dist = coarse_translation_search(pc_camera_rotated, pc_lidar,
#                                                  x_range=(-5,5), y_range=(-5,5), step=1.0)

# print(f"Best translation (x,y,z): {best_trans} with distance sum: {best_dist}")
