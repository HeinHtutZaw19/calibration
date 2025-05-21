
from helperfunctions import *
from edgemask import generate_edge_mask
from nuscenes.utils.geometry_utils import transform_matrix
# from depth_pro.inference import estimateDepth
from pyquaternion import Quaternion
import cv2
from nuimages import NuImages
from PIL import Image

DATA_ROOT = './calibration_data/sets/v1.0-mini'
print(DATA_ROOT)
nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)
print("done loading nusc")
def ground_truth(nusc, lidar_token, camera_token):
    lidar_record = nusc.get('sample_data', lidar_token)
    _, _, lidar_T, lidar_R, _ =  nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token']).values()

    camera_record = nusc.get('sample_data', camera_token)
    _, _, camera_T, camera_R, _ = (nusc.get('calibrated_sensor', camera_record['calibrated_sensor_token']).values())

    # Transform from sensor frame to ego-body frame
    T_car_cam   = transform_matrix(camera_T,   Quaternion(camera_R),   inverse=False)
    T_car_lidar = transform_matrix(lidar_T, Quaternion(lidar_R), inverse=False)
    T_cam_car   = np.linalg.inv(T_car_cam)

    # Camera to Lidar : first Camera to ego, then Ego to Lidar
    T_cam_lidar = T_cam_car @ T_car_lidar
    return T_cam_lidar

def multiscale_gicp_registration(source, target, n_scales, distance, iterations_max, initial_transformation):
    voxel_sizes = np.linspace(distance, distance / n_scales, n_scales)
    max_correspondence_distances = np.linspace(distance, distance / n_scales, n_scales) * 2.5
    iterations = np.linspace(iterations_max, iterations_max / n_scales, n_scales).astype(int)

    for i in range(n_scales):
        # target.scale(0.95, target.get_center())
        source = source.voxel_down_sample(voxel_sizes[i])
        target = target.voxel_down_sample(voxel_sizes[i])
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel_sizes[i] * 2, max_nn=100))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(voxel_sizes[i] * 2, max_nn=100))

        result_icp = o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            max_correspondence_distances[i],
            initial_transformation,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=iterations[i]
            )
        )
        initial_transformation = result_icp.transformation

    return result_icp



def icp_sample(lidar_token, camera_token):
    #Preprocess the image data
    nuim = NuImages(version='v1.0-mini', dataroot=DATA_ROOT)
    sample_data = nuim.get('sample_data', camera_token)
    file_path = os.path.join(nuim.dataroot, sample_data['filename'])
    img = np.array(Image.open(file_path))
    depth_img_path = estimateDepth(img)
    depth_npz = np.load(depth_img_path)
    depth_img = depth_npz['depth'].astype(float)
    edge_mask = generate_edge_mask(depth_img_path, kernel_size=10, show=False, save_path="../../calibration_data/imgs/edge_mask.png")
    _, _, T, R, cam_intrinsic = nusc.get('calibrated_sensor', camera_token['calibrated_sensor_token']).values()
    pcd_img = create_point_cloud_from_depth(depth_img, cam_intrinsic, 2, 30, 0.2, edge_mask)

    # Load the lidar data
    filename = os.path.join(nusc.dataroot, nusc.get('sample_data', lidar_token)['filename'])
    pcd_lidar= preprocess_lidar_data(filename)
    trans_init = np.asarray(ground_truth(nusc, lidar_token, camera_token))

    pcd_img.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=50))
    pcd_lidar.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=50))
    
    # reg_p2l = o3d.pipelines.registration.registration_icp(
    #     pcd_lidar, pcd_img, 1.1, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())


    print("Updated Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(pcd_img, pcd_lidar, trans_init)

import shutil
def copy_data():
    DATA_ROOT = './calibration_data/sets/v1.0-mini'
    DEST_DIR  = './calibration_data/imgs/sample1'

    # Initialize once
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)
    scene = nusc.scene[3]
    # Pick whichever sample you want; here we grab the first:
    sample = nusc.sample[3]

    # Ensure destination folder exists
    os.makedirs(DEST_DIR, exist_ok=True)

    for channel in [
        'CAM_FRONT', 'LIDAR_TOP',
        'CAM_BACK', 'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT'
    ]:
        token  = sample['data'][channel]
        record = nusc.get('sample_data', token)
        src_fp = os.path.join(nusc.dataroot, record['filename'])
        dst_fp = os.path.join(DEST_DIR, os.path.basename(record['filename']))
        
        print(f"Copying {src_fp} → {dst_fp}")
        shutil.copy(src_fp, dst_fp)

def get_tokens(sample_id):
    nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)
    sample = nusc.sample[sample_id]
    camera_tokens = [
        sample['data']['CAM_FRONT'], 
        sample['data']['CAM_BACK'],
        sample['data']['CAM_BACK_LEFT'],
        sample['data']['CAM_BACK_RIGHT'],
        sample['data']['CAM_FRONT_LEFT'],
        sample['data']['CAM_FRONT_RIGHT']
    ]
    filenames = []
    for token in camera_tokens:
        file = nusc.get('sample_data', token)['filename'].split('/').pop()
        filenames.append(f"depth-pro-run -i ../../../../sample_data/{file} -o ../../../../{file}")
    return filenames
        
def main():
    SAMPLE_ID = 3
    sample = nusc.sample[SAMPLE_ID]  
    camera_tokens = [
        # sample['data']['CAM_FRONT'], 
        sample['data']['CAM_BACK'],
        # sample['data']['CAM_BACK_LEFT'],
        # sample['data']['CAM_BACK_RIGHT'],
        # sample['data']['CAM_FRONT_LEFT'],
        # sample['data']['CAM_FRONT_RIGHT']
    ]
    lidar_token = sample['data']['LIDAR_TOP']
    filename = os.path.join(nusc.dataroot, nusc.get('sample_data', lidar_token)['filename'])
    pcd_lidar= preprocess_lidar_data(filename)
    pcd_img = []
    trans_init = []
    plane_model_lidar, inliers_lidar = detect_ground_plane(pcd_lidar)


    for camera_token in camera_tokens:
        camera_front_filename = nusc.get('sample_data', camera_token)['filename'].split('/').pop().split('.')[0]
        cam_data = nusc.get('sample_data', camera_token)
        _, _, T, R, cam_intrinsic = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token']).values()

        depth_npz = np.load(f"./calibration_data/imgs/sample{SAMPLE_ID}/depth_imgs/{camera_front_filename}.npz")
        depth_img = depth_npz['depth'].astype(float)
        edge_mask = generate_edge_mask(f"./calibration_data/imgs/sample{SAMPLE_ID}/depth_imgs/{camera_front_filename}.jpg", kernel_size=10, show=False, save_path="./calibration_data/imgs/edge_mask.png")
        pcd_img.append(create_point_cloud_from_depth(depth_img, cam_intrinsic, 2, 20, 0.2, edge_mask))
        # trans_init.append(np.asarray(ground_truth(nusc, lidar_token, camera_token)))
    
    for pc in pcd_img:
        plane_model_camera, inliers_camera = detect_ground_plane(pc)
        trans_init.append(align_planes_using_normals(plane_model_lidar, plane_model_camera, np.asarray(pcd_lidar.points)[inliers_lidar], np.asarray(pc.points)[inliers_camera]))
        pc.scale(0.9, pcd_img[0].get_center()) # Scaling down 90 %
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=50))
    pcd_lidar.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=50))
    
    reg_p2l = []
    for pcd, init in zip(pcd_img, trans_init):
        # reg = o3d.pipelines.registration.registration_icp(
        #     source=pcd_lidar,
        #     target=pcd,
        #     max_correspondence_distance=1.1,
        #     init=init,
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        #         relative_fitness=1e-6,
        #         relative_rmse=1e-6,
        #         max_iteration=2000
        #     )
        # )
        n_scales = 10
        distance = 1
        iterations_max = 1000

        reg = multiscale_gicp_registration(
            pcd_lidar, pcd, n_scales, distance, iterations_max, init
        )
        reg_p2l.append(reg)



    # n_scales = 10
    # distance = 1
    # iterations_max = 1000

    # reg_p2l = multiscale_gicp_registration(
    #     pcd_lidar, pcd_img[0], n_scales, distance, iterations_max, trans_init[0]
    # )


    print("Updated Transformation is:")
    transforms = [reg.transformation for reg in reg_p2l]
    print(transforms)
    draw_registration_result(pcd_img, pcd_lidar, reg_p2l)


if __name__ == "__main__":
    main()
    # for i in get_tokens(4):
    #     print(i)