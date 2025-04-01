
from helperfunctions import *
import cv2

DATA_ROOT = '../data/sets/v1.0-mini'
nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)

def main():
    first_sample_token = nusc.scene[0]['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)

    # Get sample
    sample = nusc.sample[0]  # Load the first sample

    # Get the sensor tokens
    camera_token = sample['data']['CAM_FRONT']
    lidar_token = sample['data']['LIDAR_TOP']

    cam_front_data = nusc.get('sample_data', camera_token)
    _, _, T, R, cam_intrinsic = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']).values()

    width, height = 1600, 900  

    ### IMG
    depth_npz = np.load("../data/imgs/camera_front.npz")
    depth_img = depth_npz['depth']
    edge_mask = cv2.imread("../data/imgs/edge_mask.png", cv2.IMREAD_GRAYSCALE)
    pcd_img = create_point_cloud_from_depth(depth_img, cam_intrinsic, 2, 20, 0.5, edge_mask)


    ### LIDAR
    depth_image, _ = lidar_to_depth_image(nusc, lidar_token, camera_token, width, height)
    depth_image_display = np.uint8(depth_image / np.max(depth_image) * 255)  # Normalize to 0-255 for visualization
    pcd_lidar = create_point_cloud_from_depth(depth_image, cam_intrinsic, 2, 20, 0, edge_mask)


    draw_registration_result(pcd_lidar, pcd_img, np.eye(4))

    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_lidar, pcd_img, 0.02, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)



if __name__ == "__main__":
    main()