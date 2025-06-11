import numpy as np
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='./calibration_data/sets/v1.0-mini', verbose=True)

def rotation_error(R1, R2):
        R_diff = R1.T @ R2  
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
        return np.degrees(angle)

def translation_error(t1, t2):
    return np.linalg.norm(t1 - t2)

def eval(nusc, token):
    scene = nusc.get('scene', token)
    sample_token = scene['first_sample_token']

    trans_errors = []
    rot_errors = []

    while sample_token:
        sample = nusc.get('sample', sample_token)
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        pred_translation = ego_pose['translation']  
        pred_rotation = ego_pose['rotation']        

        trans_err = translation_error(ego_pose['translation'], pred_translation)
        rot_err = rotation_error(ego_pose['rotation'], pred_rotation)

        trans_errors.append(trans_err)
        rot_errors.append(rot_err)

        sample_token = sample['next'] if sample['next'] else None

    print(f"Avg translation error: {np.mean(trans_errors):.3f} meters")
    print(f"Avg rotation error: {np.mean(rot_errors):.3f} degrees")


if __name__ == '__main__':
    
    
    sample = nusc.get('sample', scene['first_sample_token'])
    sample = nusc.get('sample', sample['next'])
    print(f"Processing scene: {scene['name']}, first sample token: {scene['first_sample_token']}")
    print("lidar token:", sample['data']['LIDAR_TOP'])
    print(nusc.get('sample_data', sample['data']['LIDAR_TOP'])['filename'])