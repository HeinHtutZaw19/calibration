import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import os

def load_point_cloud(path: str):
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)
def best_fit_transform_1dof(A, B, axis='z'):
    # centroids
    centroid_A = A.mean(0)
    centroid_B = B.mean(0)
    AA = A - centroid_A
    BB = B - centroid_B

    # pick the two indices orthogonal to the rotation axis
    if axis == 'x':
        i1, i2 = 1, 2
    elif axis == 'y':
        i1, i2 = 0, 2
    else:  # 'z'
        i1, i2 = 0, 1

    # build 2×N arrays for those coords
    A2 = AA[:, [i1, i2]]
    B2 = BB[:, [i1, i2]]

    # 2D Procrustes: H = A2^T B2
    H2 = A2.T @ B2
    U2, S2, Vt2 = np.linalg.svd(H2)
    R2 = Vt2.T @ U2.T
    # ensure right‑handed in 2D
    if np.linalg.det(R2) < 0:
        Vt2[-1,:] *= -1
        R2 = Vt2.T @ U2.T

    # embed R2 into 3×3 R
    R = np.eye(3)
    R[np.ix_([i1, i2], [i1, i2])] = R2
    return R, np.zeros(3)

import copy
def icp_1dof(source, target, init=np.eye(4), max_iterations: int = 50, tolerance: float = 1e-5, axis: str = 'z') -> tuple:
    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)

    R_init = init[:3, :3]
    t_init = init[:3, 3]
    src = (R_init @ src.T).T + t_init

    prev_error = float('inf')
    R_total = R_init
    t_total = t_init

    for i in range(max_iterations):
        tree = KDTree(tgt)
        distances, indices = tree.query(src)
        matched = tgt[indices]

        R, t = best_fit_transform_1dof(src, matched, axis)

        src = (R @ src.T).T + t  

        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

        # Update total transformation
        R_total = R @ R_total
        t_total = R @ t_total + t

    # Final transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_total
    T[:3, 3] = t_total

    return T, src



import numpy  as np
if __name__ == '__main__':
    filename = os.path.join(os.path.dirname(__file__), 'calibration_data', 'pcd', 'bunny.ply')
    target = load_point_cloud(filename)
    ax, ay, az = np.deg2rad([20, 10, 5])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax),  np.cos(ax)]])
    # Ry = np.array([[ np.cos(ay), 0, np.sin(ay)],
    #                [0, 1, 0],
    #                [-np.sin(ay), 0, np.cos(ay)]])
    # Rz = np.array([[np.cos(az), -np.sin(az), 0],
    #                [np.sin(az),  np.cos(az), 0],
    #                [0, 0, 1]])
    R_true =  Rx
    t_true = np.array([0.5, -0.3, 0.8])

    source = (R_true @ target.T).T + t_true 

    R_est, t_est, aligned = icp_1dof(target, source, np.eye(4), axis='x')

    print("True Rotation:\n", R_true)
    print("Estimated Rotation:\n", R_est)
    print("True Translation:\n", t_true)
    print("Estimated Translation:\n", t_est)
    
