from __future__ import annotations

import roma
import torch
import numpy as np
from ortholoc import utils

def pose_error(pose_c2w_pred: np.ndarray, pose_c2w_gt: np.ndarray) -> tuple[float, float]:
    """
    Compute the absolute translation and angular errors between predicted and ground truth poses.

    Args:
        pose_c2w_pred: Predicted camera-to-world pose matrix (4x4).
        pose_c2w_gt: Ground truth camera-to-world pose matrix (4x4).

    Returns:
        A tuple containing:
        - Absolute translation error (float).
        - Absolute angular error in degrees (float).
    """
    abs_transl_error = torch.linalg.norm(torch.tensor(pose_c2w_pred[:3, 3]) - torch.tensor(pose_c2w_gt[:3, 3]))
    abs_angular_error = roma.rotmat_geodesic_distance(torch.tensor(pose_c2w_pred[:3, :3]),
                                                      torch.tensor(pose_c2w_gt[:3, :3])) * 180 / np.pi
    return abs_transl_error.item(), abs_angular_error.item()


def translation_error(pose_c2w_pred: np.ndarray, pose_c2w_gt: np.ndarray) -> tuple[float, float, float]:
    """
    Compute the absolute translation errors along each axis between predicted and ground truth poses.

    Args:
        pose_c2w_pred: Predicted camera-to-world pose matrix (4x4).
        pose_c2w_gt: Ground truth camera-to-world pose matrix (4x4).

    Returns:
        A tuple containing absolute translation errors along x, y, and z axes (floats).
    """
    t_x_pred, t_y_pred, t_z_pred = pose_c2w_pred[:3, 3]
    t_x_gt, t_y_gt, t_z_gt = pose_c2w_gt[:3, 3]
    t_x_error = np.abs(t_x_pred - t_x_gt)
    t_y_error = np.abs(t_y_pred - t_y_gt)
    t_z_error = np.abs(t_z_pred - t_z_gt)
    return t_x_error, t_y_error, t_z_error


def intrinsics_error(intrinsics_matrix_pred: np.ndarray, intrinsics_matrix_gt: np.ndarray) -> tuple[float, float]:
    """
    Compute the relative focal length error and principal point error between predicted and ground truth intrinsics.

    Args:
        intrinsics_matrix_pred: Predicted camera intrinsics matrix (3x3).
        intrinsics_matrix_gt: Ground truth camera intrinsics matrix (3x3).

    Returns:
        A tuple containing:
        - Relative focal length error (float).
        - Principal point error (float).
    """
    fx_pred, fy_pred = intrinsics_matrix_pred[0, 0], intrinsics_matrix_pred[1, 1]
    cx_pred, cy_pred = intrinsics_matrix_pred[0, 2], intrinsics_matrix_pred[1, 2]
    fx_gt, fy_gt = intrinsics_matrix_gt[0, 0], intrinsics_matrix_gt[1, 1]
    cx_gt, cy_gt = intrinsics_matrix_gt[0, 2], intrinsics_matrix_gt[1, 2]

    fx_rel_error = np.abs(fx_pred - fx_gt) / fx_gt
    fy_rel_error = np.abs(fy_pred - fy_gt) / fy_gt
    f_rel_error = (fx_rel_error + fy_rel_error) / 2
    principal_point_error = np.sqrt((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2)
    return f_rel_error, principal_point_error


def reprojection_error(pts3d, pose_c2w_pred, pose_c2w_gt, intrinsics_matrix_gt, intrinsics_matrix_pred=None):
    """
    Compute the reprojection error between predicted and ground truth 2D points.

    Args:
        pts3d: 3D points (N, 3).
        pose_c2w_pred: Predicted camera-to-world pose matrix (4x4).
        pose_c2w_gt: Ground truth camera-to-world pose matrix (4x4).
        intrinsics_matrix_gt: Ground truth camera intrinsics matrix (3x3).
        intrinsics_matrix_pred: Predicted camera intrinsics matrix (3x3). Defaults to ground truth intrinsics.

    Returns:
        A numpy array of reprojection errors for each point (N,).
    """
    if intrinsics_matrix_pred is None:
        intrinsics_matrix_pred = intrinsics_matrix_gt
    pts2d_pred = utils.geometry.project_pts3d(pts3d=pts3d, pose_w2c=pose_c2w_pred, intrinsics=intrinsics_matrix_pred)
    pts2d_gt = utils.geometry.project_pts3d(pts3d=pts3d, pose_w2c=pose_c2w_gt, intrinsics=intrinsics_matrix_gt)
    reproj_errors = np.linalg.norm(pts2d_pred - pts2d_gt, axis=-1)
    return reproj_errors