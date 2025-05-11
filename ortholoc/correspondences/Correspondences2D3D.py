from __future__ import annotations

from typing import Any
from typing_extensions import Self
import numpy as np
import random
import math

from ortholoc import utils
from ortholoc.correspondences import Correspondences


class Correspondences2D3D(Correspondences):
    """
    Class to represent 2D-3D correspondences.
    """

    pts0: np.ndarray = np.empty((0, 2))
    pts1: np.ndarray = np.empty((0, 3))
    is_normalized: bool
    confidences: np.ndarray | None = None

    def __setattr__(self, key: str, value: Any) -> None:
        if key == 'pts0':
            if value.ndim != 2 or value.shape[1] != 2:
                raise ValueError(f"Invalid shape for {key}: {value.shape}")
        elif key == 'pts1':
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f"Invalid shape for {key}: {value.shape}")
        super().__setattr__(key, value)

    def normalized(self, w0: int, h0: int) -> Self:
        """
        Normalize the 2D points to the range [-1, 1] based on the image dimensions.
        """
        if not self.is_normalized:
            pts0 = utils.geometry.norm_pts2d(self.pts0, w=w0, h=h0)
            return Correspondences2D3D(pts0=pts0, pts1=self.pts1, confidences=self.confidences, is_normalized=True)
        return self

    def denormalized(self, w0: int, h0: int) -> Self:
        """
        Denormalize the 2D points from the range [-1, 1] to [0, w0-1] and [0, h0-1].
        """
        if self.is_normalized:
            pts0 = utils.geometry.denorm_pts2d(self.pts0, w=w0, h=h0)
            return Correspondences2D3D(pts0=pts0, pts1=self.pts1, confidences=self.confidences, is_normalized=False)
        return self

    def calibrate(
            self, num_points: int, width: int, height: int, intrinsics_matrix: np.ndarray | None = None,
            reprojection_error_diag_ratio=None, focal_length_init: np.ndarray | None = None,
            reprojection_error=5.0, pnp_mode='poselib', fix_principle_points: bool = True
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Estimate the camera pose and intrinsics using 2D-3D correspondences.
        """
        is_finite_mask = self.is_finite_mask
        correspondences_2d3d_denormalized = self.denormalized(w0=width, h0=height)
        correspondences_2d3d_finite = correspondences_2d3d_denormalized.take_mask(is_finite_mask)
        assert len(correspondences_2d3d_finite) > 0, "No correspondences available for localization"
        pts2d_query, pts3d_query = correspondences_2d3d_finite.pts0, correspondences_2d3d_finite.pts1

        success = False
        pose_c2w_pred = None
        inliers_mask = None
        reprojection_errors = None

        if len(pts2d_query) > 0:
            if num_points is not None and len(pts2d_query) > num_points:
                idxs = np.array(random.sample(range(len(pts2d_query)), num_points))
            else:
                idxs = np.arange(len(pts2d_query))
            pts3d_query = pts3d_query[idxs]
            pts2d_query = pts2d_query[idxs]

            if reprojection_error_diag_ratio is not None:
                reprojection_error_img = reprojection_error_diag_ratio * math.sqrt(width ** 2 + height ** 2)
            else:
                reprojection_error_img = reprojection_error

            if intrinsics_matrix is None:
                success, pose_c2w_pred, intrinsics_matrix = utils.pose.run_calibration(
                    pts2D=pts2d_query, pts3D=pts3d_query, distortion=None, mode=pnp_mode,
                    reprojectionError=reprojection_error_img, img_size=(width, height),
                    focal_length_init=focal_length_init, fix_principle_points=fix_principle_points)
                inliers_mask = np.ones(len(pts2d_query), dtype=bool)
            else:
                success, pose_c2w_pred, inliers_mask = utils.pose.run_pnp(pts2D=pts2d_query, pts3D=pts3d_query,
                                                                     K=intrinsics_matrix, distortion=None,
                                                                     mode=pnp_mode,
                                                                     reprojectionError=reprojection_error_img,
                                                                     img_size=(width, height))
            if pose_c2w_pred is not None:
                pose_c2w_pred = pose_c2w_pred[:3, :].astype(np.float32)
            if intrinsics_matrix is not None:
                intrinsics_matrix = intrinsics_matrix.astype(np.float32)
            if inliers_mask is not None and len(inliers_mask) == len(idxs):
                idxs = idxs[inliers_mask]
                idxs_finite = np.where(is_finite_mask)[0]
                inliers_mask = np.zeros(len(self), dtype=bool)
                inliers_mask[idxs_finite[idxs]] = True

            if pose_c2w_pred is not None and intrinsics_matrix is not None:
                reprojection_errors = correspondences_2d3d_denormalized.reprojection_errors(
                    pose_c2w=pose_c2w_pred, intrinsics_matrix=intrinsics_matrix)
        return success, pose_c2w_pred, intrinsics_matrix, inliers_mask, reprojection_errors

    def reprojection_errors(self, pose_c2w: np.ndarray, intrinsics_matrix: np.ndarray) -> np.ndarray:
        """
        Compute the reprojection errors of the 3D points in the 2D image plane.
        """
        assert not self.is_normalized
        pts2d_proj = utils.geometry.project_pts3d(self.pts1, pose_w2c=utils.pose.inv_pose(pose_c2w), intrinsics=intrinsics_matrix)
        reprojection_errors = np.linalg.norm(pts2d_proj - self.pts0, axis=1)
        return reprojection_errors
