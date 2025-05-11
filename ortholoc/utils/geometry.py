from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import to_rgb
import matplotlib.patches as mpatches

from ortholoc import utils

from ortholoc.correspondences import Correspondences2D2D


##################################
# Transforms
#################################
def point_to_depth_map(
        point_map: np.ndarray,
        extrinsics: np.ndarray
) -> np.ndarray:
    """
    Convert a 3D point map to a depth map using extrinsics.

    Args:
        point_map: 3D point map as a numpy array.
        extrinsics: Extrinsics matrix.

    Returns:
        Depth map as a numpy array.
    """
    h, w, _ = point_map.shape
    ones = np.ones((h, w, 1))
    homogeneous_points = np.concatenate((point_map, ones), axis=-1)
    homogeneous_points = homogeneous_points.reshape(-1, 4).T
    transformed_points = np.dot(extrinsics, homogeneous_points)
    depth_map = transformed_points[2, :].reshape(h, w)
    return depth_map


def create_grid2d(w: int, h: int) -> np.ndarray:
    """
    Create a 2D grid of pixel coordinates.

    Args:
        w: Width of the grid.
        h: Height of the grid.

    Returns:
        A numpy array of shape (h, w, 2) containing pixel coordinates.
    """
    all_pixels_coordinates = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack(all_pixels_coordinates, axis=-1)


def denorm_pts2d(pts2d: np.ndarray | torch.Tensor, h: int, w: int) -> np.ndarray | torch.Tensor:
    """
    Denormalize 2D points from [-1, 1] to pixel coordinates [0, w-1] and [0, h-1]

    Args:
        pts2d: Normalized 2D points.
        h: Image height.
        w: Image width.

    Returns:
        Denormalized 2D points.
    """
    h = h[..., None] if isinstance(h, torch.Tensor) else h
    w = w[..., None] if isinstance(w, torch.Tensor) else w
    if isinstance(pts2d, torch.Tensor):
        return torch.stack(((w - 1) / 2 * (pts2d[..., 0] + 1), (h - 1) / 2 * (pts2d[..., 1] + 1)), dim=-1)
    else:
        return np.stack(((w - 1) / 2 * (pts2d[..., 0] + 1), (h - 1) / 2 * (pts2d[..., 1] + 1)), axis=-1)


def norm_pts2d(pts2d: np.ndarray | torch.Tensor, h: int, w: int) -> np.ndarray | torch.Tensor:
    """
        Normalize 2D points from pixel coordinates [0, w-1] and [0, h-1] to [-1, 1].

        Args:
            pts2d: 2D points in pixel coordinates.
            h: Image height.
            w: Image width.

        Returns:
            Normalized 2D points.
        """
    h = h[..., None] if isinstance(h, torch.Tensor) else h
    w = w[..., None] if isinstance(w, torch.Tensor) else w
    if isinstance(pts2d, torch.Tensor):
        return torch.stack((2 / (w - 1) * pts2d[..., 0] - 1, 2 / (h - 1) * pts2d[..., 1] - 1), dim=-1)
    else:
        return np.stack((2 / (w - 1) * pts2d[..., 0] - 1, 2 / (h - 1) * pts2d[..., 1] - 1), axis=-1)


def visibility_map(pts2d: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Check if 2D points are within the image bounds.

    Args:
        pts2d: 2D points.
        w: Image width.
        h: Image height.

    Returns:
        A boolean mask indicating visibility.
    """
    return np.bitwise_and(np.bitwise_and(pts2d[..., 0] >= 0, pts2d[..., 0] < w),
                          np.bitwise_and(pts2d[..., 1] >= 0, pts2d[..., 1] < h))


def pts2d_is_normalized(pts2d: np.ndarray) -> bool:
    """
        Check if 2D points are normalized to [-1, 1].

        Args:
            pts2d: 2D points.

        Returns:
            True if all points are normalized, False otherwise.
        """
    return bool(np.all(np.bitwise_and(-1.0 <= pts2d, pts2d <= 1.0)))


def project_pts3d(pts3d: np.ndarray, pose_w2c: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Project 3D points to 2D using a camera pose and intrinsics.

    Args:
        pts3d: 3D points.
        pose_w2c: World-to-camera pose matrix.
        intrinsics: Camera intrinsic matrix.

    Returns:
        Projected 2D points.
    """
    N = pts3d.shape[0]
    pts3d_h = np.hstack([pts3d, np.ones((N, 1), dtype=pts3d.dtype)])
    pts_cam = pose_w2c @ pts3d_h.T
    pts_img = intrinsics @ pts_cam
    pts2d = (pts_img[:2, :] / pts_img[2, :]).T
    return pts2d


def sample_grid(grid: np.ndarray, pts2d: np.ndarray, mode: str = 'nearest', align_corners: bool | None = None,
                interpolate_gaps: bool = False, is_normalized: bool | None = None) -> np.ndarray:
    """
    Sample values from a grid at specified 2D points.

    Args:
        grid: Input grid (H, W, C).
        pts2d: 2D points to sample from the grid.
        mode: Interpolation mode ('nearest' or 'bilinear').
        align_corners: Whether to align corners for interpolation.
        interpolate_gaps: Whether to interpolate gaps if sampling fails.
        is_normalized: Whether the 2D points are normalized.

    Returns:
        Sampled values from the grid.
    """
    assert pts2d.ndim == 2
    assert grid.ndim == 3
    if is_normalized is None:
        is_normalized = pts2d_is_normalized(pts2d)
    if not is_normalized:
        h, w = grid.shape[:2]
        pts2d = norm_pts2d(pts2d, h=h, w=w)
    sampled_points = F.grid_sample(
        torch.from_numpy(grid.astype(np.float32))[None].permute(0, 3, 1, 2),
        torch.from_numpy(pts2d.astype(np.float32))[None][None], mode=mode,
        align_corners=align_corners).permute(0, 2, 3, 1)[0][0].numpy()
    if not np.isfinite(sampled_points).all():
        if interpolate_gaps:
            grid = fill_gaps(grid)
        sampled_points = F.grid_sample(
            torch.from_numpy(grid.astype(np.float32))[None].permute(0, 3, 1, 2),
            torch.from_numpy(pts2d.astype(np.float32))[None][None], mode=mode,
            align_corners=align_corners).permute(0, 2, 3, 1)[0][0].numpy()
    return sampled_points


def compute_dist(grid3d_0: np.ndarray, grid3d_1: np.ndarray, correspondences_2d2d: Correspondences2D2D,
                 mode: str = 'nearest', align_corners: bool | None = True):
    """
    Compute the distance between corresponding 3D points sampled from two grids.

    Args:
        grid3d_0: First 3D grid (H0, W0, C).
        grid3d_1: Second 3D grid (H1, W1, C).
        correspondences_2d2d: 2D-2D correspondences object.
        mode: Interpolation mode ('nearest' or 'bilinear').
        align_corners: Whether to align corners for interpolation.

    Returns:
        A numpy array of distances between corresponding 3D points.
    """
    h0, w0 = grid3d_0.shape[:2]
    h1, w1 = grid3d_1.shape[:2]
    correspondences_2d2d = correspondences_2d2d.normalized(w0=w0, h0=h0, w1=w1, h1=h1)
    x0, x0in1 = correspondences_2d2d.pts0, correspondences_2d2d.pts1
    pts3d_0 = sample_grid(grid3d_0, x0, mode=mode, align_corners=align_corners,
                          is_normalized=correspondences_2d2d.is_normalized)
    pts3d_1 = sample_grid(grid3d_1, x0in1, mode=mode, align_corners=align_corners,
                          is_normalized=correspondences_2d2d.is_normalized)
    return np.linalg.norm(pts3d_0 - pts3d_1, axis=-1)


def plot_map(array: np.ndarray) -> None:
    """
    Plot a 2D array as an image.

    Args:
        array: 2D numpy array to be plotted.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    ax.imshow(array)
    ax.axis('off')
    plt.show()


def create_grid_pts2d(w: int, h: int, normalized: bool = False) -> np.ndarray:
    """
    Create a 2D grid of points.

    Args:
        w: Width of the grid.
        h: Height of the grid.
        normalized: Whether to normalize the grid points to [-1, 1].

    Returns:
        A numpy array of shape (h, w, 2) containing the grid points.
    """
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1)
    if normalized:
        grid = norm_pts2d(grid, h, w)
    return grid


def fill_gaps(grid: np.ndarray) -> np.ndarray:
    """
    Fill gaps in a grid by interpolating missing values.

    Args:
        grid: Input grid with missing values (NaNs).

    Returns:
        A grid with gaps filled using nearest-neighbor interpolation.
    """
    x, y = np.indices(grid.shape[:2])
    valid = np.isfinite(grid).all(axis=-1)
    interpolated_data = griddata(points=(x[valid], y[valid]), values=grid[valid], xi=(x, y), method='nearest')
    return interpolated_data


def dist_to_confidences(distances:  np.ndarray, decay: float =1.0) -> np.ndarray:
    """
    Convert distances to confidence scores using an exponential decay function.

    Args:
        distances: Array of distances (N,).
        decay: Decay rate for the exponential function (float).

    Returns:
        A numpy array of confidence scores (N,).
    """
    confidences = np.exp(-decay * distances)
    confidences[confidences < 0.1] = 0
    return confidences


def grid_pts3d_to_mask(pts2d: np.ndarray, top_left: np.ndarray, bottom_right: np.ndarray, height: int,
                       width: int) -> np.ndarray:
    """
    Convert 3D grid points to a binary mask within specified bounds.

    Args:
        pts2d: 2D points projected from 3D grid (N, 2).
        top_left: Top-left corner of the bounding box (x, y).
        bottom_right: Bottom-right corner of the bounding box (x, y).
        height: Height of the output mask (int).
        width: Width of the output mask (int).

    Returns:
        A binary mask of shape (height, width) indicating valid points.
    """
    mask = np.zeros((height, width), dtype=bool)
    scale_x = width / (bottom_right[0] - top_left[0])
    scale_y = height / (top_left[1] - bottom_right[1])
    x0 = top_left[0]
    y0 = top_left[1]
    coords = np.stack([(pts2d[:, 0] - x0) * scale_x, (y0 - pts2d[:, 1]) * scale_y], axis=-1).astype(np.int32)
    invalid_pts2d = np.any(np.isnan(coords), axis=-1)
    coords = coords[~invalid_pts2d]
    coords = coords[np.logical_and(coords[:, 0] >= 0, coords[:, 0] < width)]
    coords = coords[np.logical_and(coords[:, 1] >= 0, coords[:, 1] < height)]
    mask[coords[:, 1], coords[:, 0]] = 1
    return mask


def crop_rasters(
    point_map: np.ndarray,
    dsm: np.ndarray,
    dop: np.ndarray,
    covisibility_ratio: float,
    direction: str | None = None,
    plot: bool = False
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Crop rasters based on covisibility ratio and direction.

    Args:
        point_map: 3D point map.
        dsm: DSM raster.
        dop: DOP raster.
        covisibility_ratio: Desired covisibility ratio.
        direction: Direction to crop ('top', 'bottom', 'left', 'right').
        plot: Whether to plot the cropped rasters.

    Returns:
        Cropped DSM, DOP, and final covisibility ratio.
    """
    point_map_in_dsm_mask = grid_pts3d_to_mask(pts2d=point_map[np.isfinite(point_map[:, :,
                                                                           2]), :2], top_left=dsm[0, 0, :2],
                                               bottom_right=dsm[-1, -1, :2], width=dsm.shape[1], height=dsm.shape[0])
    dsm_mask = np.isfinite(dsm[:, :, 2])
    intersection_mask = point_map_in_dsm_mask & dsm_mask
    max_visibility = np.sum(intersection_mask)
    sum_j = np.sum(intersection_mask, 1)
    sum_i = np.sum(intersection_mask, 0)

    point_map_in_dsm_mask_copy = point_map_in_dsm_mask.copy() if plot else None
    dsm_mask_copy = dsm_mask.copy() if plot else None
    dop_copy = dop.copy() if plot else None

    h, w = dsm.shape[:2]
    if direction is None:
        direction = np.random.choice(['top', 'bottom', 'left', 'right'])

    if direction == 'top':
        sum_j_from_top = np.cumsum(sum_j)
        idx = np.argmax(sum_j_from_top >= covisibility_ratio * max_visibility)
        dsm = dsm[:idx, :, :]
        dop = dop[:idx, :, :]
        point_map_in_dsm_mask = point_map_in_dsm_mask[:idx, :]
        if point_map_in_dsm_mask_copy is not None:
            point_map_in_dsm_mask_copy[idx:, :] = 0
        if dsm_mask_copy is not None:
            dsm_mask_copy[idx:, :] = 0
    elif direction == 'bottom':
        sum_j_from_bottom = np.cumsum(sum_j[::-1])
        idx = np.argmax(sum_j_from_bottom >= covisibility_ratio * max_visibility)
        dsm = dsm[h - idx:, :, :]
        dop = dop[h - idx:, :, :]
        point_map_in_dsm_mask = point_map_in_dsm_mask[h - idx:, :]
        if point_map_in_dsm_mask_copy is not None:
            point_map_in_dsm_mask_copy[:h - idx, :] = 0
        if dsm_mask_copy is not None:
            dsm_mask_copy[:h - idx, :] = 0
    elif direction == 'left':
        sum_i_from_left = np.cumsum(sum_i)
        idx = np.argmax(sum_i_from_left >= covisibility_ratio * max_visibility)
        dsm = dsm[:, :idx, :]
        dop = dop[:, :idx, :]
        point_map_in_dsm_mask = point_map_in_dsm_mask[:, :idx]
        if point_map_in_dsm_mask_copy is not None:
            point_map_in_dsm_mask_copy[:, idx:] = 0
        if dsm_mask_copy is not None:
            dsm_mask_copy[:, idx:] = 0
    elif direction == 'right':
        sum_i_from_right = np.cumsum(sum_i[::-1])
        idx = np.argmax(sum_i_from_right >= covisibility_ratio * max_visibility)
        dsm = dsm[:, w - idx:, :]
        dop = dop[:, w - idx:, :]
        point_map_in_dsm_mask = point_map_in_dsm_mask[:, w - idx:]
        if point_map_in_dsm_mask_copy is not None:
            point_map_in_dsm_mask_copy[:, :w - idx] = 0
        if dsm_mask_copy is not None:
            dsm_mask_copy[:, :w - idx] = 0

    if plot and dop_copy is not None and point_map_in_dsm_mask_copy is not None:
        intersection_mask_rgb = utils.image.mask_to_rgba(intersection_mask & dsm_mask, to_rgb('blue'))
        point_map_in_dsm_mask_rgb = utils.image.mask_to_rgba(point_map_in_dsm_mask_copy & dsm_mask, to_rgb('red'))
        dsm_mask_rgb = utils.image.mask_to_rgba(dsm_mask_copy & dsm_mask, to_rgb('yellow'))

        dop_copy = np.concatenate([dop_copy, np.expand_dims(dsm_mask, axis=-1).astype(np.uint8) * 255], axis=-1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(dop_copy)
        ax.axis('off')
        ax.imshow(dsm_mask_rgb, alpha=0.6)
        ax.imshow(intersection_mask_rgb, alpha=0.6)
        if covisibility_ratio != 1.0:
            ax.imshow(point_map_in_dsm_mask_rgb, alpha=0.6)
        handles = []
        handles.append(mpatches.Patch(color='blue', label='Full Covisibility'))
        if covisibility_ratio != 1.0:
            handles.append(mpatches.Patch(color='red', label=f'{int(covisibility_ratio * 100)}% Covisibility'))
        handles.append(mpatches.Patch(color='yellow', label='Visible Region in DOP and DSM'))
        ax.legend(handles=handles, loc='lower right', fontsize=10)
        fig.show()

    final_covisbility_ratio = np.sum(point_map_in_dsm_mask) / max_visibility

    return dsm, dop, final_covisbility_ratio
