from __future__ import annotations

import cv2
from typing import Any
from typing_extensions import Self
from matplotlib import lines
from matplotlib import pyplot as plt
import numpy as np
from loguru import logger

from imcui.ui.utils import ransac_zoo

from ortholoc import utils
from ortholoc.correspondences.Correspondences import Correspondences
from ortholoc.correspondences.Correspondences2D3D import Correspondences2D3D


class Correspondences2D2D(Correspondences):
    """
    Class to represent 2D-2D correspondences.
    """

    pts0: np.ndarray = np.empty((0, 2))
    pts1: np.ndarray = np.empty((0, 2))
    is_normalized: bool
    confidences: np.ndarray | None = None

    def __setattr__(self, key: str, value: Any) -> None:
        if key == 'pts0' or key == 'pts1':
            if value.ndim != 2 or value.shape[1] != 2:
                raise ValueError(f"Invalid shape for {key}: {value.shape}")
        super().__setattr__(key, value)

    @classmethod
    def from_grid3d(cls, grid3d_w: np.ndarray, pose_w2c: np.ndarray, intrinsics: np.ndarray,
                    sampling_pts2d: np.ndarray | None = None) -> Self:
        """
        Create Correspondences2D2D from a projected 3D grid on the camera and coordinates
        in the image plane (sampling_pts2d if provided else each pixel in the image).
        """
        assert grid3d_w.ndim == 3, "grid3d_w should be a 3D array"
        assert grid3d_w.shape[-1] == 3, "grid3d_w should describe 3D points"
        h, w = grid3d_w.shape[0], grid3d_w.shape[1]
        if sampling_pts2d is None:
            x1_pix = utils.geometry.create_grid_pts2d(w=w, h=h, normalized=False).reshape((-1, 2))
            pts3d = grid3d_w.reshape(-1, 3)
        else:
            if not utils.geometry.pts2d_is_normalized(sampling_pts2d):
                sampling_pts2d = utils.geometry.norm_pts2d(sampling_pts2d, h=h, w=w)
            pts3d = utils.geometry.sample_grid(grid=grid3d_w, pts2d=sampling_pts2d, mode='nearest', align_corners=True)
            x1_pix = utils.geometry.denorm_pts2d(sampling_pts2d, h=h, w=w)
        x_1in2_pix = utils.geometry.project_pts3d(pts3d=pts3d, pose_w2c=pose_w2c, intrinsics=intrinsics)
        return Correspondences2D2D(x1_pix, x_1in2_pix, is_normalized=False)

    @classmethod
    def from_grids3d(cls, grid3d_0: np.ndarray, grid3d_1: np.ndarray, pose_w2c_1: np.ndarray, intrinsics_1: np.ndarray,
                     decay: float | None = 1.0, sampling_pts2d: np.ndarray | None = None,
                     normalized: bool = False) -> Self:
        """
        Create Correspondences2D2D from a projected 3D grid (0) on the camera (1) with known 3D point map (grid3d_1)
        of the camera (1). All pixels in the image will be considered in the correspondences if sampling_pts2d is None.
        The correspondences will have confidence values based on the distance between the two grids.
        """
        assert grid3d_0.ndim == 3, "grid3d_0 should be a 3D array"
        assert grid3d_0.shape[-1] == 3, "grid3d_0 should describe 3D points"
        assert grid3d_1.ndim == 3, "grid3d_1 should be a 3D array"
        assert grid3d_1.shape[-1] == 3, "grid3d_1 should describe 3D points"

        correspondences_2d2d = cls.from_grid3d(grid3d_w=grid3d_0, pose_w2c=pose_w2c_1, intrinsics=intrinsics_1,
                                               sampling_pts2d=sampling_pts2d)

        distances = utils.geometry.compute_dist(grid3d_0=grid3d_0, grid3d_1=grid3d_1,
                                                correspondences_2d2d=correspondences_2d2d, mode='nearest',
                                                align_corners=True)
        correspondences_2d2d.confidences = utils.geometry.dist_to_confidences(
            distances=distances, decay=decay) if decay is not None else distances
        if normalized:
            correspondences_2d2d = correspondences_2d2d.normalized(w0=grid3d_0.shape[1], h0=grid3d_0.shape[0],
                                                                   w1=grid3d_1.shape[1], h1=grid3d_1.shape[0])
        return correspondences_2d2d

    def compute_geometry(self, w0: int, h0: int, ransac_method: str = "CV2_USAC_MAGSAC",
                         ransac_reproj_threshold: float = 8, ransac_confidence: float = 0.9999,
                         ransac_max_iter: int = 10000, min_num_matches: int = 4, geometry: str = 'Fundamental',
                         silent: bool = False) -> tuple[np.ndarray, ...]:
        """
        Compute the fundamental matrix and homography matrix using RANSAC.
        """
        assert not self.is_normalized
        if ransac_method not in ransac_zoo.keys():
            ransac_method = ransac_method

        mask_f, mask_h, F, H, H1, H2 = None, None, None, None, None, None

        if len(self) > min_num_matches:

            F, mask_f = utils.pose.proc_ransac_matches(
                self.pts0,
                self.pts1,
                ransac_method,
                ransac_reproj_threshold,
                ransac_confidence,
                ransac_max_iter,
                silent=silent,
                geometry_type="Fundamental",
            )

            if geometry == 'Homography':
                H, mask_h = utils.pose.proc_ransac_matches(
                    self.pts1,
                    self.pts0,
                    ransac_method,
                    ransac_reproj_threshold,
                    ransac_confidence,
                    ransac_max_iter,
                    silent=silent,
                    geometry_type="Homography",
                )

                if H is not None:
                    try:
                        _, H1, H2 = cv2.stereoRectifyUncalibrated(
                            self.pts0.reshape(-1, 2),
                            self.pts1.reshape(-1, 2),
                            F,
                            imgSize=(w0, h0),
                        )
                    except cv2.error as e:
                        logger.error(f"StereoRectifyUncalibrated failed, skip! error: {e}")

        return mask_f, mask_h, F, H, H1, H2

    def take_covisible(
        self,
        w0: int,
        h0: int,
        w1: int,
        h1: int,
        is_normalized: bool | None = None,
    ) -> Self:
        """
        Filter correspondences by taking only those that are within the image bounds.
        """
        if is_normalized is not None:
            is_normalized = self.is_normalized
        if not is_normalized:
            mask = np.ones(len(self.pts0), dtype=bool)
            mask[self.pts0[:, 0] < 0] = False
            mask[self.pts0[:, 0] > w0 - 1] = False
            mask[self.pts0[:, 1] < 0] = False
            mask[self.pts0[:, 1] > h0 - 1] = False
            mask[self.pts1[:, 0] < 0] = False
            mask[self.pts1[:, 0] > w1 - 1] = False
            mask[self.pts1[:, 1] < 0] = False
            mask[self.pts1[:, 1] > h1 - 1] = False
        else:
            mask = np.ones(len(self.pts0), dtype=bool)
            mask[self.pts0[:, 0] < -1] = False
            mask[self.pts0[:, 0] > 1] = False
            mask[self.pts0[:, 1] < -1] = False
            mask[self.pts0[:, 1] > 1] = False
            mask[self.pts1[:, 0] < -1] = False
            mask[self.pts1[:, 0] > 1] = False
            mask[self.pts1[:, 1] < -1] = False
            mask[self.pts1[:, 1] > 1] = False
        return self.take(mask)

    def normalized(self, w0: int, h0: int, w1: int, h1: int) -> Self:
        """
        Normalize the points in the correspondences to be in the range [-1, 1].
        """
        if not self.is_normalized:
            pts0 = utils.geometry.norm_pts2d(self.pts0, w=w0, h=h0)
            pts1 = utils.geometry.norm_pts2d(self.pts1, w=w1, h=h1)
            return Correspondences2D2D(pts0=pts0, pts1=pts1, confidences=self.confidences, is_normalized=True)
        return self

    def denormalized(self, w0: int, h0: int, w1: int, h1: int) -> Self:
        """
        Denormalize the points in the correspondences to be in the range [0, w-1] and [0, h-1].
        """
        if self.is_normalized:
            pts0 = utils.geometry.denorm_pts2d(self.pts0, w=w0, h=h0)
            pts1 = utils.geometry.denorm_pts2d(self.pts1, w=w1, h=h1)
            return Correspondences2D2D(pts0=pts0, pts1=pts1, confidences=self.confidences, is_normalized=False)
        return self

    def to_2d3d(self, grid3d_0: np.ndarray | None = None, grid3d_1: np.ndarray | None = None) -> Correspondences2D3D:
        """
        Convert the 2D-2D correspondences to 2D-3D correspondences using the provided 3D grids.
        """
        assert self.is_normalized, 'Correspondences should be normalized'
        assert (grid3d_0 is not None) ^ (grid3d_1 is not None), 'Either grid3d_0 or grid3d_1 should be defined'
        if grid3d_0 is not None:
            pts3d = utils.geometry.sample_grid(grid=grid3d_0, pts2d=self.pts0, mode='nearest', align_corners=True,
                                               is_normalized=self.is_normalized)
            pts2d = self.pts1
        elif grid3d_1 is not None:
            pts2d = self.pts0
            pts3d = utils.geometry.sample_grid(grid=grid3d_1, pts2d=self.pts1, mode='nearest', align_corners=True,
                                               is_normalized=self.is_normalized)
        else:
            raise NotImplementedError
        return Correspondences2D3D(pts2d, pts3d, confidences=self.confidences, is_normalized=self.is_normalized)

    def concatenate(self, other: Self) -> Self:
        """
        Concatenate two Correspondences2D2D objects.
        """
        assert self.is_normalized == other.is_normalized, 'Correspondences should be either both normalized or denormalized'
        pts0 = np.concatenate((self.pts0, other.pts0), axis=0)
        pts1 = np.concatenate((self.pts1, other.pts1), axis=0)
        confidences = np.concatenate(
            (self.confidences, other.confidences), axis=0) if self.confidences is not None else None
        return Correspondences2D2D(pts0=pts0, pts1=pts1, confidences=confidences, is_normalized=self.is_normalized)

    @property
    def inv(self) -> Self:
        """
        Invert the correspondences.
        """
        return Correspondences2D2D(pts0=self.pts1, pts1=self.pts0, confidences=self.confidences,
                                   is_normalized=self.is_normalized)

    def plot(self, img1: np.ndarray, img2: np.ndarray, img1_name: str | None = None, img2_name: str | None = None,
             axes: list[plt.Axes] | None = None, show: bool = False, title: str = '', vmin: float | None = None,
             vmax: float | None = None, alpha: float = 0.5, point_size: float = 10, linewidth: int = 2,
             arrows: bool = True, max_pts: int | None = None, cmap: str = 'jet', show_colorbar: bool = True,
             marker: str = 'o', fig_scale: float = 0.5, dpi: int = 100) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot the correspondences between two images.
        """
        correspondences_2d2d = self.take_finite()
        h0, w0 = img1.shape[:2]
        h1, w1 = img2.shape[:2]
        if correspondences_2d2d.is_normalized:
            correspondences_2d2d = correspondences_2d2d.denormalized(w0=w0, h0=h0, w1=w1, h1=h1)

        if max_pts is None:
            max_pts = len(correspondences_2d2d)

        if axes is None:
            new_fig, new_axs = plt.subplots(
                1, 2, figsize=(int((img1.shape[1] + img2.shape[1]) * fig_scale / float(dpi)),
                               int(max(img1.shape[0], img2.shape[0]) * fig_scale / float(dpi))),
                constrained_layout=True, dpi=dpi)
            if isinstance(new_axs, np.ndarray):
                axes = new_axs.flatten().tolist()
            elif isinstance(new_axs, plt.Axes):
                axes = [new_axs]
        assert isinstance(axes, list) and len(axes) == 2, 'The axes should be a list of two axes.'
        fig = axes[0].figure
        assert isinstance(fig, plt.Figure)

        positions_1 = correspondences_2d2d.pts0
        positions_2 = correspondences_2d2d.pts1
        confidences = np.asarray(
            correspondences_2d2d.confidences) if correspondences_2d2d.confidences is not None else np.ones(
                len(correspondences_2d2d), dtype=float)

        if max_pts is not None and len(positions_1) > max_pts:
            logger.info(f'{max_pts} points will be randomly selected for plotting')
            idxs = np.random.choice(len(positions_1), max_pts, replace=False)
            positions_1 = positions_1[idxs]
            positions_2 = positions_2[idxs]
            confidences = confidences[idxs]
        else:
            max_pts = len(positions_1)
        if len(positions_1) > 1000:
            logger.warning('Too many matching pairs to plot. This may take a while')

        if vmax is None:
            vmax = confidences.max()
        if vmin is None:
            vmin = confidences.min()
            if vmax == vmin:
                vmin = 0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        plt_cmap = plt.get_cmap(cmap)
        colors = plt_cmap(norm(confidences))
        if img1_name is not None:
            axes[0].set_title(f'Frame: {img1_name}')
        if img2_name is not None:
            axes[1].set_title(f'Frame: {img2_name}')
        title = title.strip()
        title += f'\n2D Correspondences\n ({min(max_pts, len(correspondences_2d2d))} / {len(correspondences_2d2d)} are shown)'
        fig.suptitle(title.strip())
        axes[0].imshow(img1)
        axes[1].imshow(img2)
        fig.tight_layout(pad=1)
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=1.02)
        axes[0].scatter(positions_1[:, 0], positions_1[:, 1], s=point_size, c=colors[..., :3], alpha=alpha,
                        marker=marker)
        axes[1].scatter(positions_2[:, 0], positions_2[:, 1], s=point_size, c=colors[..., :3], alpha=alpha,
                        marker=marker)
        if show_colorbar:
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axes[1],
                orientation='vertical',
                shrink=0.8,
            )
            cbar.set_label('Confidence Level')
            ticks = np.linspace(vmin, vmax, num=5).tolist()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])

        if arrows:
            fig.canvas.draw()
            trans_figure = fig.transFigure.inverted()
            fpositions_1 = trans_figure.transform(axes[0].transData.transform(positions_1))
            fpositions_2 = trans_figure.transform(axes[1].transData.transform(positions_2))
            fig.lines = [
                lines.Line2D((fpositions_1[i, 0], fpositions_2[i, 0]), (fpositions_1[i, 1], fpositions_2[i, 1]),
                             transform=fig.transFigure, c=colors[i], linewidth=linewidth, alpha=alpha)
                for i in range(len(positions_1))
            ]
            # freeze the axes to prevent the transform to change
            axes[0].autoscale(enable=False)
            axes[1].autoscale(enable=False)

        if show:
            plt.show()

        return fig, axes
