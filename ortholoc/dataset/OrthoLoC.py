from __future__ import annotations

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from loguru import logger
import random
import zipfile
import matplotlib.pyplot as plt
from torchvision.transforms import functional
from torch.nn import functional as F
from collections import defaultdict

from ortholoc import utils
from ortholoc.correspondences import Correspondences2D2D


class OrthoLoC(Dataset):
    def __init__(self, dirpath: str | None = None, sample_paths: list[str] | None = None, seed=47, start: float = 0.,
                 end: float = 1., use_refined_extrinsics: bool = False, mode: int = 0,
                 new_size: tuple[int, int] | None = None, limit_size: float | None = None, shuffle: bool = True,
                 scale_query_image: float = 1.0, scale_dop_dsm: float = 1.0, gt_matching_confidences_decay: float = 1.0,
                 covisibility_ratio: float = 1.0, return_tensor: bool = True) -> None:
        """
        Args:
            dirpath: path to the directory containing the samples
            sample_paths: list of paths to the samples
            seed: random seed for shuffling the samples
            start: start ratio of the samples to load
            end: end ratio of the samples to load
            use_refined_extrinsics: use refined extrinsics
            new_size: new size of the images
            limit_size: limit the number of samples to load
            shuffle: shuffle the samples
            scale_query_image: scale factor for the query image
            scale_dop_dsm: scale factor for the DOP and DSM images
            gt_matching_confidences_decay: decay factor for the matching confidences
            covisibility_ratio: ratio of the covisible area to keep
            return_tensor: return the images as tensors
            mode: mode for loading the samples
                0: all samples with different DOP and DSM domains
                1: all samples with different DOP domain only
                2: only samples with different DOP domain only
                3: only samples with different DOP and DSM domains
        """
        assert mode in (0, 1, 2, 3)

        assert (dirpath is not None) ^ (sample_paths is not None), 'Either dirpath or sample_paths should be provided'
        assert (new_size is not None) ^ (scale_query_image
                                         is not None), 'Either new_size or scale_query_image should be provided'
        assert (new_size is not None) ^ (scale_dop_dsm
                                         is not None), 'Either new_size or scale_dop_dsm should be provided'

        if dirpath is not None:
            dirpath = utils.misc.resolve_asset_path(dirpath, verbose=False)
        if sample_paths is not None:
            for i, sample_path in enumerate(sample_paths):
                sample_paths[i] = utils.misc.resolve_asset_path(sample_path, verbose=False)

        self.name = os.path.basename(dirpath) if dirpath else ''
        self.dirpath = dirpath
        self.return_tensor = return_tensor
        self.overfitting_mode = False
        self.use_refined_extrinsics = use_refined_extrinsics
        self.scale_query_image = scale_query_image
        self.scale_dop_dsm = scale_dop_dsm
        self.gt_matching_confidences_decay = gt_matching_confidences_decay
        self.covisibility_ratio = covisibility_ratio
        assert covisibility_ratio <= 1.0, 'Covisibility ratio should be between 0 and 1'
        assert start < end
        random.seed(seed)
        if dirpath is not None:
            assert os.path.exists(dirpath), f'Directory {dirpath} does not exist'
            self.files = sorted(utils.io.find_files(dirpath, suffix='.npz', recursive=True))
        else:
            self.files = sample_paths
        if mode == 1:
            self.files = list(filter(lambda f: 'R' in f, self.files))
        elif mode == 2:
            self.files = list(filter(lambda f: 'xDOP' in f and 'xDOPDSM' not in f, self.files))
        elif mode == 3:
            self.files = list(filter(lambda f: 'xDOPDSM' in f, self.files))

        if shuffle:
            random.shuffle(self.files)
        self.files = self.files[int(len(self.files) * start):int(len(self.files) * end)]
        if limit_size is not None:
            if 0 < limit_size < 1.0:
                self.files = self.files[:int(len(self.files) * limit_size)]
            elif limit_size >= 1.0 and int(limit_size) == limit_size:
                self.files = self.files[:int(limit_size)]
            elif limit_size < 0:  # for overfitting tests
                self.files = [self.files[0]] * abs(int(limit_size))
                self.overfitting_mode = True

        self.sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.files]
        self.new_size = new_size

        assert len(self) > 0, f'Dataset is empty. Make sure that the path contains .npz files: {dirpath}' \
            if dirpath else 'No samples provided'

    def __len__(self) -> int:
        return len(self.files)

    def shuffle(self) -> None:
        """
        Shuffle the dataset.
        """
        random.shuffle(self.files)
        self.sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.files]

    def get_sample(self, sample_id: str) -> dict:
        """
        Get a sample by its ID.
        """
        idx = self.sample_ids.index(sample_id)
        return self.getitem_np(idx)

    @property
    def sample_ids_by_scene(self) -> dict:
        """
        Group sample IDs by scene ID.
        """
        groups = defaultdict(list)
        for sample_id in self.sample_ids:
            scene_id = sample_id.split('_')[0]
            groups[scene_id].append(sample_id)
        return dict(groups)

    @property
    def sample_ids_by_type(self) -> dict:
        """
        Group sample IDs by type (xDOP, xDOPDSM, R).
        """
        groups = defaultdict(list)
        for sample_id in self.sample_ids:
            if '_xDOP' in sample_id and '_xDOPDSM' not in sample_id:
                groups['xDOP'].append(sample_id)
            elif '_xDOPDSM' in sample_id:
                groups['xDOPDSM'].append(sample_id)
            else:
                groups['R'].append(sample_id)
        return dict(groups)

    @staticmethod
    def group_sample_ids_by_scene(sample_ids: list[str]) -> dict:
        """
        Group sample IDs by scene ID.
        """
        groups = defaultdict(list)
        for sample_id in sample_ids:
            scene_id = sample_id.split('_')[0]
            groups[scene_id].append(sample_id)
        return dict(groups)

    @staticmethod
    def scale_intrinsic(K: torch.Tensor, wi: int, hi: int, size: tuple[int, int]) -> torch.Tensor:
        """
        Scale the intrinsic matrix K to the new size.
        """
        sx = size[0] / wi
        sy = size[1] / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    @staticmethod
    def get_correspondences_2d2d(
        sample: dict,
        normalized: bool = False,
        covisible_only=False,
        sampling_pts2d: np.ndarray | None = None,
    ) -> Correspondences2D2D:
        """
        Compute the GT 2D-2D correspondences between the query and DOP images.
        """
        point_map = sample['point_map']
        dsm = sample['dsm']
        if 'mask_dsm' in sample:
            dsm[~sample['mask_dsm'], 2] = np.nan
        if 'mask_point_map' in point_map:
            point_map[~sample['mask_point_map']] = np.nan

        h_query, w_query = point_map.shape[:2]
        h_dop, w_dop = dsm.shape[:2]

        if sampling_pts2d is None:
            pts0 = utils.geometry.create_grid_pts2d(h=h_query, w=w_query, normalized=True).reshape((-1, 2))
            pts1 = sample['matches'].reshape((-1, 2))
            confidences = sample['matching_confidences'].reshape((-1))
        else:
            pts0 = sampling_pts2d
            pts1 = utils.geometry.sample_grid(sample['matches'], sampling_pts2d, mode='nearest',
                                              align_corners=True).reshape((-1, 2))
            confidences = utils.geometry.sample_grid(sample['matching_confidences'], sampling_pts2d, mode='nearest',
                                                     align_corners=True).reshape((-1))
        query_to_dop_correspondences_2d2d = Correspondences2D2D(pts0=pts0, pts1=pts1, confidences=confidences,
                                                                is_normalized=True)

        if covisible_only:
            query_to_dop_correspondences_2d2d = query_to_dop_correspondences_2d2d.take_min_max_pts0(
                min_val_0=-1, max_val_0=1, min_val_1=-1, max_val_1=1)
            query_to_dop_correspondences_2d2d = query_to_dop_correspondences_2d2d.take_min_max_pts1(
                min_val_0=-1, max_val_0=1, min_val_1=-1, max_val_1=1)
            query_to_dop_correspondences_2d2d = query_to_dop_correspondences_2d2d.take_finite()
        if not normalized:
            query_to_dop_correspondences_2d2d = query_to_dop_correspondences_2d2d.denormalized(
                w0=w_query, h0=h_query, w1=w_dop, h1=h_dop)
        return query_to_dop_correspondences_2d2d

    @staticmethod
    def compute_matching_error(sample: dict, correspondences_2d2d: Correspondences2D2D) -> np.ndarray:
        """
        Compute the matching error between the GT correspondences (from sample) and the predicted correspondences_2d2d.
        """
        h_query, w_query = sample['image_query'].shape[:2]
        h_dop, w_dop = sample['image_dop'].shape[:2]
        correspondences_2d2d = correspondences_2d2d.normalized(w0=w_query, h0=h_query, w1=w_dop, h1=h_dop)
        correspondences_2d2d_gt = OrthoLoC.get_correspondences_2d2d(sample=sample, normalized=False,
                                                                    covisible_only=False,
                                                                    sampling_pts2d=correspondences_2d2d.pts0)
        correspondences_2d2d = correspondences_2d2d.denormalized(w0=w_query, h0=h_query, w1=w_dop, h1=h_dop)
        return np.linalg.norm(correspondences_2d2d_gt.pts1 - correspondences_2d2d.pts1, axis=-1)

    def getitem_np(self, idx: int) -> dict:
        """
        Get a sample by its index.
        """
        file_path = self.files[idx]
        try:
            data = np.load(file_path)
            image_query = data['image_query']
            if np.all(image_query == 0):
                if file_path in self.files:
                    self.files.remove(file_path)
                if os.path.exists(file_path):
                    os.remove(file_path)
                logger.warning(f"Empty image found in {file_path}")
                return self.getitem_np(idx)
        except (zipfile.BadZipFile, OSError, ValueError, EOFError):
            if os.path.exists(file_path):
                raise IOError(f"Bad Zip file {file_path}")
            else:
                raise IOError(f"File {file_path} not found")

        # get raw data
        h_query, w_query = image_query.shape[:2]
        dsm = data['dsm']
        point_map = data['point_map']
        image_dop = data['image_dop']

        # crop the rasters
        if self.covisibility_ratio < 1.0:
            dsm, image_dop, _ = utils.geometry.crop_rasters(point_map=point_map, dsm=dsm, dop=image_dop,
                                                            covisibility_ratio=self.covisibility_ratio)
        h_dop, w_dop = image_dop.shape[:2]
        assert dsm.shape[0] == h_dop and dsm.shape[
            1] == w_dop, f"DSM shape {dsm.shape} does not match image_dop shape {image_dop.shape}"
        offset = np.array([dsm[0, 0][0], dsm[0, 0][1]])
        assert not np.any(np.isnan(offset))
        scale = data['scale']
        intrinsics_query = data['intrinsics'].astype(np.float32)

        # compute masks
        mask_dsm = np.isfinite(dsm).all(axis=-1)
        dsm[~mask_dsm, :] = 0
        assert not np.any(np.isnan(dsm))
        mask_point_map = np.isfinite(point_map).all(axis=-1)
        point_map[~mask_point_map, :] = 0
        assert not np.any(np.isnan(point_map))

        # compute GT correspondences with full resolution
        pose_world2dop, intrinsics_dop = utils.pose.compute_raster_intrinsics_extrinsics(scale=scale, offset=offset)
        query_to_dop_correspondences_2d2d = Correspondences2D2D.from_grids3d(grid3d_0=point_map, grid3d_1=dsm,
                                                                             pose_w2c_1=pose_world2dop,
                                                                             intrinsics_1=intrinsics_dop,
                                                                             decay=self.gt_matching_confidences_decay,
                                                                             normalized=True)
        matches = query_to_dop_correspondences_2d2d.pts1.reshape((h_query, w_query, 2))
        matching_confidences = query_to_dop_correspondences_2d2d.confidences.reshape((h_query, w_query, 1))

        # resize
        size_query, size_dop_dsm = None, None
        if self.new_size is not None:
            size_query = self.new_size
            size_dop_dsm = self.new_size
        else:
            if self.scale_query_image != 1.0:
                size_query = (int(w_query * self.scale_query_image), int(h_query * self.scale_query_image))
            if self.scale_dop_dsm != 1.0:
                size_dop_dsm = (int(w_dop * self.scale_dop_dsm), int(h_dop * self.scale_dop_dsm))

        if size_query is not None:
            image_query = self.resize_image(image_query, mode='linear', size=size_query)
            mask_point_map = self.resize_image(mask_point_map.astype(np.uint8), size=size_query,
                                               mode='nearest').astype(bool)
            intrinsics_query = self.scale_intrinsic(torch.from_numpy(intrinsics_query), hi=h_query, wi=w_query,
                                                    size=size_query).numpy()
            point_map = self.resize_map(point_map, size=size_query, align_corners=False, mode='bilinear')
            matches = self.resize_map(matches, size=size_query, align_corners=False, mode='bilinear')
            matching_confidences = self.resize_map(matching_confidences, size=size_query, align_corners=False,
                                                   mode='bilinear')

        if size_dop_dsm is not None:
            image_dop = self.resize_image(image_dop, mode='linear', size=size_dop_dsm)
            mask_dsm = self.resize_image(mask_dsm.astype(np.uint8), size=size_dop_dsm, mode='nearest').astype(bool)
            dsm = self.resize_map(dsm, size=size_dop_dsm, align_corners=False, mode='bilinear')
            scale_dsm = np.array(size_dop_dsm) / np.array([w_dop, h_dop])
            scale = scale / scale_dsm
        h_query, w_query = image_query.shape[:2]
        h_dop, w_dop = image_dop.shape[:2]

        # compute camera params
        pose_world2query = data['extrinsics' if not self.use_refined_extrinsics else 'extrinsics_refined'].astype(
            np.float32)
        pose_query2world = utils.pose.inv_pose(pose_world2query)
        pose_world2dop, intrinsics_dop = utils.pose.compute_raster_intrinsics_extrinsics(scale=scale, offset=offset)
        pose_world2query = np.linalg.inv(np.concatenate([pose_query2world, np.array([[0, 0, 0, 1]])],
                                                        axis=0))[:3].astype(pose_query2world.dtype)

        # resize data
        output = {
            'sample_id': self.sample_ids[idx],
            'sample_id_original': data['sample_id'],
            'path': file_path,
            'image_query': image_query,  # query
            'point_map': point_map,  # point map
            'mask_point_map': mask_point_map,  # point map mask
            'intrinsics_query': intrinsics_query,  # intrinsics query
            'pose_world2query': pose_world2query,  # world to query
            'pose_query2world': pose_query2world,  # query to world
            'image_dop': image_dop,  # image_dop
            'dsm': dsm,  # dsm
            'mask_dsm': mask_dsm,  # dsm mask
            'vertices': data['vertices'],  # vertices
            'faces': data['faces'],  # normals
            'scale': scale,  # scale
            'offset': offset,  # offset
            'intrinsics_dop': intrinsics_dop,  # image_dop intrinsics
            'pose_world2dop': pose_world2dop,  # world to image_dop
            'matches': matches,
            'matching_confidences': matching_confidences,
            'keypoints': data['keypoints'],
            'h_query': h_query,  # original query height
            'w_query': w_query,  # original query width
            'h_dop': h_dop,  # original image_dop height
            'w_dop': w_dop,  # original image_dop width
        }
        return output

    @staticmethod
    def resize_tensor(array: torch.Tensor, size: tuple[int, int], align_corners: bool | None = False,
                      mode='bilinear') -> torch.Tensor:
        """
        Resize a tensor to the given size using interpolation.
        """
        n_dim = array.ndim
        assert n_dim in (2, 3, 4)
        if n_dim == 2:
            array = array.unsqueeze(0).unsqueeze(0)
        elif n_dim == 3:
            array = array.unsqueeze(0)
        output = F.interpolate(array, size=(size[1], size[0]), mode=mode, align_corners=align_corners)
        return output[0] if n_dim == 3 else (output[0, 0] if n_dim == 2 else output)

    @staticmethod
    def resize_image(array: np.ndarray, size: tuple[int, int], mode='linear') -> np.ndarray:
        """
        Resize an image to the given size using interpolation.
        """
        return cv2.resize(array, size, interpolation=cv2.INTER_LINEAR if mode == 'linear' else cv2.INTER_NEAREST)

    def resize_map(self, array: np.ndarray, size: tuple[int, int], align_corners: bool | None = False,
                   mode='bilinear') -> np.ndarray:
        """
        Resize a map to the given size using interpolation.
        """
        return self.resize_tensor(
            torch.from_numpy(array).permute(2, 0, 1), size, align_corners=align_corners, mode=mode).permute(1, 2,
                                                                                                            0).numpy()

    @staticmethod
    def points_map_to_depth(points_map: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
        """
        Convert a point map to a depth map using the camera pose.
        """
        pose_world2query = np.linalg.inv(np.concatenate([pose_c2w, np.array([[0, 0, 0, 1]])], axis=0))
        pts3d = points_map.reshape((-1, 3))
        pts3d_hom = np.concatenate([pts3d, np.ones((pts3d.shape[0], 1))], axis=1)
        scm_c = (pose_world2query @ pts3d_hom.T).T[:, :3]
        return scm_c.reshape(points_map.shape)[:, :, 2].astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        if self.overfitting_mode and hasattr(self, 'overfitting_data'):
            return self.overfitting_data

        if idx >= len(self.files):
            logger.warning(f"Index {idx} out of range. Taking modulo {idx % len(self.files)}")
            return self.__getitem__(idx % len(self.files))

        # for idx in range(len(self)):
        data = self.getitem_np(idx)

        if not self.return_tensor:
            return data

        #################
        # convert to tensors
        #################

        data_dict = {}
        for k, v in data.items():
            if k.startswith('I_'):
                data_dict[k] = functional.to_tensor(v)
            elif isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v.astype(np.float32))
            else:
                data_dict[k] = v

        return data_dict

    def plot_sample(self, sample_id: str, n_rows: int = 1, n_cols: int = 4, figsize: tuple[float, float] = (15, 3.2),
                    show: bool = False, title: str = '', fontsize: int = 12, dsm_vmin: float | None = None,
                    dsm_vmax: float | None = None, show_sample_id: bool = True, axs: list[plt.Axes] | None = None,
                    subtitles: bool = True) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plot a sample with its query image, point map (as depth map), DOP image, and DSM image.
        """
        # Create figure and subplots
        if axs is None:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
            axs = axs.flatten()
        assert len(axs) >= 4, 'At least 4 subplots should be provided'
        fig = axs[0].figure

        sample_idx = self.sample_ids.index(sample_id)
        sample = self.getitem_np(sample_idx)

        # Apply masks for DSM and point_map
        if 'dsm' in sample and 'mask_dsm' in sample:
            sample['dsm'][~sample['mask_dsm'], 2] = np.nan
        if 'image_dop' in sample and 'mask_dsm' in sample:
            sample['image_dop'][~sample['mask_dsm'], :] = 255
        if 'point_map' in sample and 'mask_point_map' in sample:
            sample['point_map'][~sample['mask_point_map']] = np.nan

        # For RGB images, pad with white (assuming float images in [0,1])
        query_img = utils.image.pad_to_square(sample['image_query'], pad_value=255)
        dop_img = utils.image.pad_to_square(sample['image_dop'], pad_value=255)

        # For scalar data (depth and elevation), pad with np.nan.
        point_map = sample['point_map']
        extrinsics = sample['pose_world2query']
        depth_map = utils.geometry.point_to_depth_map(point_map, extrinsics=extrinsics)
        depth_map = utils.image.pad_to_square(depth_map, pad_value=np.nan)
        depth_vmin, depth_vmax = np.nanmin(depth_map), np.nanmax(depth_map)

        dsm = sample['dsm']

        dsm_elevations = dsm[:, :, 2]
        dsm_elevations = utils.image.pad_to_square(dsm_elevations, pad_value=np.nan)
        if dsm_vmin is None:
            dsm_vmin = np.nanmin(dsm_elevations)
        if dsm_vmax is None:
            dsm_vmax = np.nanmax(dsm_elevations)

        # Query image
        axs[0].imshow(query_img, aspect='equal')
        if subtitles:
            axs[0].set_title('Query Image', fontsize=fontsize)
        axs[0].axis('off')
        if show_sample_id:
            axs[0].text(-0.08, 0.5, sample_id, va='center', ha='left', rotation=90, transform=axs[0].transAxes,
                        fontsize=fontsize)

        # Depth image with colormap and colorbar
        im_depth = axs[1].imshow(depth_map, cmap='plasma', aspect='equal', vmin=depth_vmin, vmax=depth_vmax)
        cbar1 = fig.colorbar(im_depth, ax=axs[1])
        cbar1.set_label('Depth (m)', fontsize=fontsize)
        cbar1.ax.tick_params(labelsize=fontsize - 2)
        if subtitles:
            axs[1].set_title('Point Map', fontsize=fontsize)
        axs[1].axis('off')

        # DOP image
        axs[2].imshow(dop_img, aspect='equal')
        if subtitles:
            axs[2].set_title('DOP', fontsize=fontsize)
        axs[2].axis('off')

        # DSM image with colormap and colorbar
        im_dsm = axs[3].imshow(dsm_elevations, cmap='viridis', aspect='equal', vmin=dsm_vmin, vmax=dsm_vmax)
        cbar2 = fig.colorbar(im_dsm, ax=axs[3])
        cbar2.set_label('Elevation (m)', fontsize=fontsize)
        cbar2.ax.tick_params(labelsize=fontsize - 2)
        if subtitles:
            axs[3].set_title('DSM', fontsize=fontsize)
        axs[3].axis('off')

        if title:
            fig.suptitle(title, fontsize=fontsize)

        if show:
            plt.show()
        return fig, axs
