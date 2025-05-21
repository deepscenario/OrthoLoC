from __future__ import annotations
import warnings

warnings.simplefilter("ignore")

import torch
import cv2
import os.path
from loguru import logger
from typing import Any
import numpy as np
from abc import abstractmethod

from ortholoc.correspondences import Correspondences2D2D


class Matcher:
    """
    Base class for image matching.
    """
    def __init__(self, angles: list[float], name: str, device: str = 'cuda') -> None:
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        if self.device.type == 'cpu':
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.name = name
        self.angles = angles
        assert len(self.angles) > 0

    @abstractmethod
    def __call__(self, img0: np.ndarray, img1: np.ndarray, sample_id: str | None = None, covisible_only: bool = True,
                 normalized: bool = True, silent: bool = False, *args: Any, **kwargs: Any) -> Correspondences2D2D:
        pass

    @staticmethod
    def build_correspondences(pts0: np.ndarray, pts1: np.ndarray, h0: int, w0: int, h1: int, w1: int,
                              normalized: bool = True, covisible_only: bool = True,
                              confidences: np.ndarray | None = None) -> Correspondences2D2D:
        """
        Build correspondences from 2D points. This is useful for converting numpy arrays to Correspondences2D2D objects.
        """
        if pts0.ndim == 1:
            pts0 = pts0[np.newaxis]
        if pts1.ndim == 1:
            pts1 = pts1[np.newaxis]
        confidences = confidences if confidences is not None else np.ones(len(pts0))
        confidences = np.squeeze(confidences) if len(confidences) != 1 else confidences
        correspondences_2d2d = Correspondences2D2D(pts0=pts0, pts1=pts1, confidences=confidences, is_normalized=False)
        correspondences_2d2d = correspondences_2d2d.normalized(w0=w0, h0=h0, w1=w1, h1=h1)
        correspondences_2d2d = correspondences_2d2d.take_finite()
        # keep covisible points only
        if covisible_only:
            correspondences_2d2d = correspondences_2d2d.take_min_max_pts0(min_val_0=-1, max_val_0=1, min_val_1=-1,
                                                                          max_val_1=1)
            correspondences_2d2d = correspondences_2d2d.take_min_max_pts1(min_val_0=-1, max_val_0=1, min_val_1=-1,
                                                                          max_val_1=1)
        if not normalized:
            correspondences_2d2d = correspondences_2d2d.denormalized(h0=h0, w0=w0, h1=h1, w1=w1)

        return correspondences_2d2d

    def run(self, img0: np.ndarray, img1: np.ndarray, silent: bool = False, covisible_only: bool = True,
            angles: list[float] | None = None, normalized: bool = True, *args: Any,
            **kwargs: Any) -> list[Correspondences2D2D]:
        """
        Run the matcher on a stereo pair of images. The function will rotate the first image by the specified angles
        """
        img0_height, img0_width = img0.shape[:2]
        img0_center = (img0_width // 2, img0_height // 2)

        all_coorespondences = []
        for angle in (angles or self.angles):
            if not silent:
                logger.info(f'Matching stereo pair at {angle} degrees')
            if angle != 0:
                # compute the destination image size
                theta = np.radians(angle)
                img0_rotated_width = int(np.floor(img0_height * abs(np.sin(theta)) + img0_width * abs(np.cos(theta))))
                img0_rotated_height = int(np.floor(img0_height * abs(np.cos(theta)) + img0_width * abs(np.sin(theta))))
                img0_rotated_size = (img0_rotated_width, img0_rotated_height)

                # compute the image transformation matrix
                rotation_matrix = cv2.getRotationMatrix2D(img0_center, angle, 1)
                rotation_matrix[0, 2] += ((img0_rotated_width / 2) - img0_center[0])
                rotation_matrix[1, 2] += ((img0_rotated_height / 2) - img0_center[1])

                # apply the transformation
                img0_rotated = cv2.warpAffine(np.array(img0), rotation_matrix, img0_rotated_size)
            else:
                img0_rotated = img0
                img0_rotated_width, img0_rotated_height = img0_width, img0_height

            # apply the matching with the rotated img0
            try:
                correspondences = self(img0_rotated, img1, covisible_only=covisible_only, normalized=normalized,
                                       silent=silent, *args, **kwargs)
            except IndexError as e:
                logger.error(f'Error in sample: {e}')
                all_coorespondences.append(
                    Correspondences2D2D(pts0=np.empty((0, 2)), pts1=np.empty((0, 2)), confidences=np.empty((0, )),
                                        is_normalized=False))
                continue

            if len(correspondences) > 0 and angle != 0:
                # compute the inverse transformation matrix
                correspondences = correspondences.denormalized(h0=img0_rotated_height, w0=img0_rotated_width,
                                                               h1=img1.shape[0], w1=img1.shape[1])
                img0_rotated_center = (img0_rotated_width // 2, img0_rotated_height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(img0_rotated_center, -angle, 1)
                rotation_matrix[0, 2] += ((img0_width / 2) - img0_rotated_center[0])
                rotation_matrix[1, 2] += ((img0_height / 2) - img0_rotated_center[1])
                # rotate the matched 2d points back to the original orientation
                correspondences.pts0 = cv2.transform(correspondences.pts0[np.newaxis], rotation_matrix)[0].round()
                if covisible_only:
                    correspondences = correspondences.take_min_max_pts0(min_val_0=0, max_val_0=img0_width - 1,
                                                                        min_val_1=0, max_val_1=img0_height - 1,
                                                                        inclusive=True)
                    correspondences = correspondences.take_min_max_pts1(min_val_0=0, max_val_0=img1.shape[1] - 1,
                                                                        min_val_1=0, max_val_1=img1.shape[0] - 1,
                                                                        inclusive=True)
                if normalized:
                    correspondences = correspondences.normalized(h0=img0_height, w0=img0_width, h1=img1.shape[0],
                                                                 w1=img1.shape[1])

            all_coorespondences.append(correspondences)
        return all_coorespondences

    def adhop(
        self,
        correspondences_2d2d: Correspondences2D2D,
        img0: np.ndarray,
        img1: np.ndarray,
        min_conf: float = 0.0,
        silent: bool = False,
        keep_at_least: int = 1000,
        normalized: bool = True,
    ) -> Correspondences2D2D:
        """
        Refine the correspondences using AdHop method.
        """
        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]

        correspondences_2d2d = correspondences_2d2d.denormalized(h0=h0, w0=w0, h1=h1, w1=w1)
        mask = ~np.all(img1 == 0, axis=-1)
        correspondences_2d2d = correspondences_2d2d.take_mask(mask[correspondences_2d2d.pts1[:, 1].astype(int),
                                                                   correspondences_2d2d.pts1[:, 0].astype(int)])

        try:
            mask_f, mask_h, F, H, H1, H2 = correspondences_2d2d.take_min_conf(0.0).compute_geometry(
                w0=w0, h0=h0, silent=silent, geometry='Homography')

            img1_rectified = cv2.warpPerspective(img1, H, (w0, h0))

            refined_correspondences_2d2d = \
                self.run(img0, img1_rectified, angles=[0.0], silent=silent, normalized=False)[0]
            refined_correspondences_2d2d.pts1 = cv2.perspectiveTransform(
                refined_correspondences_2d2d.pts1.reshape(-1, 1, 2), np.linalg.inv(H)).squeeze(1)

            refined_correspondences_2d2d = refined_correspondences_2d2d.take_min_conf(
                min_conf, keep_at_least=keep_at_least)
            refined_correspondences_2d2d = refined_correspondences_2d2d.take_covisible(h0=h0, w0=w0, h1=h1, w1=w1)
            assert len(refined_correspondences_2d2d) > 0, "No correspondences available after AdHop"
        except Exception as e:
            logger.info(f'AdHop failed: {e}')
            refined_correspondences_2d2d = correspondences_2d2d

        if normalized:
            refined_correspondences_2d2d = refined_correspondences_2d2d.normalized(h0=h0, w0=w0, h1=h1, w1=w1)

        return refined_correspondences_2d2d
