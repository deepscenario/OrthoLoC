from __future__ import annotations
from typing import Any
import numpy as np

from ortholoc.image_matching import Matcher
from ortholoc.correspondences import Correspondences2D2D
from ortholoc.dataset.OrthoLoC import OrthoLoC


class MatcherGT(Matcher):
    """
    Ground truth matcher for OrthoLoC dataset.
    """
    def __init__(self, dataset: OrthoLoC, num_points: int | None = None) -> None:
        super().__init__(name='GT', device='cpu', angles=[0])
        self.dataset = dataset
        self.num_points = num_points

    def __call__(self, img0: np.ndarray, img1: np.ndarray, sample_id: str | None = None, covisible_only: bool = True,
                 normalized: bool = True, *args: Any, **kwargs: Any) -> Correspondences2D2D:
        assert sample_id is not None
        sample = self.dataset[self.dataset.sample_ids.index(sample_id)]
        if self.num_points is not None:
            sampling_pts2d = np.random.uniform(-1, 1, size=(self.num_points, 2))
        else:
            sampling_pts2d = None
        query_to_dop_correspondences_2d2d = OrthoLoC.get_correspondences_2d2d(sample, normalized=normalized,
                                                                              sampling_pts2d=sampling_pts2d,
                                                                              covisible_only=covisible_only)
        return query_to_dop_correspondences_2d2d
