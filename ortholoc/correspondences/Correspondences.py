from __future__ import annotations

from typing_extensions import Self
import numpy as np
from dataclasses import dataclass


@dataclass
class Correspondences:
    """
    Base class for correspondences.
    """

    pts0: np.ndarray
    pts1: np.ndarray
    is_normalized: bool
    confidences: np.ndarray | None = None

    def __post_init__(self) -> None:
        assert len(self.pts0) == len(self.pts1), "pts0 and pts1 must have the same length"  #
        if self.confidences is not None:
            assert len(self.pts0) == len(self.confidences), "pts0 and confidences must have the same length"

    def take(self: Self, idxs: np.ndarray) -> Self:
        """
        Take a subset of correspondences based on the provided indices.
        """
        return self.__class__(
            pts0=self.pts0[idxs],
            pts1=self.pts1[idxs],
            is_normalized=self.is_normalized,
            confidences=self.confidences[idxs] if self.confidences is not None else None,
        )

    def take_min_conf(self, min_conf: float, keep_at_least: int | None = None, inclusive=False) -> Self:
        """
        Filter correspondences based on confidence values.
        """
        if self.confidences is not None:
            if len(self.confidences) > 0:
                argsorted = np.argsort(self.confidences)[::-1]
                if keep_at_least is not None:
                    keep_at_least = min(keep_at_least, len(argsorted))
                    min_conf = min(min_conf, self.confidences[argsorted[keep_at_least - 1]])
                    inclusive = True
                idxs = np.where(self.confidences > min_conf)[0] if not inclusive else \
                    np.where(self.confidences >= min_conf)[0]
                return self.take(idxs)
            else:
                return self
        else:
            raise ValueError("No confidences available to filter correspondences")

    def take_mask(self, mask: np.ndarray) -> Self:
        """
        Filter correspondences based on a boolean mask.
        """
        if self.confidences is not None:
            idxs = np.where(mask)[0]
            return self.take(idxs)
        else:
            raise ValueError("No confidences available to filter correspondences")

    def take_min_max_pts0(self, min_val_0: float, max_val_0: float, min_val_1: float, max_val_1: float,
                          inclusive: bool = True) -> Self:
        """
        Filter correspondences based on min and max values for pts0.
        """
        mask0 = np.bitwise_and((self.pts0[:, 0] > min_val_0) if not inclusive else (self.pts0[:, 0] >= min_val_0),
                               (self.pts0[:, 0] < max_val_0) if not inclusive else (self.pts0[:, 0] <= max_val_0))
        mask1 = np.bitwise_and((self.pts0[:, 1] > min_val_1) if not inclusive else (self.pts0[:, 1] >= min_val_1),
                               (self.pts0[:, 1] < max_val_1) if not inclusive else (self.pts0[:, 1] <= max_val_1))
        return self.take(np.bitwise_and(mask0, mask1))

    def take_min_max_pts1(self, min_val_0: float, max_val_0: float, min_val_1: float, max_val_1: float,
                          inclusive: bool = True) -> Self:
        """
        Filter correspondences based on min and max values for pts1.
        """
        mask0 = np.bitwise_and((self.pts1[:, 0] > min_val_0) if not inclusive else (self.pts1[:, 0] >= min_val_0),
                               (self.pts1[:, 0] < max_val_0) if not inclusive else (self.pts1[:, 0] <= max_val_0))
        mask1 = np.bitwise_and((self.pts1[:, 1] > min_val_1) if not inclusive else (self.pts1[:, 1] >= min_val_1),
                               (self.pts1[:, 1] < max_val_1) if not inclusive else (self.pts1[:, 1] <= max_val_1))
        return self.take(np.bitwise_and(mask0, mask1))

    @property
    def is_finite_mask(self) -> np.ndarray:
        """
        Create a mask for finite values in pts0, pts1, and confidences.
        """
        mask = np.isfinite(self.pts0).all(axis=1) & np.isfinite(self.pts1).all(axis=1)
        if self.confidences is not None:
            mask &= np.isfinite(self.confidences)
        return mask

    def take_finite(self) -> Self:
        return self.take(self.is_finite_mask)

    def __len__(self) -> int:
        return len(self.pts0)

    @property
    def is_valid(self) -> bool:
        finite = self.take_finite().take_min_conf(0.0)
        return len(finite) > 0 and len(finite.pts0) == len(finite.pts1)
