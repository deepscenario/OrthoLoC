from __future__ import annotations

import numpy as np
import cv2


def resize_image(img: np.ndarray, dst_max_size: int) -> np.ndarray:
    """
    Resize an image to fit within a maximum size while maintaining aspect ratio.

    Args:
        img: Input image as a numpy array.
        dst_max_size: Maximum size for the larger dimension of the image.

    Returns:
        Resized image as a numpy array.
    """
    h, w = img.shape[:2]
    ratio = dst_max_size / max(w, h)
    return rescale_image(img, ratio, ratio)


def rescale_image(img: np.ndarray, scale_w: float, scale_h: float) -> np.ndarray:
    """
    Rescale an image by specified width and height scaling factors.

    Args:
        img: Input image as a numpy array.
        scale_w: Scaling factor for the width.
        scale_h: Scaling factor for the height.

    Returns:
        Rescaled image as a numpy array.
    """
    height, width = img.shape[:2]
    width_dst = int(width * scale_w)
    height_dst = int(height * scale_h)
    return cv2.resize(img, (width_dst, height_dst), interpolation=cv2.INTER_LINEAR)


def pad_to_square(data: np.ndarray, pad_value: float | int) -> np.ndarray:
    """
    Pad an image or array to make it square.

    Args:
        data: Input array (H, W, C) or (H, W).
        pad_value: Value to use for padding.

    Returns:
        A square array with padding applied.
    """
    h, w = data.shape[:2]
    max_dim = max(h, w)
    pad_h = max_dim - h
    pad_w = max_dim - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
    if data.ndim == 3:
        pad_width += ((0, 0), )
    padded = np.pad(data, pad_width, mode='constant', constant_values=pad_value)
    return padded


def mask_to_rgba(mask, color):
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)  # Add an alpha channel
    for i, c in enumerate(color):
        rgba[..., i] = mask * c
    rgba[..., 3] = mask
    return rgba
