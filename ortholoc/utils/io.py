from __future__ import annotations

import numpy as np
import cv2
import os
import rasterio
import json
from matplotlib import pyplot as plt


#################################
# I/O
#################################
def list_dir(dir_path: str) -> tuple[list[str], list[str]]:
    """
    List all files and folders in a directory.

    Args:
        dir_path: Path to the directory.

    Returns:
        A tuple containing a list of folder names and a list of file names.
    """
    folders = []
    file_names = []
    if os.path.isdir(dir_path):
        for entry in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, entry)):
                folders.append(entry)
            else:
                file_names.append(entry)
    return folders, file_names


def find_files(directory: str, suffix: str | tuple[str, ...] | None = None, recursive: bool = False) -> list[str]:
    """
    Find all files in a directory with a given suffix.

    Args:
        directory: Path to the directory.
        suffix: File suffix or tuple of suffixes to filter files.
        recursive: Whether to search recursively.

    Returns:
        A list of file paths matching the suffix.
    """
    folders, fnames = list_dir(directory)
    files = [os.path.join(directory, fname) for fname in fnames if suffix is None or fname.endswith(suffix)]
    if recursive:
        for folder in folders:
            files.extend(find_files(os.path.join(directory, folder), suffix, recursive=recursive))
    return files


def json_serializer(obj: np.ndarray | np.float32 | np.float64, ) -> float | list:
    """
    JSON serializer for numpy data types.

    Args:
        obj: Object to serialize.

    Returns:
        Serialized object.

    Raises:
        TypeError: If the object type is not serializable.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    raise TypeError("Type not serializable")


def save_json(path: str, data: dict) -> None:
    """
    Save data to a JSON file.

    Args:
        path: Path to save the JSON file.
        data: Data to save.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=3, default=json_serializer)


def load_image(path: str, grayscale: bool = False) -> np.ndarray:
    """
    Load an image from a file.

    Args:
        path: Path to the image file.
        grayscale: Whether to load the image in grayscale.

    Returns:
        Loaded image as a numpy array.

    Raises:
        ValueError: If the image could not be loaded.
    """
    if grayscale:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    return image


def load_tif(path: str, get_coords: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load a GeoTIFF file.

    Args:
        path: Path to the GeoTIFF file.
        get_coords: Whether to retrieve coordinates.

    Returns:
        A tuple containing the data, mask, and coordinates (if requested).
    """
    with rasterio.open(path) as src:
        data = src.read()
        data = data.transpose(1, 2, 0)
        mask = np.asarray(data == src.nodata)
        if get_coords:
            rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing='ij')
            xs, ys = rasterio.transform.xy(src.transform, rows.tolist(), cols)
            x_coords = np.array(xs).reshape(src.height, src.width)
            y_coords = np.array(ys).reshape(src.height, src.width)
            coords = np.stack((x_coords, y_coords), axis=-1)
        else:
            coords = None
    return data, mask, coords


def load_dop_tif(path: str) -> np.ndarray:
    """
    Load a DOP GeoTIFF file.

    Args:
        path: Path to the DOP GeoTIFF file.

    Returns:
        Loaded DOP image as a numpy array.

    Raises:
        AssertionError: If the DOP image does not have 3 channels.
    """
    image_dop, _, _ = load_tif(path, get_coords=False)
    assert image_dop.ndim == 3 and image_dop.shape[2] == 3, "DOP image must have 3 channels"
    return image_dop


def load_dsm_tif(path: str) -> np.ndarray:
    """
    Load a DSM GeoTIFF file.

    Args:
        path: Path to the DSM GeoTIFF file.

    Returns:
        Loaded DSM image as a numpy array.

    Raises:
        AssertionError: If the DSM image does not have 3 channels.
    """
    elevations, mask, coords = load_tif(path, get_coords=True)
    elevations[mask] = np.nan
    dsm = np.concatenate((coords, elevations), axis=-1)
    assert dsm.ndim == 3 and dsm.shape[2] == 3, "DSM image must have 3 channel"
    return dsm


def save_npz(npz_path: str, **data: dict) -> None:
    """
    Save data to a compressed NPZ file.

    Args:
        npz_path: Path to save the NPZ file.
        data: Data to save.
    """
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    with open(npz_path, 'wb') as npz_file:
        np.savez_compressed(npz_file, **data)


def save_txt(txt_path: str, data: str) -> None:
    """
    Save data to a text file.

    Args:
        txt_path: Path to save the text file.
        data: Data to save.
    """
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w') as txt_file:
        txt_file.write(data)


def load_txt(txt_path: str) -> str:
    """
    Load data from a text file.

    Args:
        txt_path: Path to the text file.

    Returns:
        Loaded data as a string.
    """
    with open(txt_path, 'r') as txt_file:
        data = txt_file.read()
    return data


def load_json(path: str) -> dict:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Loaded data as a dictionary.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_fig(fig: plt.Figure, path: str, dpi: int = 100) -> None:
    """
    Save a matplotlib figure to a file.

    Args:
        fig: Matplotlib figure to save.
        path: Path to save the figure.
        dpi: Resolution of the saved figure.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def save_camera_params(pose_w2c: np.ndarray | None, intrinsics: np.ndarray | None, path: str) -> None:
    """
    Save camera parameters to a JSON file.

    Args:
        pose_w2c: Camera pose matrix (world-to-camera).
        intrinsics: Camera intrinsics matrix.
        path: Path to save the JSON file.
    """
    data = {}
    if pose_w2c is not None:
        data['pose_w2c'] = pose_w2c
    if intrinsics is not None:
        data['intrinsics'] = intrinsics
    save_json(path, data)


def load_camera_params(path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load camera parameters from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        A tuple containing the camera pose matrix and intrinsics matrix.
    """
    data = load_json(path)
    pose_w2c = np.array(data['pose_w2c']) if 'pose_w2c' in data and data['pose_w2c'] is not None else None
    intrinsics = np.array(data['intrinsics']) if 'intrinsics' in data and data['intrinsics'] is not None else None
    return pose_w2c, intrinsics
