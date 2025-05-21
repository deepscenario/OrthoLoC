from __future__ import annotations

import numpy as np
import cv2
import rasterio
import json
from matplotlib import pyplot as plt
import os
import re
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import importlib.resources
import importlib.util
from appdirs import user_cache_dir

DATASET_URL = 'https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/'
CACHE_DIR = os.environ.get('ORTHOLOC_CACHE_DIR', user_cache_dir('ortholoc'))


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
    path = resolve_path(path, verbose=False)
    assert os.path.exists(path), f"Image path does not exist: {path}"
    if grayscale:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    return image


def load_npz(path: str) -> dict:
    """
    Load data from a compressed NPZ file.
    """
    path = resolve_path(path, verbose=False)
    assert os.path.exists(path), f"NPZ path does not exist: {path}"
    return np.load(path, allow_pickle=True)


def load_tif(path: str, get_coords: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Load a GeoTIFF file.

    Args:
        path: Path to the GeoTIFF file.
        get_coords: Whether to retrieve coordinates.

    Returns:
        A tuple containing the data, mask, and coordinates (if requested).
    """
    path = resolve_path(path, verbose=False)
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
    path = resolve_path(path, verbose=False)
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
    txt_path = resolve_path(txt_path, verbose=False)
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
    path = resolve_path(path, verbose=False)
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_fig(fig: plt.Figure, path: str, dpi: int = 100, bbox_inches='tight', *args, **kwargs) -> None:
    """
    Save a matplotlib figure to a file.

    Args:
        fig: Matplotlib figure to save.
        path: Path to save the figure.
        dpi: Resolution of the saved figure.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, *args, **kwargs)
    plt.close()


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
    path = resolve_path(path, verbose=False)
    data = load_json(path)
    pose_w2c = np.array(data['pose_w2c']) if 'pose_w2c' in data and data['pose_w2c'] is not None else None
    intrinsics = np.array(data['intrinsics']) if 'intrinsics' in data and data['intrinsics'] is not None else None
    return pose_w2c, intrinsics


def resolve_path(input_path: str | Path, verbose: bool = True) -> str | None:
    local_path = None
    if input_path.startswith("http"):
        rel_dir = input_path.removeprefix(DATASET_URL).removeprefix("/")
        local_path = os.path.join(CACHE_DIR, rel_dir)
        if os.path.splitext(input_path)[1] == "":
            local_path = download_files(input_path, local_path, pattern=r"^.*\.npz$", verbose=verbose)
        else:
            local_path = download_file(input_path, local_path)
    if local_path is None:
        local_path = resolve_asset_path(input_path, verbose=verbose)
        if local_path is None:
            local_path = os.path.join(CACHE_DIR, input_path)
            input_path_tmp = os.path.join(DATASET_URL, input_path)
            if os.path.splitext(input_path)[1] == "":
                download_files(input_path_tmp, local_path, pattern=r"^.*\.npz$")
            else:
                download_file(input_path_tmp, local_path)
        return str(local_path if os.path.exists(local_path) else input_path)
    return str(local_path) if local_path is not None else input_path


def resolve_asset_path(input_path: str | Path, verbose: bool = True) -> Path | None:
    """
    Resolve a path by checking if it exists, if not, try to find it in package assets.

    Args:
        input_path: Original path provided
        asset_folder: Subfolder in assets to look for the file
        verbose: Whether to print info about path resolution

    Returns:
        Resolved Path object or None if not found
    """
    # Convert input to Path object
    input_path = Path(input_path)

    # If path exists, return it
    if input_path.exists():
        if verbose:
            logger.info(f"Using provided path: {input_path}")
        return input_path

    # Try to find in assets
    try:
        lib_path = Path(next(iter(importlib.util.find_spec("ortholoc").submodule_search_locations)))
        asset_path = lib_path / input_path
        if asset_path.exists():
            if verbose:
                logger.info(f"Found in assets: {asset_path}")
            return asset_path
    except Exception as e:
        if verbose:
            logger.info(f"Error accessing assets: {e}")

    return None


def download_file(url: str, save_path: str, verbose=True) -> str | None:
    """
    Downloads a file from a given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path to save the file.
    """
    if os.path.exists(save_path):
        return save_path
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        if verbose:
            logger.info(f"Downloaded: {save_path}")
        return save_path
    else:
        if verbose:
            logger.info(f"Failed to download: {url}")
        return None


def get_file_links(url: str, pattern: str | None = None) -> list[str]:
    """
    Fetches the file links from an Apache directory listing, filtering by a regex pattern.

    Args:
        url (str): The URL of the Apache directory.
        pattern (str): A regex pattern to filter file names.

    Returns:
        list[str]: A list of file URLs matching the regex pattern.
    """
    response = requests.get(url)
    if response.status_code != 200:
        logger.info(f"Failed to access {url}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = [link.get('href') for link in soup.find_all('a') if link.get('href')]
    if pattern is not None:
        regex = re.compile(pattern)

        return [f"{url.rstrip('/')}/{href}" for href in links if regex.match(href)]
    else:
        return links


def download_files(url: str, save_directory: str, pattern: str, verbose: bool = True) -> str | None:
    """
    Downloads files from an Apache directory, filtered by a regex pattern.

    Args:
        url (str): The URL of the Apache directory.
        save_directory (str): The local directory to save the downloaded files.
        pattern (str): A regex pattern to filter file names.
    """
    os.makedirs(save_directory, exist_ok=True)
    file_links = get_file_links(url, pattern)

    if not file_links:
        logger.info("No files found matching the specified pattern.")
        return save_directory
    pbar = tqdm(file_links, desc="Downloading files")
    output_paths = []
    for file_url in pbar:
        file_name = os.path.basename(file_url)
        save_path = os.path.join(save_directory, file_name)
        if os.path.exists(save_path):
            continue
        output_path = download_file(file_url, save_path, verbose=verbose)
        if output_path is None:
            logger.warning(f"Failed to download: {file_url}")
        pbar.set_postfix_str(f"Downloaded: {file_name}")
        output_paths.append(output_path)
    return save_directory if any(output_paths) else None
