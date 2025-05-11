from __future__ import annotations

import torch
import subprocess
import cpuinfo
import GPUtil
import psutil
import argparse
from pathlib import Path
import importlib.resources
import importlib.util
from loguru import logger

def get_git_commit_id() -> str | None:
    """
    Get the current git commit ID.
    """
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                            stderr=subprocess.STDOUT).strip().decode("utf-8")
        return commit_id
    except subprocess.CalledProcessError as e:
        logger.info("Error while getting commit ID:", e.output.decode("utf-8"))
        return None

def get_hardware_names() -> tuple[str, str]:
    """
    Get the names of the CPU and GPU(s).

    Returns:
        A tuple containing the CPU name and a comma-separated string of GPU names.
    """
    cpu_info = cpuinfo.get_cpu_info()
    cpu_name = cpu_info.get('brand_raw', 'Unknown CPU')
    gpus = GPUtil.getGPUs()
    gpu_names = [gpu.name for gpu in gpus]
    return cpu_name, ', '.join(gpu_names)


def get_ram_sizes() -> tuple[float, float]:
    """
    Get the sizes of system RAM and GPU VRAM.

    Returns:
        A tuple containing the total system RAM in GB and the total GPU VRAM in GB.
    """
    virtual_mem = psutil.virtual_memory()
    total_gb = virtual_mem.total / (1024**3)
    gpus = GPUtil.getGPUs()
    gpu_vram_sizes = [gpu.memoryTotal / 1024 for gpu in gpus]
    return total_gb, sum(gpu_vram_sizes)


def get_model_size_in_gb(model: torch.nn.Module) -> float:
    """
    Calculate the size of a PyTorch model in GB.

    Args:
        model: The PyTorch model.

    Returns:
        The size of the model in GB.
    """
    total_size_in_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_in_gb = total_size_in_bytes / (1024**3)
    return total_size_in_gb


def get_model_n_params(model: torch.nn.Module) -> int:
    """
    Get the total number of parameters in a PyTorch model.

    Args:
        model: The PyTorch model.

    Returns:
        The total number of parameters.
    """
    total_size = sum(p.numel() for p in model.parameters())
    return total_size


def human_readable_params(num_params: int) -> str:
    """
    Convert the number of parameters to a human-readable format.

    Args:
        num_params: The number of parameters.

    Returns:
        A string representing the number of parameters in a readable format.
    """
    if num_params < 1_000:
        return f"{num_params}"
    elif num_params < 1_000_000:
        return f"{num_params / 1_000:.1f}K"
    elif num_params < 1_000_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params < 1_000_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    else:
        return f"{num_params / 1_000_000_000_000:.1f}T"

def resolve_asset_path(
        input_path: str | Path,
        verbose: bool = True
) -> Path | None:
    """
    Resolve a path by checking if it exists, if not, try to find it in package assets.

    Args:
        input_path: Original path provided
        asset_folder: Subfolder in assets to look for the file
        verbose: Whether to logger.info info about path resolution

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


def update_args_with_asset_paths(args: argparse.Namespace) -> argparse.Namespace:
    """
    Update all potential asset paths in parser arguments.

    Args:
        args: ArgumentParser namespace containing command line arguments

    Returns:
        Updated ArgumentParser namespace
    """
    # Dictionary to track which arguments were updated
    updates = {}

    # Get all attributes that might be paths
    path_attributes = [attr for attr in dir(args)
                       if not attr.startswith('_') and
                       isinstance(getattr(args, attr), (str, Path)) and
                       any(kw in attr.lower() for kw in ['path', 'file', 'dir', 'config', 'model'])]

    for attr in path_attributes:
        original_path = getattr(args, attr)
        if original_path is not None:
            # Try to resolve the path
            resolved_path = resolve_asset_path(original_path)

            if resolved_path is not None:
                setattr(args, attr, resolved_path)
                updates[attr] = {
                    'original': original_path,
                    'resolved': resolved_path
                }
            else:
                logger.info(f"Warning: Could not resolve path for '{attr}': {original_path}")

    # logger.info summary of updates
    if updates:
        logger.info("\nPath resolutions:")
        for attr, paths in updates.items():
            logger.info(f"- {attr}:")
            logger.info(f"  Original: {paths['original']}")
            logger.info(f"  Resolved: {paths['resolved']}")

    return args