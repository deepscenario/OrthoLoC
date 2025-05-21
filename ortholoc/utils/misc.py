from __future__ import annotations

import torch
import subprocess
import cpuinfo
import GPUtil
import psutil
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
