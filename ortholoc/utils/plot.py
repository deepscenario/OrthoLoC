from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ortholoc import utils

PAPER_PLOTS = True
if PAPER_PLOTS:
    from tueplots import fontsizes, figsizes

    # Combine NeurIPS 2024 font sizes and figure sizes
    style = fontsizes.neurips2024() | figsizes.neurips2024()
    # Apply the combined style
    plt.rcParams.update(style)


def apply_cmap(data: np.ndarray, cmap_name: str = 'plasma', vmin: float | None = None,
               vmax: float | None = None) -> np.ndarray:
    """
    Apply a colormap to data.

    Args:
        data: Input data array.
        cmap_name: Name of the colormap to apply.
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.

    Returns:
        Colored data as a numpy array.
    """
    cmap = plt.get_cmap(cmap_name)
    cmap.set_bad(color='white')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colored = cmap(norm(data))
    return colored


def plot_pts2d(pts2d, img: np.ndarray | None = None, ax: plt.Axes | None = None, title='', c='r', s=1,
               show: bool = False, figsize=(6, 4), label=None, alpha: float = 1.0, marker: str = 'o',
               metrics: dict[str, tuple[float, str]] | None = None, heatmap=None, vmin=None, vmax=None, cmap='plasma',
               colorbar_label='', show_colorbar: bool = False):
    """
    Plot 2D points on an image or a blank canvas.

    Args:
        pts2d: 2D points to plot (N, 2).
        img: Background image (optional).
        ax: Matplotlib Axes object (optional).
        title: Plot title (str).
        c: Color of points (str or array).
        s: Size of points (int).
        show: Whether to display the plot (bool).
        figsize: Figure size (tuple).
        label: Label for the points (str).
        alpha: Transparency of points (float).
        marker: Marker style (str).
        metrics: Dictionary of metrics to display on the plot (optional).
        heatmap: Heatmap values for coloring points (optional).
        vmin: Minimum value for heatmap normalization (optional).
        vmax: Maximum value for heatmap normalization (optional).
        cmap: Colormap for heatmap (str).
        colorbar_label: Label for the colorbar (str).
        show_colorbar: Whether to display the colorbar (bool).

    Returns:
        A tuple containing the Matplotlib Figure and Axes objects.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    assert isinstance(ax, plt.Axes)
    fig = ax.figure
    if img is not None:
        ax.imshow(img)
        ax.axis('off')
    norm = None
    if heatmap is not None:
        if vmax is None:
            vmax = heatmap.max()
        if vmin is None:
            vmin = heatmap.min()
            if vmax == vmin:
                vmin = 0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        plt_cmap = plt.get_cmap(cmap)
        colors = plt_cmap(norm(heatmap))
    else:
        colors = c
    ax.scatter(pts2d[:, 0], pts2d[:, 1], c=colors, s=s, label=label, alpha=alpha, marker=marker)

    if show_colorbar and norm is not None:
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            orientation='vertical',
            shrink=0.8,
        )
        if colorbar_label:
            cbar.set_label(colorbar_label)
        ticks = np.linspace(vmin, vmax, num=5).tolist()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])

    if metrics is not None:
        metrics_str = '\n'.join([f"{key}={value[0]:.2f}{value[1]}" for key, value in metrics.items()])
        ax.text(0.02, 0.98, metrics_str, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                fontsize=10, bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='lightgray',
                    linewidth=1,
                ))

    if title:
        ax.set_title(title.strip())
    if show:
        plt.show()

    return fig, ax


def draw_lines(pts2d_0, pts2d_1, img: np.ndarray | None = None, ax: plt.Axes | None = None, c='r', linewidth: float = 1,
               show: bool = False, figsize=(6, 6), alpha: float = 1.0, label: str | None = None):
    """
    Draw lines connecting pairs of 2D points on an image or a blank canvas.

    Args:
        pts2d_0: First set of 2D points (N, 2).
        pts2d_1: Second set of 2D points (N, 2).
        img: Background image (optional).
        ax: Matplotlib Axes object (optional).
        c: Line color (str).
        linewidth: Line width (float).
        show: Whether to display the plot (bool).
        figsize: Figure size (tuple).
        alpha: Transparency of lines (float).
        label: Label for the lines (str).

    Returns:
        None
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    assert isinstance(ax, plt.Axes)

    if img is not None:
        ax.imshow(img)
        ax.axis('off')

    all_x = np.array([[p0[0], p1[0], None] for p0, p1 in zip(pts2d_0, pts2d_1)]).flatten()
    all_y = np.array([[p0[1], p1[1], None] for p0, p1 in zip(pts2d_0, pts2d_1)]).flatten()

    ax.plot(all_x, all_y, c=c, linewidth=linewidth, alpha=alpha, label=label)

    if show:
        plt.show()


def plot_reprojections(image_query, pts3d, pose_c2w_pred: np.ndarray, pose_c2w_gt: np.ndarray,
                       intrinsics_matrix_gt: np.ndarray, intrinsics_matrix_pred: np.ndarray | None = None,
                       fig_size=(12, 12), ax: plt.Axes | None = None, title: str = '', marker: str = 'o',
                       metrics: dict[str, tuple[float, str]] | None = None, show: bool = False, s: int = 1,
                       linewidth: float = 1, alpha: float = 1.0) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot reprojections of 3D points onto an image, comparing predicted and ground truth poses.

    Args:
        image_query: Query image as a numpy array.
        pts3d: 3D points to project.
        pose_c2w_pred: Predicted camera-to-world pose matrix (4x4).
        pose_c2w_gt: Ground truth camera-to-world pose matrix (4x4).
        intrinsics_matrix_gt: Ground truth camera intrinsics matrix (3x3).
        intrinsics_matrix_pred: Predicted camera intrinsics matrix (3x3). Defaults to ground truth intrinsics.
        fig_size: Size of the figure (tuple).
        ax: Matplotlib Axes object (optional).
        title: Title of the plot (str).
        marker: Marker style for points (str).
        metrics: Dictionary of metrics to display on the plot (optional).
        show: Whether to display the plot (bool).
        s: Size of the points (int).
        linewidth: Line width for connecting lines (float).
        alpha: Transparency of points and lines (float).

    Returns:
        A tuple containing the Matplotlib Figure and Axes objects.
    """
    from ortholoc.correspondences import Correspondences2D2D
    pts2d_pred = utils.geometry.project_pts3d(
        pts3d=pts3d, pose_w2c=utils.pose.inv_pose(pose_c2w_pred),
        intrinsics=intrinsics_matrix_pred if intrinsics_matrix_pred is not None else intrinsics_matrix_gt)
    pts2d_gt = utils.geometry.project_pts3d(pts3d=pts3d, pose_w2c=utils.pose.inv_pose(pose_c2w_gt),
                                            intrinsics=intrinsics_matrix_gt)
    reproj_errors = np.nanmedian(np.linalg.norm(pts2d_pred - pts2d_gt, axis=-1))

    correspondences_2d2d = Correspondences2D2D(pts0=pts2d_pred, pts1=pts2d_gt, is_normalized=False)
    correspondences_2d2d = correspondences_2d2d.take_covisible(w0=image_query.shape[1], h0=image_query.shape[0],
                                                               w1=image_query.shape[1], h1=image_query.shape[0])

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=fig_size)
    assert isinstance(ax, plt.Axes)
    fig = ax.figure
    if title:
        ax.set_title(title.strip())

    ax.imshow(image_query)
    plot_pts2d(pts2d=correspondences_2d2d.pts1, ax=ax, c='g', s=s, show=False, label='GT', alpha=alpha, marker=marker)
    plot_pts2d(pts2d=correspondences_2d2d.pts0, ax=ax, c='r', s=s, show=False, label='Reprojections', alpha=alpha,
               marker=marker)
    draw_lines(pts2d_0=correspondences_2d2d.pts0, pts2d_1=correspondences_2d2d.pts1, ax=ax, c='cyan', show=False,
               alpha=alpha, linewidth=linewidth, label='Discrepancy')
    legend = ax.legend(loc='upper right', fontsize=10)
    legend_font = legend.get_texts()[0].get_fontproperties()
    ax.set_xlim(0, image_query.shape[1])
    ax.set_ylim(image_query.shape[0], 0)
    ax.axis('off')

    if metrics is not None:
        metrics['RPE'] = (reproj_errors, 'px')
        metrics_str = '\n'.join([f"{key}={value[0]:.2f}{value[1]}" for key, value in metrics.items()])
        # add text to the top left corner similar to legend (same font size)
        ax.text(0.02, 0.98, metrics_str, transform=ax.transAxes, fontproperties=legend_font, verticalalignment='top',
                horizontalalignment='left', fontsize=10, bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='lightgray',
                    linewidth=1,
                ))
    if show:
        plt.show()

    return fig, ax

def plot_map(array: np.ndarray, show: bool = True,
             ax: plt.Axes | None = None,
             colormap: str = 'viridis') -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D array as an image.

    Args:
        array: 2D numpy array to be plotted.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig = ax.figure
    ax.imshow(array, cmap=colormap)
    ax.axis('off')
    if show:
        plt.show()
    return fig, ax
