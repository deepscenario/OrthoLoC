from __future__ import annotations

import argparse
import os
import numpy as np

from ortholoc.image_matching.MatcherIMCUI import MatcherIMCUI, MATCHER_ZOO
from ortholoc.image_matching.MatcherGT import MatcherGT
from ortholoc import utils
from ortholoc.dataset import OrthoLoC


def run_matching(matcher_name: str, img0_path: str | None = None, img1_path: str | None = None,
                 sample_path: str | None = None, output_path: str | None = None, angles: list[float] | None = None,
                 device: str = 'cuda', min_conf: float = 0.0, use_adhop: bool = False, show: bool = False,
                 max_pts: int = 1000) -> None:
    """
    Run matching on two images or a sample from the OrthoLoC dataset.
    """
    assert (img0_path is not None
            and img1_path is not None) ^ (sample_path is not None), "Either img_paths or sample_path must be provided"
    dataset = None
    sample = None
    if img0_path is not None and img1_path is not None:
        img0 = utils.io.load_image(img0_path)
        img1 = utils.io.load_image(img1_path)
    elif sample_path is not None:
        dataset = OrthoLoC(sample_paths=[sample_path], return_tensor=False)
        sample = dataset[0]
        img0 = sample['image_query']
        img1 = sample['image_dop']
    else:
        raise ValueError("Either img_paths or sample_path must be provided")

    # perform matching
    if matcher_name == 'GT':
        assert dataset is not None, 'GT matcher only supported for a dataset sample input'
        matcher = MatcherGT(dataset=dataset)
    else:
        matcher = MatcherIMCUI(name=matcher_name, device=device, angles=angles)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    correspondences_2d2d = matcher.run(img0, img1, normalized=False)[0]
    correspondences_2d2d = correspondences_2d2d.take_min_conf(min_conf=min_conf, inclusive=False)
    correspondences_2d2d = correspondences_2d2d.take_covisible(h0=h0, w0=w0, h1=h1, w1=w1)

    if use_adhop:
        correspondences_2d2d = matcher.adhop(correspondences_2d2d=correspondences_2d2d, img0=img0, img1=img1,
                                             min_conf=min_conf, normalized=True, silent=False)
        correspondences_2d2d = correspondences_2d2d.take_min_conf(min_conf=min_conf, inclusive=False)
        correspondences_2d2d = correspondences_2d2d.take_covisible(h0=h0, w0=w0, h1=h1, w1=w1)

    # get GT matches
    title = 'Matches\n'
    if dataset is not None and sample is not None:
        matching_errors = dataset.compute_matching_error(sample, correspondences_2d2d)
        title += f'Matching Error: median = {np.median(matching_errors):.2f} px, mean = {np.mean(matching_errors):.2f} px\n'

    # plot matches
    fig, _ = correspondences_2d2d.plot(img0, img1, max_pts=max_pts, title=title.strip(), fig_scale=1.0, vmin=0, vmax=1,
                                       show=show)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)


def parse_args():
    argparser = argparse.ArgumentParser(description='Run matching')
    # inputs
    group = argparser.add_mutually_exclusive_group(required=True)
    image_group = group.add_argument_group()
    image_group.add_argument('--img0', type=str, help='Path of the first image', dest='img0_path')
    image_group.add_argument('--img2', type=str, help='Path of the second image', dest='img1_path')
    group.add_argument('--sample', type=str, help='Sample path .npz', dest='sample_path')

    argparser.add_argument('--matcher', type=str, help='Matcher name', required=True,
                           choices=['GT'] + list(MATCHER_ZOO.keys()), dest='matcher_name')
    argparser.add_argument(
        '--angles',
        nargs='+',
        type=int,
        help='This rotates the query image before '
        'matching and take the best rotation results as the '
        'final correspondences. If not defined, default '
        'rotation values will be considered depending on '
        'the matcher used.',
    )
    argparser.add_argument('--device', type=str, help='Device used to run the matchers', default='cuda',
                           choices=['cuda', 'cpu'])
    argparser.add_argument('--show', action='store_true', help='Whether to show the results')
    argparser.add_argument('--use_adhop', action='store_true', help='Use Homography Preconditioning')
    argparser.add_argument(
        '--min_conf', type=float, help='Minimum correspondences confidences will be '
        'used to filter matchings below this value. Default= 0.5', default=0.5)
    argparser.add_argument('--plot_max_pts', type=int, help='Max number of matchings to visualize', dest='max_pts',
                           default=1000)
    argparser.add_argument('--output_dir', type=str, help='Output directory')
    return utils.misc.update_args_with_asset_paths(argparser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    run_matching(**vars(args))
