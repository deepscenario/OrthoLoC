from __future__ import annotations
import argparse
import torch

from ortholoc.benchmarking import Benchmark
from ortholoc.image_matching.MatcherIMCUI import MatcherIMCUI, MATCHER_ZOO
from ortholoc.image_matching import MatcherGT
from ortholoc.dataset import OrthoLoC
from ortholoc import utils


def run_benchmark(dataset_dir: str, matcher_name: str, output_dir: str, pnp_mode: str, num_points: int | None,
                  min_conf: float, limit_size: float, device: str, angles: list[int] | None = None,
                  use_intrinsics: bool = True, mode: int = 0, use_adhop: bool = False,
                  use_refined_extrinsics: bool = False, focal_length_init_from_gt: float | None = None,
                  scale_query_image: float = 1., scale_dop_dsm: float = 1., reprojection_error: float = 5,
                  covisibility_ratio: float = 1.0) -> None:
    if device == 'cuda':
        assert torch.cuda.is_available(), "CUDA is not available, please use CPU"
    # load dataset
    dataset = OrthoLoC(dirpath=dataset_dir, limit_size=limit_size, return_tensor=False, mode=mode,
                         use_refined_extrinsics=use_refined_extrinsics, scale_query_image=scale_query_image,
                         scale_dop_dsm=scale_dop_dsm, covisibility_ratio=covisibility_ratio)
    # load matcher
    if matcher_name == 'GT':
        matcher = MatcherGT(dataset=dataset, num_points=num_points)
    else:
        matcher = MatcherIMCUI(name=matcher_name, device=device, angles=angles)

    # run benchmark
    benchmark = Benchmark(dataset=dataset, matcher=matcher, pnp_mode=pnp_mode, use_intrinsics=use_intrinsics,
                          num_points=num_points, output_dir=output_dir, min_conf=min_conf,
                          focal_length_init_from_gt=focal_length_init_from_gt, use_adhop=use_adhop,
                          reprojection_error=reprojection_error)
    benchmark.run()
    benchmark.save_results_as_txt()
    benchmark.save_results_as_json()


def parse_args():
    argparser = argparse.ArgumentParser()
    # inputs
    argparser.add_argument('--dataset_dir', type=str, help='dataset_dir', required=True)
    argparser.add_argument('--matcher', type=str, help='set_name', required=True,
                           choices=['GT'] + list(MATCHER_ZOO.keys()), dest='matcher_name')
    argparser.add_argument('--output_dir', type=str, help='output_dir', required=True)
    argparser.add_argument('--pnp_mode', type=str, help='pnp_mode', default='poselib')
    argparser.add_argument('--mode', type=int, help='0: all sample, 1: same domain, 2: xDOP, 3: xDOPDSM', default=0,
                           choices=[0, 1, 2, 3])
    # options
    argparser.add_argument('--use_refined_extrinsics', action='store_true', help='Use refine extrinsics as GT')
    argparser.add_argument('--no_intrinsics', action='store_false', help='Do not use prior intrinsics',
                           dest='use_intrinsics')
    argparser.add_argument('--focal_length_init_from_gt', type=float,
                           help='Focal length init value relative to the GT value')
    argparser.add_argument('--reprojection_error', type=float, help='Reprojection error', default=5.0)
    argparser.add_argument('--num_points', type=int, help='pnp_max_points', required=False)
    argparser.add_argument('--use_adhop', action='store_true', help='Use Homography Preconditioning')
    argparser.add_argument('--scale_query_image', type=float, help='Scale of query image', default=1.0)
    argparser.add_argument('--scale_dop_dsm', type=float, help='Scale of DOP and DSM', default=1.0)
    argparser.add_argument('--covisibility_ratio', type=float, help='covisibility_ratio', default=1.0)
    argparser.add_argument('--angles', nargs='+', type=int, help='angles', required=False)
    argparser.add_argument('--min_conf', type=float, help='min_conf', default=0.5)
    argparser.add_argument('--limit_size', type=float, help='limit_size', required=False)
    argparser.add_argument('--device', type=str, help='device', default='cuda')
    return utils.misc.update_args_with_asset_paths(argparser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    run_benchmark(**vars(args))
