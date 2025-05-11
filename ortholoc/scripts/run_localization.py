from __future__ import annotations

import argparse
import os
import numpy as np
from loguru import logger

from ortholoc.image_matching.MatcherIMCUI import MatcherIMCUI, MATCHER_ZOO
from ortholoc.image_matching.MatcherGT import MatcherGT
from ortholoc import utils
from ortholoc.dataset import OrthoLoC


def run_localization(matcher_name: str, img_path: str | None = None, dop_path: str | None = None,
                     dsm_path: str | None = None, intrinsics_path: str | None = None,
                     extrinsics_path: str | None = None, sample_path: str | None = None, use_intrinsics: bool = True,
                     use_refined_extrinsics: bool = False, output_dir: str | None = None,
                     angles: list[float] | None = None, device: str = 'cuda', min_conf: float = 0.5,
                     reprojection_error: float = 5.0, pnp_mode: str = 'poselib', num_points: int | None = None,
                     use_adhop: bool = False, covisibility_ratio: float = 1.0, scale_dop_dsm: float = 1.0,
                     scale_query_image: float = 1.0, show: bool = False, fix_principle_points: bool = True,
                     plot_max_pts: int = 1000, fig_ext: str = '.png') -> None:
    """
    Run localization on two images or a sample from the OrthoLoC dataset.
    """
    dataset = None
    sample = None
    if img_path is not None and dop_path is not None and dsm_path is not None:
        assert os.path.splitext(dop_path)[1] == '.tif', "DOP path must be a .tif file"
        assert os.path.splitext(dsm_path)[1] == '.tif', "DSM path must be a .tif file"
        assert intrinsics_path is None or os.path.splitext(
            intrinsics_path)[1] == '.json', "Intrinsics path must be a .json file"
        assert extrinsics_path is None or os.path.splitext(
            extrinsics_path)[1] == '.json', "Extrinsics path must be a .json file"
        image_query = utils.io.load_image(img_path)
        sample_id = os.path.splitext(os.path.basename(img_path))[0]
        image_dop = utils.io.load_dop_tif(dop_path)
        dsm = utils.io.load_dsm_tif(dsm_path)
        height, width = image_query.shape[:2]
        intrinsics_matrix_gt = utils.io.load_camera_params(intrinsics_path)[1] if intrinsics_path is not None else None
        pose_c2w_gt = utils.pose.inv_pose(
            utils.io.load_camera_params(extrinsics_path)[0]) if extrinsics_path is not None else None
        if intrinsics_matrix_gt is None:
            use_intrinsics = False
        elif not use_intrinsics:
            logger.warning('--no_intrinsics is set but user provided intrinsics. Prior intrinsics will be used.')
            use_intrinsics = True
        if covisibility_ratio != 1.0:
            logger.warning(
                'covisibility_ratio is set to {} but it will be ignored since the input is not a sample'.format(
                    covisibility_ratio))
        if scale_dop_dsm != 1.0:
            logger.warning('scale_dop_dsm is set to {} but it will be ignored since the input is not a sample'.format(
                scale_dop_dsm))
        if scale_query_image != 1.0:
            logger.warning(
                'scale_query_image is set to {} but it will be ignored since the input is not a sample'.format(
                    scale_query_image))
    elif sample_path is not None:
        dataset = OrthoLoC(sample_paths=[sample_path], return_tensor=False,
                           use_refined_extrinsics=use_refined_extrinsics, covisibility_ratio=covisibility_ratio,
                           scale_dop_dsm=scale_dop_dsm, scale_query_image=scale_query_image)
        sample = dataset[0]
        sample_id = sample['sample_id']
        image_query = sample['image_query']
        image_dop = sample['image_dop']
        width, height = sample['w_query'], sample['h_query']
        dsm = sample['dsm']
        if 'mask_dsm' in sample:
            dsm[~sample['mask_dsm'], 2] = np.nan
        pose_c2w_gt = sample['pose_query2world']
        intrinsics_matrix_gt = sample['intrinsics_query']
    else:
        raise ValueError("Either img_paths or sample_path must be provided")

    h0, w0 = image_query.shape[:2]
    h1, w1 = image_dop.shape[:2]

    # perform matching
    if matcher_name == 'GT':
        assert dataset is not None, 'GT matcher only supported for a dataset sample input'
        matcher = MatcherGT(dataset=dataset)
    else:
        matcher = MatcherIMCUI(name=matcher_name, device=device)

    all_correspondences_2d2d = matcher.run(image_query, image_dop, angles=angles, normalized=True)
    best_angle_idx = max(range(len(all_correspondences_2d2d)),
                         key=lambda i: len(all_correspondences_2d2d[i].take_min_conf(min_conf)))
    correspondences_2d2d = all_correspondences_2d2d[best_angle_idx]
    best_angle = matcher.angles[best_angle_idx]
    logger.info(f'Best angle: {best_angle}°, {len(correspondences_2d2d)} matches')

    # lift to 3D
    correspondences_2d2d = correspondences_2d2d.take_min_conf(min_conf=min_conf, keep_at_least=num_points,
                                                              inclusive=False)
    correspondences_2d2d = correspondences_2d2d.take_covisible(h0=h0, w0=w0, h1=h1, w1=w1, is_normalized=True)

    if correspondences_2d2d.is_valid:
        correspondences_2d3d = correspondences_2d2d.to_2d3d(grid3d_1=dsm)

        # calibrate
        success, pose_c2w_pred, intrinsics_matrix_pred, inliers_mask, opt_reproj_errors = \
            correspondences_2d3d.calibrate(
                num_points=num_points, intrinsics_matrix=intrinsics_matrix_gt if use_intrinsics else None,
                width=width, height=height, reprojection_error_diag_ratio=None, reprojection_error=reprojection_error,
                pnp_mode=pnp_mode, fix_principle_points=fix_principle_points
            )

        if success:
            title = f'Localization ({sample_id})\n' if use_intrinsics else f'Calibration ({sample_id})\n'

            transl_error, angular_error = None, None
            if pose_c2w_gt is not None:
                transl_error, angular_error = utils.metrics.pose_error(pose_c2w_pred, pose_c2w_gt)

            if use_adhop:
                success_refined, pose_c2w_pred_refined, intrinsics_matrix_pred_refined, inliers_mask_refined, \
                    opt_reproj_errors_refined, correspondences_2d2d_refined, correspondences_2d3d_refined = \
                    utils.pose.adhop_refinement(
                        matcher=matcher, correspondences_2d2d_init=correspondences_2d2d, image_query=image_query,
                        image_dop=image_dop, dsm=dsm, min_conf=min_conf,
                        intrinsics_matrix_gt=intrinsics_matrix_gt, width=width, height=height,
                        use_intrinsics=use_intrinsics, num_points=num_points, silent=False,
                        reprojection_error=reprojection_error, pnp_mode=pnp_mode,
                        fix_principle_points=fix_principle_points, reproj_errors_init=opt_reproj_errors)

                opt_reproj_error_refined = np.nanmedian(
                    opt_reproj_errors_refined) if opt_reproj_errors_refined is not None else np.nan
                opt_reproj_error = np.nanmedian(opt_reproj_errors)

                transl_error_refined, angular_error_refined = None, None
                if pose_c2w_gt is not None and pose_c2w_pred_refined is not None:
                    transl_error_refined, angular_error_refined = utils.metrics.pose_error(
                        pose_c2w_pred_refined, pose_c2w_gt)

                if success_refined:
                    title += (f'AdHoP chould have been improved results, since '
                              f'{opt_reproj_error:.2f}px > {opt_reproj_error_refined:.2f}px ')
                    if transl_error is not None and transl_error_refined is not None:
                        title += f'TE: from {transl_error:.2f}m to {transl_error_refined:.2f}m, '
                    if angular_error is not None and angular_error_refined is not None:
                        title += f'RE: from {angular_error:.2f}° to {angular_error_refined:.2f}°'
                    title += '\n'

                    # plot initial results
                    if output_dir is not None or show:

                        # plot matches
                        fig_matches, _ = correspondences_2d2d.plot(image_query, image_dop, max_pts=plot_max_pts,
                                                                   title=title, fig_scale=1.0, show=show)
                        if output_dir:
                            utils.io.save_fig(
                                fig_matches,
                                os.path.join(output_dir, f'{sample_id}_{matcher_name}_matches_init' + fig_ext))
                        if sample is not None and pose_c2w_gt is not None:
                            matching_errors = dataset.compute_matching_error(sample, correspondences_2d2d)
                            matching_error = float(np.nanmedian(matching_errors))

                            # matching errors plot
                            fig_matching_errors, _ = utils.plot.plot_pts2d(
                                utils.geometry.denorm_pts2d(correspondences_2d2d.pts0, h=h0,
                                                            w=w0), image_query, show_colorbar=True, alpha=0.5, s=1,
                                heatmap=dataset.compute_matching_error(sample, correspondences_2d2d),
                                metrics={'ME': (matching_error, 'px')}, show=True, colorbar_label='ME (px)')
                            utils.io.save_fig(
                                fig_matching_errors,
                                os.path.join(output_dir, f'{sample_id}_{matcher_name}_matching_errors_init' + fig_ext))

                            # reprojection plot
                            fig_reproj, _ = utils.plot.plot_reprojections(
                                image_query, pts3d=sample['keypoints'], pose_c2w_pred=pose_c2w_pred,
                                pose_c2w_gt=pose_c2w_gt, intrinsics_matrix_pred=intrinsics_matrix_pred, title=title,
                                metrics={
                                    'ME': (matching_error, 'px'),
                                    'TE': (transl_error, 'm'),
                                    'RE': (angular_error, '°')
                                }, intrinsics_matrix_gt=intrinsics_matrix_gt, show=show,
                                fig_size=(5, 3.8 if not title else 5), alpha=0.8, marker='X')
                            if output_dir:
                                utils.io.save_fig(
                                    fig_reproj,
                                    os.path.join(output_dir, f'{sample_id}_{matcher_name}_reprojection_init' + fig_ext))

                    pose_c2w_pred = pose_c2w_pred_refined
                    intrinsics_matrix_pred = intrinsics_matrix_pred_refined
                    correspondences_2d2d = correspondences_2d2d_refined
                    opt_reproj_errors = opt_reproj_errors_refined
                    inliers_mask = inliers_mask_refined
                    transl_error = transl_error_refined
                    angular_error = angular_error_refined
                elif transl_error_refined is not None:
                    logger.info(f'AdHoP most likely did not improve the results. Results will be ignored')
                    title += (f'AdHoP most likely did not improve the results, since '
                              f'{opt_reproj_error:.2f}px < {opt_reproj_error_refined:.2f}px ')
                    if transl_error is not None and transl_error_refined is not None:
                        title += f'TE: from {transl_error:.2f}m to {transl_error_refined:.2f}m, '
                    if angular_error is not None and angular_error_refined is not None:
                        title += f'RE: from {angular_error:.2f}° to {angular_error_refined:.2f}°'
                    title += '\n'
                else:
                    logger.info(f'AdHoP most likely did not improve the results. Results will be ignored')
                    title += (f'AdHoP most likely did not improve the results, since '
                              f'{opt_reproj_error:.2f}px < {opt_reproj_error_refined:.2f}px\n')

            if len(inliers_mask) == len(correspondences_2d2d):
                correspondences_2d2d = correspondences_2d2d.take(inliers_mask)

            if dataset is not None and sample is not None:
                matching_errors = dataset.compute_matching_error(sample, correspondences_2d2d)
                title += f'ME: median = {np.median(matching_errors):.2f}px, mean = {np.mean(matching_errors):.2f}px\n'
            if intrinsics_matrix_gt is not None and not use_intrinsics:
                f_error, c_error = utils.metrics.intrinsics_error(intrinsics_matrix_pred, intrinsics_matrix_gt)
                title += f'RFE: {f_error * 100:.2f}%\n'

            if transl_error is not None and angular_error is not None:
                title += f'TE: {transl_error:.2f}m, RE: {angular_error:.2f}°\n'

            title += f'PnP-RPE: median: {np.nanmedian(opt_reproj_errors):.2f}px, mean: {np.nanmean(opt_reproj_errors):.2f}px\n'

            if sample is not None and pose_c2w_gt is not None:
                kpts_reproj_errors = utils.metrics.reprojection_error(pts3d=sample['keypoints'],
                                                                      pose_c2w_pred=pose_c2w_pred,
                                                                      pose_c2w_gt=pose_c2w_gt,
                                                                      intrinsics_matrix_pred=intrinsics_matrix_pred,
                                                                      intrinsics_matrix_gt=intrinsics_matrix_gt)
                title += f'Kpts-RPE: median: {np.nanmedian(kpts_reproj_errors):.2f}px, mean: {np.nanmean(kpts_reproj_errors):.2f}px\n'

            # plot results
            if output_dir is not None or show:
                fig_matches, _ = correspondences_2d2d.plot(image_query, image_dop, max_pts=plot_max_pts, title=title,
                                                           fig_scale=1.0, show=show)
                if output_dir:
                    utils.io.save_fig(fig_matches,
                                      os.path.join(output_dir, f'{sample_id}_{matcher_name}_matches' + fig_ext))
                if sample is not None and pose_c2w_gt is not None:
                    matching_errors = dataset.compute_matching_error(sample, correspondences_2d2d)
                    matching_error = float(np.nanmedian(matching_errors))

                    # matching errors plot
                    fig_matching_errors, _ = utils.plot.plot_pts2d(
                        utils.geometry.denorm_pts2d(correspondences_2d2d.pts0, h=h0, w=w0), image_query,
                        show_colorbar=True, alpha=0.5, s=1, metrics={'ME': (matching_error, 'px')},
                        title=title, heatmap=matching_errors, show=True, colorbar_label='ME (px)')
                    utils.io.save_fig(fig_matching_errors,
                                      os.path.join(output_dir, f'{sample_id}_{matcher_name}_matching_errors' + fig_ext))

                    # reprojection plot
                    fig_reproj, _ = utils.plot.plot_reprojections(
                        image_query, pts3d=sample['keypoints'], pose_c2w_pred=pose_c2w_pred, pose_c2w_gt=pose_c2w_gt,
                        intrinsics_matrix_pred=intrinsics_matrix_pred, title=title, metrics={
                            'ME': (matching_error, 'px'),
                            'TE': (transl_error, 'm'),
                            'RE': (angular_error, '°')
                        }, intrinsics_matrix_gt=intrinsics_matrix_gt, show=show, s=2, linewidth=0.5,
                        fig_size=(5, 3.8 if not title else 5), alpha=0.8, marker='X')
                    if output_dir:
                        utils.io.save_fig(
                            fig_reproj, os.path.join(output_dir, f'{sample_id}_{matcher_name}_reprojection' + fig_ext))

                if output_dir is not None:
                    utils.io.save_camera_params(pose_w2c=utils.pose.inv_pose(pose_c2w_pred),
                                                intrinsics=intrinsics_matrix_pred,
                                                path=os.path.join(output_dir, 'camera_params.json'))

                logger.info(f'Localization successful')
                logger.info(f'\nPose (world to cam):\n{utils.pose.inv_pose(pose_c2w_pred)}')
                logger.info(f'\nIntrinsics:\n{intrinsics_matrix_pred}')

        else:
            logger.warning("Localization failed")
    else:
        logger.warning("No valid correspondences found")


def parse_args():
    argparser = argparse.ArgumentParser(description='Run Localization / Calibration')
    argparser.add_argument('--image', type=str, help='Image path', dest='img_path')
    argparser.add_argument('--dop', type=str, help='DOP path as .tif file', dest='dop_path')
    argparser.add_argument('--dsm', type=str, help='DSM path as .tif file', dest='dsm_path')
    argparser.add_argument('--sample', type=str, help='Sample path as .npz file', dest='sample_path')

    argparser.add_argument('--intrinsics', type=str, help='Intrinsics path as .json file', dest='intrinsics_path')
    argparser.add_argument('--extrinsics', type=str, help='GT Extrinsics path as .json file '
                           '(for evaluation) as pose world to cam', dest='extrinsics_path')
    argparser.add_argument('--matcher', type=str, help='Matcher name', required=True,
                           choices=['GT'] + list(MATCHER_ZOO.keys()), dest='matcher_name')
    argparser.add_argument('--pnp_mode', type=str, default='poselib', choices=['cv2', 'poselib', 'pycolmap'],
                           help='Library to use for PnP RANSAC')
    argparser.add_argument('--no_intrinsics', action='store_false',
                           help='Do not use prior intrinsics (this will run a full calibration)', dest='use_intrinsics')
    argparser.add_argument('--no_fix_principle_points', action='store_false', help='Do not fix principle points',
                           dest='fix_principle_points')
    argparser.add_argument('--use_refined_extrinsics', action='store_true', help='Use refine extrinsics as GT')
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
    argparser.add_argument('--reprojection_error', type=float, help='RANSAC reprojection error', default=5.0)
    argparser.add_argument(
        '--covisibility_ratio', type=float, help='Covisibility ratio between query '
        'image and the geodata. Default: 1 '
        '(maximum possible ratio)', default=1.0)
    argparser.add_argument('--scale_query_image', type=float, help='Scale of query image, between 0 '
                           '(exclusive) and 1 (full resolution)', default=1.0)
    argparser.add_argument('--scale_dop_dsm', type=float, help='Scale of DOP and DSM, between 0 '
                           '(exclusive) and 1 (full resolution)', default=1.0)
    argparser.add_argument(
        '--num_points', type=int, help='Maximum number of point to consider in PnP. '
        'If not defined, all points will be used')
    argparser.add_argument('--device', type=str, help='Device used to run the matchers', default='cuda',
                           choices=['cuda', 'cpu'])
    argparser.add_argument('--show', action='store_true', help='Whether to show the results')
    argparser.add_argument('--use_adhop', action='store_true', help='Use Homography Preconditioning')
    argparser.add_argument(
        '--min_conf', type=float, help='Minimum correspondences confidences will be '
        'used to filter matchings below this value. Default= 0.5', default=0.5)
    argparser.add_argument('--fig_ext', type=str, help='Figure extension', default='.png')
    argparser.add_argument('--plot_max_pts', type=int, help='Max number of matchings to visualize', default=1000)
    argparser.add_argument('--output_dir', type=str, help='Output directory')

    args = argparser.parse_args()
    if not args.sample_path:
        image_options = [args.img_path, args.dop_path, args.dsm_path]
        if sum(bool(opt) for opt in image_options) != 3:
            argparser.error("You must specify exactly --image, --dop, and --dsm when --sample is not provided.")

    return args


def main():
    args = parse_args()
    run_localization(**vars(args))

if __name__ == '__main__':
    main()