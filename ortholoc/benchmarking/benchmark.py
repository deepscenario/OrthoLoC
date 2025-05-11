from __future__ import annotations

import os
import numpy as np
from tqdm import tqdm
import collections
import time
from loguru import logger
from scipy.spatial.transform import Rotation
from collections import defaultdict
import torch

from ortholoc import utils
from ortholoc.dataset import OrthoLoC
from ortholoc.image_matching import Matcher


class Benchmark:
    """
    Benchmark class for evaluating the performance of a localizer using a matcher on a dataset.
    """
    def __init__(self, dataset: OrthoLoC, matcher: Matcher | None, pnp_mode: str, num_points: int | None,
                 output_dir: str, min_conf: float = 0.0, focal_length_init_from_gt: float | None = None,
                 fix_principle_points: bool = True, reprojection_error: float = 5.0, use_intrinsics: bool = True,
                 use_adhop: bool = False) -> None:
        """
        Args:
            dataset (OrthoLoC): Dataset to benchmark.
            matcher (Matcher): Matcher to use.
            pnp_mode (str): PnP mode to use.
            num_points (int | None): Number of points to use for localization.
            output_dir (str): Directory to save the results.
            min_conf (float, optional): Minimum confidence for correspondences. Defaults to 0.0.
            focal_length_init_from_gt (float | None, optional): Focal length initialization from ground truth. Defaults to None.
            fix_principle_points (bool, optional): Whether to fix the principle points. Defaults to True.
            reprojection_error (float, optional): Reprojection error threshold. Defaults to 5.0.
            use_intrinsics (bool, optional): Whether to use intrinsics. Defaults to True.
            use_adhop (bool, optional): Whether to use adhoc refinement. Defaults to False.
        """

        self.dataset = dataset
        self.matcher = matcher
        self.pnp_mode = pnp_mode
        self.num_points = num_points
        self.output_dir = output_dir
        self.min_conf = min_conf
        self.use_intrinsics = use_intrinsics
        self.focal_length_init_from_gt = focal_length_init_from_gt
        self.use_adhop = use_adhop
        if self.matcher.name == 'GT' and self.use_adhop:
            logger.warning('Using GT matcher with adhoc refinement is not possible. Setting use_adhop to False')
            self.use_adhop = False
        self.fix_principle_points = fix_principle_points
        self.reprojection_error = reprojection_error

        self.pose_errors = defaultdict(list)
        self.angular_errors = defaultdict(list)
        self.matching_errors = defaultdict(list)
        self.reprojection_errors = defaultdict(list)
        self.f_errors = defaultdict(list)
        self.c_errors = defaultdict(list)
        self.poses_pred = defaultdict(list)
        self.intrinsics_pred = defaultdict(list)
        self.matching_durations = defaultdict(list)
        self.durations = defaultdict(list)
        self.sample_ids = defaultdict(list)
        self.best_angles = defaultdict(float)

        self.thresholds = [(0.1, 1), (0.25, 2), (0.5, 5), (2, 2), (3, 3), (5, 5), (5, 10)]
        self.thresholds = sorted(self.thresholds)
        self.experiment_name = f'{self.dataset.name}_{self.matcher.name}_{self.pnp_mode}'

        self.model_size = 0
        self.model_n_params = 0
        for attr in dir(self.matcher):
            obj = getattr(self.matcher, attr)
            if isinstance(obj, torch.nn.Module):
                self.model_size += utils.misc.get_model_size_in_gb(obj)
                self.model_n_params += utils.misc.get_model_n_params(obj)

    def run(self) -> None:
        """
        Run the benchmark on the dataset using the matcher.
        """
        pbar = tqdm(list(range(len(self.dataset))), desc='Predicting')
        for idx in pbar:
            try:
                sample = self.dataset[idx]
            except Exception as e:
                logger.error(f'Error loading sample {idx} {self.dataset.sample_ids[idx]}: {e}')
                raise e

            if sample is None:
                continue

            #########################
            # Preprocessing
            #########################
            sample_id = sample["sample_id"]
            if 'dsm' in sample and 'mask_dsm' in sample:
                sample['dsm'][~sample['mask_dsm'], 2] = np.nan
            if 'point_map' in sample and 'mask_point_map' in sample:
                sample['point_map'][~sample['mask_point_map']] = np.nan

            #########################
            # matching
            #########################
            try:
                tic = time.time()

                all_correspondences_2d2d = self.matcher.run(sample_id=sample['sample_id'], img0=sample['image_query'],
                                                            img1=sample['image_dop'], silent=True, covisible_only=True,
                                                            normalized=True)
                matching_duration = time.time() - tic
            except Exception as e:
                for angle in self.matcher.angles:
                    self.sample_ids[angle].append(sample_id)
                    self.pose_errors[angle].append(float('nan'))
                    self.angular_errors[angle].append(float('nan'))
                    self.matching_errors[angle].append(float('nan'))
                    self.reprojection_errors[angle].append(float('nan'))
                    self.f_errors[angle].append(float('nan'))
                    self.c_errors[angle].append(float('nan'))
                    self.poses_pred[angle].append(float('nan'))
                    self.intrinsics_pred[angle].append(float('nan'))
                    self.matching_durations[angle].append(float('nan'))
                    self.durations[angle].append(float('nan'))
                logger.error(f'Error in matching sample {sample_id}: {e}')
                continue

            # get best correspondences
            best_angle_idx = max(range(len(all_correspondences_2d2d)),
                                 key=lambda i: len(all_correspondences_2d2d[i].take_min_conf(self.min_conf)))
            best_angle = self.matcher.angles[best_angle_idx]

            for angle, correspondences_2d2d_normalized in zip(self.matcher.angles, all_correspondences_2d2d):
                # localize
                tic = time.time()
                intrinsics_matrix_pred = None
                pose_c2w_pred = None
                abs_transl_error = float('nan')
                abs_angular_error = float('nan')
                abs_transl_error_init = float('nan')
                abs_angular_error_init = float('nan')
                matching_error = float('nan')
                f_error = float('nan')
                c_error = float('nan')
                reproj_error = float('nan')

                correspondences_2d2d_normalized = correspondences_2d2d_normalized.take_min_conf(
                    min_conf=self.min_conf, inclusive=False)

                if correspondences_2d2d_normalized.is_valid:
                    correspondences_2d3d_normalized = correspondences_2d2d_normalized.to_2d3d(grid3d_1=sample['dsm'])

                    if correspondences_2d3d_normalized.is_valid:

                        if self.focal_length_init_from_gt is not None and not self.use_intrinsics:
                            fx, fy = sample['intrinsics_query'][0, 0], sample['intrinsics_query'][1, 1]
                            focal_length_init = np.array([fx, fy]) * (1 + self.focal_length_init_from_gt)
                        else:
                            focal_length_init = None

                        success, pose_c2w_pred, intrinsics_matrix_pred, inliers_mask, opt_reproj_errors = \
                            correspondences_2d3d_normalized.calibrate(
                                num_points=self.num_points,
                                intrinsics_matrix=sample['intrinsics_query'] if self.use_intrinsics else None,
                                focal_length_init=focal_length_init,
                                width=sample['w_query'], height=sample['h_query'], pnp_mode=self.pnp_mode,
                                fix_principle_points=self.fix_principle_points,
                            )

                        if success:
                            abs_transl_error_init, abs_angular_error_init = utils.metrics.pose_error(
                                pose_c2w_pred, sample['pose_query2world'])
                            if angle == best_angle and self.use_adhop:
                                success_refined, pose_c2w_pred_refined, intrinsics_matrix_pred_refined, inliers_mask_refined, \
                                    opt_reproj_errors_refined, correspondences_2d2d_refined, correspondences_2d3d_refined = \
                                    utils.pose.adhop_refinement(
                                        matcher=self.matcher, correspondences_2d2d_init=correspondences_2d2d_normalized,
                                        image_query=sample['image_query'],
                                        image_dop=sample['image_dop'], dsm=sample['dsm'], min_conf=self.min_conf,
                                        width=sample['w_query'], height=sample['h_query'],
                                        intrinsics_matrix_gt=sample['intrinsics_query'], silent=True,
                                        use_intrinsics=self.use_intrinsics, num_points=self.num_points,
                                        reprojection_error=self.reprojection_error, pnp_mode=self.pnp_mode,
                                        fix_principle_points=self.fix_principle_points,
                                        reproj_errors_init=opt_reproj_errors
                                    )
                                if success_refined:
                                    pose_c2w_pred = pose_c2w_pred_refined
                                    intrinsics_matrix_pred = intrinsics_matrix_pred_refined
                                    correspondences_2d2d_normalized = correspondences_2d2d_refined

                            abs_transl_error, abs_angular_error = utils.metrics.pose_error(
                                pose_c2w_pred, sample['pose_query2world'])

                            intrinsics_matrix_gt = sample['intrinsics_query']
                            reproj_errors = utils.metrics.reprojection_error(
                                pts3d=sample['keypoints'], pose_c2w_pred=pose_c2w_pred,
                                pose_c2w_gt=sample['pose_query2world'], intrinsics_matrix_pred=intrinsics_matrix_pred,
                                intrinsics_matrix_gt=intrinsics_matrix_gt)

                            matching_errors = self.dataset.compute_matching_error(sample,
                                                                                  correspondences_2d2d_normalized)
                            f_error, c_error = utils.metrics.intrinsics_error(intrinsics_matrix_pred,
                                                                              intrinsics_matrix_gt)
                            matching_error = np.nanmedian(matching_errors)
                            reproj_error = np.nanmedian(reproj_errors)

                pose_solver_duration = time.time() - tic

                duration = pose_solver_duration + matching_duration / len(self.matcher.angles)

                self.sample_ids[angle].append(sample_id)
                self.pose_errors[angle].append(abs_transl_error)
                self.angular_errors[angle].append(abs_angular_error)
                self.matching_errors[angle].append(matching_error)
                self.reprojection_errors[angle].append(reproj_error)
                self.f_errors[angle].append(f_error)
                self.c_errors[angle].append(c_error)
                self.poses_pred[angle].append(pose_c2w_pred)
                self.intrinsics_pred[angle].append(intrinsics_matrix_pred)
                self.matching_durations[angle].append(matching_duration)
                self.durations[angle].append(duration)

                if angle == best_angle:
                    if self.use_adhop:
                        improved = abs_transl_error_init >= abs_transl_error
                        pbar.set_description(f'Sample {sample_id} - {angle} - Improved: {improved} - '
                                             f'pos_error={abs_transl_error_init:.3f} -> {abs_transl_error:.3f} '
                                             f'ang_error={abs_angular_error_init:.3f} -> {abs_angular_error:.3f} ')
                    else:
                        pbar.set_description(f'Sample {sample_id} - {angle} - '
                                             f'pos_error={abs_transl_error:.3f} '
                                             f'ang_error={abs_angular_error:.3f}')

                    self.best_angles[sample_id] = angle
                    self.sample_ids['best'].append(sample_id)
                    self.pose_errors['best'].append(abs_transl_error)
                    self.angular_errors['best'].append(abs_angular_error)
                    self.matching_errors['best'].append(matching_error)
                    self.reprojection_errors['best'].append(reproj_error)
                    self.f_errors['best'].append(f_error)
                    self.c_errors['best'].append(c_error)
                    self.poses_pred['best'].append(pose_c2w_pred)
                    self.intrinsics_pred['best'].append(intrinsics_matrix_pred)
                    self.matching_durations['best'].append(matching_duration)
                    self.durations['best'].append(duration)

    def save_results_as_json(self) -> None:
        """
        Save the benchmark results as a JSON file.
        """
        data = {
            'sample_ids': dict(self.sample_ids),
            'pose_errors': dict(self.pose_errors),
            'angular_errors': dict(self.angular_errors),
            'poses_pred': dict(self.poses_pred),
            'intrinsics_pred': dict(self.intrinsics_pred),
            'matching_errors': dict(self.matching_errors),
            'reprojection_errors': dict(self.reprojection_errors),
            'f_errors': dict(self.f_errors),
            'c_errors': dict(self.c_errors),
            'matching_durations': dict(self.matching_durations),
            'durations': dict(self.durations),
        }
        utils.io.save_json(os.path.join(self.output_dir, f'{self.experiment_name}.json'), data)

    def save_results_as_txt(self):
        """
        Save the benchmark results as text files.
        """
        out_string = self.predictions(self.sample_ids, self.poses_pred, self.intrinsics_pred, self.best_angles)
        utils.io.save_txt(os.path.join(self.output_dir, f'predictions_{self.experiment_name}.txt'), out_string)

        out_string = self.error_per_sample(sample_ids=self.sample_ids, pose_errors=self.pose_errors,
                                           angular_errors=self.angular_errors, best_angles=self.best_angles,
                                           matching_errors=self.matching_errors,
                                           reprojection_errors=self.reprojection_errors, f_errors=self.f_errors,
                                           c_errors=self.c_errors)
        utils.io.save_txt(os.path.join(self.output_dir, f'errors_{self.experiment_name}.txt'), out_string)

        out_string = self.aggregate_stats(pose_errors=self.pose_errors, angular_errors=self.angular_errors,
                                          durations=self.durations, matching_durations=self.matching_durations,
                                          matching_errors=self.matching_errors,
                                          reprojection_errors=self.reprojection_errors, f_errors=self.f_errors,
                                          c_errors=self.c_errors, thresholds=self.thresholds,
                                          model_size=self.model_size, model_n_params=self.model_n_params)
        logger.info(f'Result: {out_string}')
        utils.io.save_txt(os.path.join(self.output_dir, f'summary_{self.experiment_name}.txt'), out_string)

    @staticmethod
    def predictions(sample_ids: dict[str, list], poses_pred: dict[str, list], intrinsics_pred: dict[str, list],
                    best_angles: dict[str, float]) -> str:
        """
        Convert the predictions to a string format for saving.
        """
        out_str = ''
        for angle in sample_ids:
            if angle == 'best':
                continue
            for sample_id, pose_pred, intrinsic_pred in zip(sample_ids[angle], poses_pred[angle],
                                                            intrinsics_pred[angle]):
                angle_info = '(best)' if best_angles[sample_id] == angle else ''
                if pose_pred is not None and np.isfinite(pose_pred).all():
                    pose_w2c_pred = utils.pose.inv_pose(pose_pred)
                    pose_pred_q = Rotation.from_matrix(pose_w2c_pred[:3, :3]).as_quat()
                    pose_pred_t = pose_w2c_pred[:3, 3]
                    line_q = pose_pred_q.tolist()
                    line_t = pose_pred_t.flatten().tolist()
                    if intrinsic_pred is not None and np.isfinite(intrinsic_pred).all():
                        fx, fy, cx, cy = intrinsic_pred[0, 0], intrinsic_pred[1, 1], \
                            intrinsic_pred[0, 2], intrinsic_pred[1, 2]
                    else:
                        fx, fy, cx, cy = None, None, None, None
                    out_str += f'{sample_id} - {angle} {angle_info}: q: {line_q} t: {line_t} ' \
                               f'fx: {fx} fy: {fy} cx: {cx} cy: {cy}\n'
                else:
                    out_str += f'{sample_id} None\n'
        return out_str

    @staticmethod
    def error_per_sample(sample_ids: dict[str, list], pose_errors: dict[str, list], angular_errors: dict[str, list],
                         best_angles: dict[str, float], matching_errors: dict[str, list],
                         reprojection_errors: dict[str, list], f_errors: dict[str, list], c_errors: dict[str,
                                                                                                         list]) -> str:
        """
        Convert the errors to a string format for saving.
        """
        out_str = ''
        for angle in sample_ids:
            if angle == 'best':
                continue
            for sample_id, pose_error, angular_error, matching_error, reprojection_error, f_error, c_error in zip(
                    sample_ids[angle], pose_errors[angle], angular_errors[angle], matching_errors[angle],
                    reprojection_errors[angle], f_errors[angle], c_errors[angle]):
                angle_info = '(best)' if best_angles[sample_id] == angle else ''
                if pose_error == float('nan'):
                    info = 'fail'
                elif pose_error > 1:
                    info = 'high'
                elif pose_error > 0.5:
                    info = 'mid'
                elif pose_error > 0.1:
                    info = 'low'
                else:
                    info = 'very_low'
                out_str += f'{sample_id} - {angle} {angle_info} - {info} - pos_error={pose_error:.3f} pos_error={angular_error:.3f} ' \
                           f'matching_error={matching_error:.3f} reprojection_error={reprojection_error:.3f} ' \
                           f'f_error={f_error:.3f} c_error={c_error:.3f}\n'
        return out_str

    @staticmethod
    def aggregate_stats(pose_errors: dict[str, list], angular_errors: dict[str, list], durations: dict[str, list],
                        matching_durations: dict[str, list], matching_errors: dict[str, list],
                        reprojection_errors: dict[str, list], f_errors: dict[str, list], c_errors: dict[str, list],
                        thresholds: list[tuple[float,
                                               float]] = [(2, 2), (3, 3),
                                                          (5, 5)], model_size=0.0, model_n_params: int = 0) -> str:
        """
        Aggregate the statistics for the benchmark results.
        """
        # [(0.1, 1), (0.25, 2), (0.5, 5), (5, 10)]
        cpu_name, gpu_names = utils.misc.get_hardware_names()
        cpu_ram_size, tot_vram_size = utils.misc.get_ram_sizes()
        out_str = f'{cpu_name} ({cpu_ram_size:.3f} GB) - {gpu_names} ({tot_vram_size:.3f} GB)\n'
        out_str += f'Model size: {model_size:.3f} GB - {utils.misc.human_readable_params(model_n_params)} parameters\n\n'
        pose_errors = {**{'best': pose_errors['best']}, **{k: v for k, v in pose_errors.items() if k != 'best'}}
        for angle in pose_errors:
            stats = collections.Counter()
            pose_errors_curr = np.array(pose_errors[angle])
            angular_errors_curr = np.array(angular_errors[angle])
            durations_curr = np.array(durations[angle])
            matching_durations_curr = np.array(matching_durations[angle])
            matching_errors_curr = np.array(matching_errors[angle])
            reprojection_errors_curr = np.array(reprojection_errors[angle])
            f_errors_curr = np.array(f_errors[angle])
            c_errors_curr = np.array(c_errors[angle])

            pose_errors_curr = np.where(np.isinf(pose_errors_curr), np.nan, pose_errors_curr)
            angular_errors_curr = np.where(np.isinf(angular_errors_curr), np.nan, angular_errors_curr)
            matching_errors_curr = np.where(np.isinf(matching_errors_curr), np.nan, matching_errors_curr)
            reprojection_errors_curr = np.where(np.isinf(reprojection_errors_curr), np.nan, reprojection_errors_curr)
            f_errors_curr = np.where(np.isinf(f_errors_curr), np.nan, f_errors_curr)
            c_errors_curr = np.where(np.isinf(c_errors_curr), np.nan, c_errors_curr)
            durations_curr = np.where(np.isinf(durations_curr), np.nan, durations_curr)
            matching_durations_curr = np.where(np.isinf(matching_durations_curr), np.nan, matching_durations_curr)

            median_pos_error = float(np.nanmedian(pose_errors_curr))
            median_angular_error = float(np.nanmedian(angular_errors_curr))
            matching_error = float(np.nanmedian(matching_errors_curr))
            reprojection_error = float(np.nanmedian(reprojection_errors_curr))
            f_error = float(np.nanmedian(f_errors_curr))
            c_error = float(np.nanmedian(c_errors_curr))
            median_duration = float(np.nanmedian(durations_curr))
            median_matching_durations = float(np.nanmedian(matching_durations_curr))
            out_str += (f'{len(pose_errors_curr)} images - {angle}\n'
                        f'\tmedian_pos_error={median_pos_error:.3f}\n'
                        f'\tmedian_angular_error={median_angular_error:.3f}\n'
                        f'\tmedian_matching_error={matching_error:.3f}\n'
                        f'\tmedian_reprojection_error={reprojection_error:.3f}\n'
                        f'\tmedian_f_error={f_error:.3f}\n'
                        f'\tmedian_c_error={c_error:.3f}\n'
                        f'\tmedian_matching_durations={median_matching_durations:.3f}\n'
                        f'\tmedian_duration={median_duration:.3f}\n')

            for trl_thr, ang_thr in thresholds:
                for pose_error, angular_error in zip(pose_errors_curr, angular_errors_curr):
                    correct_for_this_threshold = (pose_error < trl_thr) and (angular_error < ang_thr)
                    stats[trl_thr, ang_thr] += correct_for_this_threshold
            stats = {
                f'acc@{float(key[0]):.3f}m,{float(key[1]):.3f}deg': 100 * val / len(pose_errors[angle])
                for key, val in stats.items()
            }
            for metric, perf in stats.items():
                out_str += f'\t{metric:12s}={float(perf):.3f}\n'
            out_str += '\n'
        return out_str
