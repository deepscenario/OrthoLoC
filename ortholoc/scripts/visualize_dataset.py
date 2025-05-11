from __future__ import annotations

import argparse
import os
import numpy as np
import random
from matplotlib import pyplot as plt
from ortholoc.dataset import OrthoLoC
from ortholoc import utils


def visualize_sample(
        dataset_dir: str,
        n_scenes: int = 5,
        sample_ids: list | None = None,
        output_path: str | None = None,
        select_good_samples: bool = True,
        with_title: bool = False,
        by_groups: bool = False,
        show: bool = False,
) -> None:
    random.seed(47)
    np.random.seed(47)

    dataset = OrthoLoC(dirpath=dataset_dir, return_tensor=False)
    dataset.shuffle()
    dataset.shuffle()
    groups = dataset.sample_ids_by_type

    n_scenes = min(n_scenes, len(dataset))

    if sample_ids is None and not by_groups:
        sample_ids = random.sample(dataset.sample_ids, n_scenes)

    if sample_ids is None:
        sample_ids = []
        ratios = {'R': 0.5, 'xDOP': 0.25, 'xDOPDSM': 0.25}
        n_scenes_per_type = {k: round(n_scenes * ratios[k]) for k in groups.keys()}
        n_scenes_per_type['R'] += n_scenes - sum(n_scenes_per_type.values())

        for type_name, sample_ids_curr in groups.items():
            sample_ids_by_scene = dataset.group_sample_ids_by_scene(sample_ids_curr)
            scene_ids = random.sample(sorted(sample_ids_by_scene), n_scenes_per_type[type_name])
            for scene_id in scene_ids:
                sample_ids_in_scene = sample_ids_by_scene[scene_id]
                while True:
                    sample_id = random.choice(sample_ids_in_scene)
                    sample_idx = dataset.sample_ids.index(sample_id)
                    sample = dataset[sample_idx]
                    ratio = np.count_nonzero(sample['mask_dsm']) / sample['mask_dsm'].size
                    if not select_good_samples or ratio > 0.85:
                        break
                sample_ids.append(sample_id)

    fig, axs = plt.subplots(len(sample_ids), 4, figsize=(15.5, 3.2 * len(sample_ids)))
    for idx, sample_id in enumerate(sample_ids):
        sample_idx = dataset.sample_ids.index(sample_id)
        sample = dataset[sample_idx]
        dataset.plot_sample(sample['sample_id'], show=False, n_cols=4, n_rows=1, figsize=(15, 3.2), axs=axs[idx],
                            subtitles=idx == 0, show_sample_id=True, fontsize=12)

    if with_title:
        fig.suptitle(f'Examples from the OrthoLoC Dataset ({dataset.name} set)', fontsize=16, y=0.99)

    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    try:
        fig.tight_layout(h_pad=1, w_pad=0, pad=1)
    except Exception as e:
        pass

    if show:
        plt.show()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        utils.io.save_fig(fig, output_path)


def parse_args():
    argparser = argparse.ArgumentParser(description='Visualize samples of the dataset')
    argparser.add_argument('--dataset_dir', type=str, help='Dataset directory containing .npz files', required=True)
    argparser.add_argument('--n_scenes', type=int, help='Number of scenes to visualize', default=5)
    argparser.add_argument('--sample_ids', nargs='+', help='Specific Sample IDs to visualize')
    argparser.add_argument('--show', action='store_true', help='Show the plot')
    argparser.add_argument('--select_good_samples', action='store_true', help='Select samples with good coverage')
    argparser.add_argument('--with_title', action='store_true', help='Add title to the plot')
    argparser.add_argument('--output_path', type=str, required=False,
                           help='Output path to save the plot')
    return utils.misc.update_args_with_asset_paths(argparser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    visualize_sample(**vars(args))
