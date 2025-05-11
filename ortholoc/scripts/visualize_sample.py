from __future__ import annotations

import argparse
import os

from ortholoc.dataset import OrthoLoC
from ortholoc import utils


def visualize_sample(
    sample_path: str,
    output_path: str | None = None,
    no_title: bool = False,
    show: bool = False,
) -> None:
    """
    Visualize a sample from the OrthoLoC dataset.
    """
    dataset = OrthoLoC(sample_paths=[sample_path], return_tensor=False)
    sample = dataset[0]
    fig, _ = dataset.plot_sample(sample['sample_id'], show=show, n_cols=4, n_rows=1,
                                 title=f'Sample {sample["sample_id"]}' if not no_title else '', figsize=(15, 3.2))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    if output_path is not None:
        utils.io.save_fig(fig, output_path)


def parse_args():
    argparser = argparse.ArgumentParser(description='Visualize a sample (.npz)')
    # inputs
    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample', type=str, help='Sample path .npz', dest='sample_path')
    argparser.add_argument('--show', action='store_true', help='Show the plot')
    argparser.add_argument('--no_title', action='store_true', help='Without title')
    argparser.add_argument('--output_path', type=str, required=False, help='Output path to save the plot')
    return argparser.parse_args()


def main():
    args = parse_args()
    visualize_sample(**vars(args))

if __name__ == '__main__':
    main()