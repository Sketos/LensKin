import os
import sys

import matplotlib.pyplot as plt

sys.path.append(
    "{}/tutorials/autofit/tutorial_13".format(
        os.environ["GitHub"]
    )
)

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import plot_utils as plot_utils

import autolens_utils.autolens_plot_utils as autolens_plot_utils


def data(
    masked_dataset,
    transformers,
    output_path=None,
    output_filename=None,
    output_format="show"
):

    dirty_cube = autolens_plot_utils.dirty_cube_from_visibilities(
        visibilities=masked_dataset.visibilities,
        transformers=transformers,
        shape=masked_dataset.mask_3d.shape_3d,
    )

    plot_utils.plot_cube(
        cube=dirty_cube, ncols=10, figsize=(15, 5), save=False, show=False
    )

    if "show" in output_format:
        plt.show()
    elif "png" in output_format:
        plt.savefig(
            "{}/{}.png".format(output_path, output_filename)
        )
    plt.clf()
