import os, sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #

path = os.environ["GitHub"] + "{}/utils"
sys.path.append(path)

import plot_utils as plot_utils

# ---------------------------------------------------------------------------- #

import autolens as al


def plot_dirty_cube(
    uv_wavelengths,
    visibilities,
    grid_3d,
):

    transformers = []
    for i in range(grid_3d.n_channels):
        transformer = al.TransformerFINUFFT(
            uv_wavelengths=uv_wavelengths[i],
            grid=grid_3d.grid_2d.in_radians
        )
        transformers.append(transformer)

    # NOTE:
    dirty_cube = np.zeros(
        shape=grid_3d.shape_3d
    )
    for i in range(visibilities.shape[0]):
        dirty_cube[i, :, :] = transformers[i].image_from_visibilities(
            visibilities=visibilities[i]
        )

    figure, axes = plot_utils.plot_cube(
        cube=dirty_cube,
        ncols=10,
        figsize=(14, 8.5),
        show=False,
        return_axes=True
    )
    plt.show()
