import os, sys

import numpy as np
import matplotlib.pyplot as plt

# NOTE:
from astropy import (
    units,
    constants,
)

# NOTE:
from scipy import (
    ndimage,
    interpolate,
)

# ---------------------------------------------------------------------------- #
path = os.environ["GitHub"] + "/utils"
sys.path.append(path)
import spectral_utils as spectral_utils
# ---------------------------------------------------------------------------- #


#import autolens as al

def extract(cube, x, y, order=1, visualize=False):

    nz = cube.shape[0]
    nx = len(x)
    ny = len(y)

    zi = np.outer(
        np.arange(nz, dtype=int),
        np.ones(nx)
    )
    xi = np.outer(
        np.ones(nz),
        x
    )
    yi = np.outer(
        np.ones(nz),
        y
    )

    # NOTE:
    if visualize:
        for i in range(cube.shape[0]):
            plt.imshow(
                cube[i, :, :],
                cmap="jet",
                aspect="auto",
            )
            plt.plot(
                xi, yi,
                marker="o",
                markersize=10,
                color="black"
            )
            plt.show()
        exit()

    return ndimage.map_coordinates(
        cube,
        [zi, yi, xi],
        #[zi, xi, yi], # NOTE: WRONG???
        order=order,
        cval=np.nan
    )


# def compute_z_step_kms(frequencies, frequency_0=None):
#
#     if frequency_0 is None:
#         frequency_0 = np.mean(
#             a=frequencies,
#             axis=0
#         )
#
#     return spectral_utils.convert_frequency_to_velocity_resolution(
#         frequency_resolution=np.divide(
#             np.subtract(
#                 frequencies[-1], frequencies[0]
#             ),
#             len(frequencies) - 1
#         ),
#         frequency_0=frequency_0,
#     )


# def convert_frequencies_to_velocities(frequencies, model):
#
#     frequencies_interp = interpolate.interp1d(
#         x=np.arange(len(frequencies)),
#         y=frequencies,
#         kind="linear"
#     )
#
#     frequency_0 = frequencies_interp(model.z_centre)
#
#     return spectral_utils.convert_frequencies_to_velocities(
#         frequencies=frequencies,
#         frequency_0=frequency_0,
#     )


def major_axis(phi, centre, n_pixels, pixel_scale, dx=0.1):
    print("Calling \'major_axis\'.")
    a = np.tan((phi - 90.0) * units.deg.to(units.rad))

    x1 = n_pixels / 2.0 - centre[0] / pixel_scale
    y1 = n_pixels / 2.0 + centre[1] / pixel_scale
    print(x1, y1, "|", "")

    x = np.arange(0, n_pixels + 1, dx)
    y = a * x + (y1 - a * x1)

    idx = np.logical_and(
        np.logical_and(y > 0, y < n_pixels),
        np.logical_and(x > 0, x < n_pixels)
    )

    x = x[idx]
    y = y[idx]

    return x, y, x1, y1

# def minor_axis(phi, centre, n_pixels, pixel_scale):
#
#     a = np.tan((phi - 180.0) * units.deg.to(units.rad))
#
#     x1 = centre[1] / pixel_scale + n_pixels / 2.0
#     y1 = -centre[0] / pixel_scale + n_pixels / 2.0
#
#     x = np.arange(0, n_pixels + 1, 1)
#     y = a * x + (y1 - a * x1)
#
#     idx = np.logical_and(
#         np.logical_and(y > 0, y < n_pixels),
#         np.logical_and(x > 0, x < n_pixels)
#     )
#
#     x = x[idx]
#     y = y[idx]
#
#     return x, y, x1, y1


# def major_axis_from_model(model, n_pixels, pixel_scale):
#
#     return major_axis(
#         phi=model.phi,
#         centre=model.centre,
#         n_pixels=n_pixels,
#         pixel_scale=pixel_scale
#     )





# def plot_cube(cube, model, grid_3d, z_mask, frequencies, nrows, ncols, figsize):
#
#     def main(axes, cube, vmin=None, vmax=None, cmap="jet"):
#
#         nrows, ncols = axes.shape
#
#         if vmin is None:
#             vmin = np.nanmin(cube)
#         if vmax is None:
#             vmax = np.nanmax(cube)
#
#         k = 0
#         for i in range(nrows):
#             for j in range(ncols):
#                 im = axes[i, j].imshow(
#                     cube[k, :, :],
#                     cmap=cmap,
#                     vmin=vmin,
#                     vmax=vmax
#                 )
#
#                 axes[i, j].set_xticks([])
#                 axes[i, j].set_yticks([])
#
#                 k += 1
#
#
#     def ticks(axes, xticks=None, yticks=None):
#
#         nrows, ncols = axes.shape
#
#         if xticks is None:
#             xticks = []
#         if yticks is None:
#             yticks = []
#
#         for i in range(nrows):
#             for j in range(ncols):
#
#                 axes[i, j].set_xticks(xticks)
#                 axes[i, j].set_yticks(yticks)
#
#
#     def add_point(axes, x, y):
#
#         nrows, ncols = axes.shape
#
#         for i in range(nrows):
#             for j in range(ncols):
#
#                 axes[i, j].plot([x], [y], marker="o", color="w")
#
#
#     def add_line(axes, x, y):
#
#         nrows, ncols = axes.shape
#
#         for i in range(nrows):
#             for j in range(ncols):
#
#                 axes[i, j].plot(x, y, color="w")
#
#
#     def add_velocities(axes, velocities, x_position, y_position, color="w"):
#
#         nrows, ncols = axes.shape
#
#         k = 0
#         for i in range(nrows):
#             for j in range(ncols):
#
#                 label = "{0:.1f} km / s".format(
#                     velocities[k]
#                 )
#
#                 axes[i, j].text(
#                     x_position,
#                     y_position,
#                     label,
#                     color=color
#                 )
#
#                 k += 1
#
#         return axes
#
#
#     figure, axes = plt.subplots(
#         nrows=nrows,
#         ncols=ncols,
#         figsize=figsize
#     )
#
#     main(
#         axes=axes,
#         cube=cube,
#         cmap="jet"
#     )
#
#     x, y, x_centre, y_centre = major_axis(
#         model=model,
#         n_pixels=grid_3d.n_pixels,
#         pixel_scale=grid_3d.pixel_scale
#     )
#
#     add_point(
#         axes=axes,
#         x=x_centre,
#         y=y_centre
#     )
#
#     add_line(
#         axes=axes,
#         x=x,
#         y=y
#     )
#
#     velocities = convert_frequencies_to_velocities(
#         frequencies=frequencies,
#         model=model
#     )
#
#     add_velocities(
#         axes=axes,
#         velocities=velocities[~z_mask],
#         x_position=10,
#         y_position=10,
#         color="w"
#     )
#
#
#     subplots_kwargs={
#         "wspace":0.01,
#         "hspace":0.01,
#         "left":0.025,
#         "right":0.975,
#         "bottom":0.05,
#         "top":0.995
#     }
#     plt.subplots_adjust(
#         **subplots_kwargs
#     )
#     plt.show()
