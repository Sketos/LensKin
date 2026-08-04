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


def extract(
    cube,
    x,
    y,
    order=1,
    visualize=False
):

    nz = cube.shape[0]
    nx = len(x)
    ny = len(y)

    # NOTE:
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

    # # NOTE:
    # if visualize:
    #     for i in range(cube.shape[0]):
    #         plt.imshow(
    #             cube[i, :, :],
    #             cmap="jet",
    #             aspect="auto",
    #         )
    #         plt.plot(
    #             xi, yi,
    #             marker="o",
    #             markersize=10,
    #             color="black"
    #         )
    #         plt.show()
    #     exit()

    return ndimage.map_coordinates(
        cube,
        [zi, yi, xi],
        #[zi, xi, yi], # NOTE: WRONG???
        order=order,
        cval=np.nan
    )


def major_axis(
    phi,
    centre,
    n_pixels,
    pixel_scale,
    dx=0.1,
):

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


def extract_from_major_axis(
    cube,
    phi,
    centre,
    n_pixels,
    pixel_scale,
    dx=0.1,
    order=1,
    visualize=False
):

    x, y, _, _ = major_axis(
        phi=phi,
        centre=centre,
        n_pixels=n_pixels,
        pixel_scale=pixel_scale,
        dx=dx,
    )

    return extract(
        cube=cube,
        x=x,
        y=y,
        order=order,
        visualize=visualize
    )

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
