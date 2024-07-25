import numpy as np

# NOTE:
from scipy import (
    interpolate,
)

# NOTE:
import autolens as al


# NOTE: PyAutoLens
def traced_grids_of_planes_from(
    tracer,
    grid,
):
    # NOTE: If tracer is fixed this needs only be computed once ...
    if al.__version__ == "0.45.0":
        return tracer.traced_grids_of_planes_from_grid(
            grid=grid,
            plane_index_limit=tracer.upper_plane_index_with_light_profile
        )
    else:
        raise NotImplementedError()


def ray_trace(
    traced_grids_of_planes,
    image: np.ndarray,
    #interpolator=None,
):

    traced_grid_i = traced_grids_of_planes[0] # This is the regular grid.
    traced_grid_j = traced_grids_of_planes[1]

    # NOTE: Visualization
    # plt.figure()
    # plt.plot(
    #     traced_grid_i[:, 0],
    #     traced_grid_i[:, 1],
    #     linestyle="None",
    #     marker="o",
    #     color="b",
    # )
    # plt.plot(
    #     traced_grid_j[:, 0],
    #     traced_grid_j[:, 1],
    #     linestyle="None",
    #     marker="o",
    #     color="r",
    # )
    # plt.show()
    # exit()

    # NOTE:
    x_interp = np.unique(traced_grid_i[:, 0])
    y_interp = np.unique(traced_grid_i[:, 1])
    # # NOTE: MADE A CHANGE HERE!
    # x_interp = np.unique(grid[:, 0])
    # y_interp = np.unique(grid[:, 1])

    # NOTE:
    image_interp = interpolate.RegularGridInterpolator(
        points=(y_interp, x_interp),
        values=image[::-1, :],
        method="linear",
        bounds_error=False,
        fill_value=0.0
    )

    # NOTE:
    lensed_image = image_interp(
        traced_grid_j
    ).reshape(image.shape)

    return lensed_image


def lensed_cube_from_tracer(
    cube,
    tracer,
    grid, # NOTE: grid_3d.grid_2d
    z_mask=None,
    interpolator=None
):

    # NOTE:
    traced_grids_of_planes = traced_grids_of_planes_from(
        tracer=tracer, grid=grid
    )

    # NOTE:
    lensed_cube = np.zeros(shape=cube.shape)
    for i, image in enumerate(cube):
        lensed_cube[i] = ray_trace(traced_grids_of_planes=traced_grids_of_planes, image=image,)

    return lensed_cube
