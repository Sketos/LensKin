import os
import sys

sys.path.append(
    "{}/tutorials/autofit/tutorial_13".format(
        os.environ["GitHub"]
    )
)
from src.plot import plotter

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import list_utils as list_utils
import autolens_utils.autolens_plot_utils as autolens_plot_utils


# def data(
#     fit,
#     transformers,
#     output_path=None,
#     output_filename=None,
#     output_format="show"
# ):
#
#     cube = autolens_plot_utils.dirty_cube_from_visibilities(
#         visibilities=fit.data,
#         transformers=transformers,
#         shape=fit.masked_dataset.grid_3d.shape_3d
#     )
#
#     plotter.cube_plotter(
#         cube=cube,
#         masked_slices=list_utils.get_bool_indexes(
#             list=fit.masked_dataset.mask_3d.z_mask,
#             value=True
#         ),
#         output_path=output_path,
#         output_filename=output_filename,
#         output_format=output_format,
#     )

def model_data(
    fit,
    transformers,
    output_path=None,
    output_filename=None,
    output_format="show"
):

    cube = autolens_plot_utils.dirty_cube_from_visibilities(
        visibilities=fit.model_data,
        transformers=transformers,
        shape=fit.masked_dataset.grid_3d.shape_3d
    )

    plotter.cube_plotter(
        cube=cube,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )

def residual_map(
    fit,
    transformers,
    output_path=None,
    output_filename=None,
    output_format="show"
):

    cube = autolens_plot_utils.dirty_cube_from_visibilities(
        visibilities=fit.residual_map,
        transformers=transformers,
        shape=fit.masked_dataset.grid_3d.shape_3d
    )

    plotter.cube_plotter(
        cube=cube,
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
    )
