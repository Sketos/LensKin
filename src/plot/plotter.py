import os
import sys
import matplotlib.pyplot as plt

sys.path.append(
    "{}/utils".format(os.environ["GitHub"])
)
import plot_utils as plot_utils


def cube_plotter(
    cube,
    masked_slices=None,
    output_path=None,
    output_filename=None,
    output_format="show",
):

    plot_utils.plot_cube(
        cube=cube,
        ncols=10,
        figsize=(15, 5),
        masked_slices=masked_slices,
        show=False
    )

    if "show" in output_format:
        plt.show()
    elif "png" in output_format:
        plt.savefig(
            "{}/{}.png".format(output_path, output_filename)
        )
    plt.clf()
