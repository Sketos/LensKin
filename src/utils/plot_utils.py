import numpy as np
import matplotlib.pyplot as plt

# Autolens NUFFT dirty images are vertically flipped to match astronomical
# coordinates and should be displayed with origin="lower".
DEFAULT_IMAGE_ORIGIN = "lower"


# NOTE
def plot_cube(
    cube,
    ncols=10,
    extent=None,
    vmin=None,
    vmax=None,
    xlim=None,
    ylim=None,
    figsize=None,
    origin=DEFAULT_IMAGE_ORIGIN,
    cmap="jet",
    aspect="auto",
    interpolation="None",
    colorbar=False,
    colorbar_label=None,
    subplots_kwargs={
        "wspace":0.01,
        "hspace":0.01,
        "left":0.01,
        "right":0.99,
        "bottom":0.01,
        "top":0.99
    },
):

    if cube.shape[0] % ncols == 0:
        nrows = int(cube.shape[0] / ncols)
    else:
        nrows = int(cube.shape[0] / ncols) + 1
    # Leave room on the right when a colour bar is requested.
    if colorbar:
        subplots_kwargs = dict(subplots_kwargs)
        subplots_kwargs["right"] = min(float(subplots_kwargs.get("right", 0.99)), 0.90)
    figure, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(15, 1.25 * nrows)
    )
    axes_flattened = np.ndarray.flatten(axes)

    # NOTE:
    if vmin is None:
        vmin = np.nanmin(cube)
    if vmax is None:
        vmax = np.nanmax(cube)

    last_im = None
    for i, (ax, image) in enumerate(zip(axes_flattened, cube)):
        if i < cube.shape[0]:
            last_im = ax.imshow(
                image,
                cmap=cmap,
                aspect=aspect,
                interpolation=interpolation,
                origin=origin,
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            ax.minorticks_on()
            ax.tick_params(
                axis='y',
                which="major",
                length=5,
                width=1.,
                right=True,
                top=True,
                direction="in",
                colors='black'
            )
            ax.tick_params(
                axis='y',
                which="minor",
                length=2.5,
                width=1,
                right=True,
                top=True,
                direction="in",
                colors='black'
            )
            ax.tick_params(
                axis='x',
                which="major",
                length=5,
                width=1.,
                bottom=True,
                top=True,
                direction="in",
                colors='black',
            )
            ax.tick_params(
                axis='x',
                which="minor",
                length=2.5,
                width=1,
                bottom=True,
                top=True,
                direction="in",
                colors='black',
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if xlim is not None and ylim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
    for j in range(i, len(axes_flattened)):
        axes_flattened[j].axis("off")

    figure.subplots_adjust(
        **subplots_kwargs
    )
    if colorbar and last_im is not None:
        cbar = figure.colorbar(
            last_im,
            ax=list(axes_flattened),
            fraction=0.02,
            pad=0.02,
        )
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    return figure, axes
