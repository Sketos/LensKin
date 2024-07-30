import numpy as np
import matplotlib.pyplot as plt

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
    origin=None,
    cmap="jet",
    aspect="auto",
    interpolation="None",
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
    figure, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(15, 1.25 * nrows)
    )
    axes_flattened = np.ndarray.flatten(axes)

    # NOTE:
    if vmin is None:
        vmin = np.nanmin(cube)
    if vmax is None:
        vmax = np.nanmax(cube)

    for i, (ax, image) in enumerate(zip(axes_flattened, cube)):
        if i < cube.shape[0]:
            ax.imshow(
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

    return figure, axes
