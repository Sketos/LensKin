import os, sys
import numpy as np

# NOTE:
try:
    import autolens as al
except:
    print("\'autolens\' could not be imported")

# ---------------------------------------------------------------------------- #
# from src.grid.grid import (
#     Grid3D,
# )
from src.mask.mask import (
    Mask3D,
)
# ---------------------------------------------------------------------------- #


def transformers_from(
    uv_wavelengths,
    mask_3d: Mask3D,
    transformer_class=al.TransformerNUFFT
):

    transformers = []
    for i in range(mask_3d.n_channels):
        transformer = transformer_class(
            uv_wavelengths=uv_wavelengths[i],
            real_space_mask=mask_3d.mask_2d
        )
        transformers.append(transformer)

    return transformers

class Image:
    def __init__(self, array_2d):
        self.array_2d = array_2d

    @property
    def native(self):
        return self.array_2d


def visibilities_from_transformers_and_cube(
    cube: np.ndarray,
    transformers: list,
    shape,
    z_mask=None
):

    if z_mask is None:
        pass

    # NOTE:
    visibilities = np.zeros(shape=shape)
    for i, transformer in enumerate(transformers):
        if transformer is not None:
            visibilities_i = transformers[i].visibilities_from(
                image=Image(array_2d=cube[i])
            )
            visibilities[i, :, 0] = visibilities_i.real
            visibilities[i, :, 1] = visibilities_i.imag
        else:
            raise NotImplementedError()

    return visibilities


def dirty_cube_from(
    visibilities: np.ndarray,
    transformers: list
):

    return np.array([
        transformer.image_from(
            visibilities=visibilities[i, :, 0] + 1j * visibilities[i, :, 1]
        ).native
        for i, transformer in enumerate(transformers)
    ])
