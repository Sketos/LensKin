import os, sys
import numpy as np

# NOTE:
import autolens as al

# ---------------------------------------------------------------------------- #
path = os.environ["GitHub"] + "{}/tutorials/autofit/tutorial_13"
sys.path.append(path)

#from src.grid.grid import Grid3D
# ---------------------------------------------------------------------------- #


def func(mask_2d):

    return mask_2d


class AbstractMask3D:
    def __init__(self):
        pass

class Mask3D(AbstractMask3D):
    def __init__(
        self,
        mask_2d,
        z_mask: np.ndarray
    ):

        # NOTE: PyAutoArray obj
        self.mask_2d = func(mask_2d=mask_2d)

        # NOTE:
        self.z_mask = z_mask
        if self.z_mask is None:
            raise NotImplementedError()
        else:
            self.n_channels = len(self.z_mask)


    @property
    def shape_2d(self):
        if hasattr(self.mask_2d, "shape_2d"):
            return self.mask_2d.shape_2d
        else:
            raise NotImplementedError()


    @property
    def shape_3d(self):
        return (self.n_channels, ) + self.shape_2d


    @property
    def n_pixels(self):
        return self.shape_2d[0]


    @property
    def pixel_scale(self):
        if hasattr(self.mask_2d, "pixel_scale"):
            return self.mask_2d.pixel_scale
        else:
            raise NotImplementedError()


    @property
    def pixel_scales(self):
        if hasattr(self.mask_2d, "pixel_scales"):
            return self.mask_2d.pixel_scales
        else:
            raise NotImplementedError()


    @property
    def sub_size(self):
        if hasattr(self.mask_2d, "sub_size"):
            return self.mask_2d.sub_size
        else:
            raise NotImplementedError()


    @classmethod
    def unmasked(
        cls,
        n_channels,
        shape_2d,
        pixel_scales,
        sub_size=1,
        invert=False
    ) -> "Mask3D":

        z_mask = z_unmasked(
            shape=(n_channels, ),
        )

        # NOTE:
        if al.__version__ == "0.45.0":
            mask_2d = al.Mask.unmasked(
                shape_2d=shape_2d,
                pixel_scales=pixel_scales,
                sub_size=sub_size,
                invert=invert,
            )
        else:
            raise NotImplementedError()

        return Mask3D(
            mask_2d=mask_2d,
            z_mask=z_mask
        )


    @classmethod
    def xy_unmasked_and_z_mask(
        cls,
        z_mask,
        shape_2d,
        pixel_scales,
        sub_size=1,
        invert=False
    ) -> "Mask3D":

        # NOTE:
        if al.__version__ == "0.45.0":
            mask_2d = al.Mask.unmasked(
                shape_2d=shape_2d,
                pixel_scales=pixel_scales,
                sub_size=sub_size,
                invert=invert,
            )
        else:
            raise NotImplementedError()

        return Mask3D(
            mask_2d=mask_2d,
            z_mask=z_mask
        )


def z_unmasked(
    shape,
):
    return np.full(
        shape=shape,
        fill_value=False
    )

def z_mask_from_zmin_and_zmax(
    shape,
    zmin=None,
    zmax=None,
):
    mask = np.full(
        shape=shape,
        fill_value=False
    )
    mask[:zmin] = True
    mask[zmax:] = True

    return mask
