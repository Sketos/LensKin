import os, sys
import numpy as np

# NOTE:
try:
    import autolens as al
except:
    print("\'autolens\' could not be imported")

# NOTE:
from src.grid.grid import (
    Grid3D,
)
from src.mask.mask import (
    Mask3D,
)


# NOTE: ...
def grid_from_mask(
    mask_3d: Mask3D
):

    return Grid3D.uniform(
        n_pixels=mask_3d.n_pixels,
        pixel_scale=mask_3d.pixel_scale,
        n_channels=mask_3d.n_channels,
    )


class Dataset:
    def __init__(
        self,
        uv_wavelengths: np.ndarray,
        visibilities: np.ndarray,
        noise_map: np.ndarray,
        z_step_kms: float=None,
        redshift_lens: float=0.5,
        redshift_source: float=2.0,
    ):

        # NOTE: shape = (n_c, n_v, 2)
        self.uv_wavelengths = uv_wavelengths

        # NOTE:
        if visibilities.shape[0] == self.uv_wavelengths.shape[0]:
            self.visibilities = visibilities
        else:
            raise NotImplementedError()

        # NOTE: shape = (n_c, n_v, 2)
        self.noise_map = noise_map

        # NOTE:
        self.z_step_kms = z_step_kms

        # NOTE:
        self.redshift_lens = redshift_lens
        self.redshift_source = redshift_source

    @property
    def shape(self):
        return visibilities.shape


class MaskedDataset:
    def __init__(
        self,
        dataset: Dataset,
        mask_3d: Mask3D=None,
        uv_mask: np.ndarray=None,
        instance=None,
    ):

        self.dataset = dataset

        # NOTE:
        self.mask_3d = mask_3d

        # NOTE:
        self.grid_3d = grid_from_mask(
            mask_3d=self.mask_3d
        )

        # NOTE:
        self.uv_wavelengths = dataset.uv_wavelengths

        # NOTE:
        self.visibilities = dataset.visibilities
        if uv_mask is None:
            self.uv_mask = np.full(
                shape=self.visibilities.shape,
                fill_value=False
            )
        else:
            self.uv_mask = uv_mask
        self.uv_mask[self.mask_3d.z_mask, ...] = True

        # self.uv_mask_real_and_imag_averaged = np.full(
        #     shape=self.uv_mask.shape[:-1],
        #     fill_value=False
        # )

        # NOTE:
        self.noise_map = dataset.noise_map

        # self.noise_map_real_and_imag_averaged = np.average(
        #     a=self.noise_map, axis=-1
        # )

        # NOTE:
        self.z_step_kms = dataset.z_step_kms

        # NOTE:
        self.redshift_lens = dataset.redshift_lens
        self.redshift_source = dataset.redshift_source

        # NOTE:
        self.instance = instance


    @property
    def data(self):
        return self.visibilities

    @property
    def sigma(self):
        return self.noise_map

    @property
    def z_mask(self):
        return self.mask_3d.z_mask

    @property
    def n_channels(self):
        return self.mask_3d.n_channels

    @property
    def n_pixels(self):
        return self.mask_3d.n_pixels

    @property
    def pixel_scale(self):
        return self.mask_3d.pixel_scale
