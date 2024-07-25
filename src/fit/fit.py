import numpy as np
import matplotlib.pyplot as plt


class DatasetFit:

    def __init__(
        self,
        masked_dataset,
        model_data,
    ):

        # NOTE:
        self.masked_dataset = masked_dataset

        # NOTE:
        self.model_data = model_data

    @property
    def mask(self):
        return self.masked_dataset.uv_mask

    # @property
    # def mask_real_and_imag_averaged(self):
    #     return self.masked_dataset.uv_mask_real_and_imag_averaged

    @property
    def data(self):
        return self.masked_dataset.data # NOTE: The data are now visibilities ...

    @property
    def noise_map(self):
        return self.masked_dataset.noise_map

    # @property
    # def noise_map_real_and_imag_averaged(self):
    #     return self.masked_dataset.noise_map_real_and_imag_averaged

    @property
    def residual_map(self):
        return residual_map_from_data_model_data_and_mask(
            data=self.data,
            model_data=self.model_data,
            mask=self.mask
        )

    @property
    def normalized_residual_map(self):
        return normalized_residual_map_from_residual_map_noise_map_and_mask(
            residual_map=self.residual_map,
            noise_map=self.noise_map,
            mask=self.mask
        )

    @property
    def chi_squared_map(self):
        return chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=self.residual_map,
            noise_map=self.noise_map,
            mask=self.mask
        )

    @property
    def signal_to_noise_map(self):
        signal_to_noise_map = np.divide(
            self.data,
            self.noise_map
        )
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def chi_squared(self):
        return chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=self.chi_squared_map,
            mask=self.mask
        )

    @property
    def noise_normalization(self):
        return noise_normalization_from_noise_map_and_mask(
            noise_map=self.noise_map,
            mask=self.mask
        )

    @property
    def likelihood(self):
        return likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared,
            noise_normalization=self.noise_normalization
        )


def residual_map_from_data_model_data_and_mask(data, mask, model_data):

    # print(data.shape)
    # print(mask.shape)
    # print(model_data.shape)
    # print(mask)
    # figure, axes = plt.subplots(nrows=1, ncols=4)
    # axes[0].imshow(data)
    # axes[1].imshow(model_data)
    # axes[2].imshow(data-model_data)
    # axes[3].imshow(np.subtract(
    #     data, model_data, out=np.zeros_like(data), where=np.asarray(mask) == 0
    # ))
    # plt.show()
    # exit()

    return np.subtract(
        data, model_data, out=np.zeros_like(data), where=np.asarray(mask) == 0
    )


def normalized_residual_map_from_residual_map_noise_map_and_mask(
    residual_map, noise_map, mask
):

    return np.divide(
        residual_map,
        noise_map,
        out=np.zeros_like(residual_map),
        where=np.asarray(mask) == 0,
    )


def chi_squared_map_from_residual_map_noise_map_and_mask(residual_map, noise_map, mask):

    return np.square(
        np.divide(
            residual_map,
            noise_map,
            out=np.zeros_like(residual_map),
            where=np.asarray(mask) == 0,
        )
    )


def chi_squared_from_chi_squared_map_and_mask(chi_squared_map, mask):

    return np.sum(chi_squared_map[np.asarray(mask) == 0])


def noise_normalization_from_noise_map_and_mask(noise_map, mask):

    return np.sum(np.log(2.0 * np.pi * noise_map[np.asarray(mask) == 0] ** 2.0))


def likelihood_from_chi_squared_and_noise_normalization(
    chi_squared, noise_normalization
):

    return -0.5 * (chi_squared + noise_normalization)
