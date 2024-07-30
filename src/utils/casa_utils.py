import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import (
    units,
    constants
)
from astropy.io import fits



# NOTE: THIS ONLY WORKS WHEN THE NUMBER OF CHANNELS IS EVEN NUMBER.
def generate_frequencies(central_frequency, n_channels, bandwidth=2.0 * units.GHz):
    """
    BUG: In the number of channels (n_channels) is an odd number then the number of frequencies is n_channels - 1.
    """


    if n_channels == 1:
        frequencies = central_frequency.to(units.Hz).value
    else:

        df = bandwidth.to(units.Hz).value / n_channels
        #print(df);exit()

        f_min = central_frequency.to(units.Hz).value - int(n_channels / 2.0) * df
        f_max = central_frequency.to(units.Hz).value + int(n_channels / 2.0) * df

        frequencies = np.arange(f_min, f_max, df)

    return frequencies


def convert_array_to_wavelengths(array, frequency):

    array_converted = (
        (array * units.m) * (frequency * units.Hz) / constants.c
    ).decompose()

    return array_converted.value


def convert_uv_coords_from_meters_to_wavelengths(uv, frequencies):
    """
    The frequecies must be in units of Hz
    """
    if np.shape(frequencies):

        u_wavelengths, v_wavelengths = np.zeros(
            shape=(
                2,
                len(frequencies),
                uv.shape[1]
            )
        )

        for i in range(len(frequencies)):
            u_wavelengths[i, :] = convert_array_to_wavelengths(array=uv[0, :], frequency=frequencies[i])
            v_wavelengths[i, :] = convert_array_to_wavelengths(array=uv[1, :], frequency=frequencies[i])

    else:

        u_wavelengths = convert_array_to_wavelengths(array=uv[0, :], frequency=frequencies)
        v_wavelengths = convert_array_to_wavelengths(array=uv[1, :], frequency=frequencies)

    return np.stack(
        arrays=(u_wavelengths, v_wavelengths),
        axis=-1
    )
