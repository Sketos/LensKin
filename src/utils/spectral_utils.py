import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import (
    units as au,
    constants
)


# NOTE:
def observed_line_frequency_from_rest_line_frequency(frequency, redshift):
    """

    Parameters
    ----------

    frequency: [GHz]

    redshift:

    """
    return frequency / (redshift + 1.0)


def convert_frequencies_to_velocities(
    frequencies, frequency_0, units=au.km / au.s
):
    """

    NOTE: This has been cross-validated against the spectral_cube package and CASA

    Parameters
    ----------

    frequencies: [GHz]

    frequency_0: [GHz]

    """
    velocities = constants.c * (1.0 - frequencies / frequency_0)

    return (velocities).to(units).value
    

def convert_frequency_to_velocity_resolution(frequency_resolution, frequency_0, units=au.km / au.s):
    """

    Parameters
    ----------

    frequency_resolution: [GHz]

    frequency_0: [GHz]

    units: - default [km/s]

    """
    return (constants.c * frequency_resolution / frequency_0).to(units).value


def compute_z_step_kms(frequencies, frequency_0=None):

    # NOTE: In case the shape of the \'frequencies\' array is (n_c, 1), where
    # n_c is the number of channels.
    frequencies = np.squeeze(frequencies)

    # NOTE:
    if frequency_0 is None:

        frequency_0 = np.mean(
            a=frequencies,
            axis=0
        )

    frequency_resolution = np.divide(
        np.subtract(
            frequencies[-1], frequencies[0]
        ),
        len(frequencies) - 1
    )

    return np.abs(
        convert_frequency_to_velocity_resolution(
            frequency_resolution=frequency_resolution,
            frequency_0=frequency_0,
        )
    )
