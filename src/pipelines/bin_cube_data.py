import argparse
from pathlib import Path

import numpy as np

from src.pipelines.cube_io import load_exported_array, write_exported_array


def _trim_axis(array, axis, bin_factor):
    n_use = (array.shape[axis] // bin_factor) * bin_factor
    if n_use == 0:
        raise ValueError(
            f"Array axis {axis} has length {array.shape[axis]}, "
            f"which is smaller than bin_factor={bin_factor}."
        )
    n_drop = array.shape[axis] - n_use
    idx = [slice(None)] * array.ndim
    idx[axis] = slice(0, n_use)
    return array[tuple(idx)], n_use, n_drop


def bin_along_axis_mean(array, axis, bin_factor):
    trimmed, n_use, _ = _trim_axis(array=array, axis=axis, bin_factor=bin_factor)
    new_shape = list(trimmed.shape)
    n_bins = n_use // bin_factor
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_factor)
    reshaped = trimmed.reshape(new_shape)
    return reshaped.mean(axis=axis + 1)


def bin_along_axis_sigma(array, axis, bin_factor):
    trimmed, n_use, _ = _trim_axis(array=array, axis=axis, bin_factor=bin_factor)
    new_shape = list(trimmed.shape)
    n_bins = n_use // bin_factor
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_factor)
    reshaped = trimmed.reshape(new_shape)
    return np.sqrt(np.sum(reshaped**2, axis=axis + 1)) / bin_factor


def bin_frequencies(frequencies, bin_factor):
    return bin_along_axis_mean(np.asarray(frequencies), axis=0, bin_factor=bin_factor)


def bin_uv_wavelengths(uv_wavelengths, bin_factor):
    return bin_along_axis_mean(np.asarray(uv_wavelengths), axis=0, bin_factor=bin_factor)


def bin_visibilities(visibilities, bin_factor):
    return bin_along_axis_mean(np.asarray(visibilities), axis=1, bin_factor=bin_factor)


def bin_sigma(sigma, bin_factor):
    return bin_along_axis_sigma(np.asarray(sigma), axis=1, bin_factor=bin_factor)


def bin_cube_products(
    frequencies,
    uv_wavelengths,
    visibilities,
    sigma,
    bin_factor,
):
    n_chan = len(frequencies)
    n_use = (n_chan // bin_factor) * bin_factor
    n_drop = n_chan - n_use

    binned = {
        "frequencies": bin_frequencies(frequencies, bin_factor),
        "uv_wavelengths": bin_uv_wavelengths(uv_wavelengths, bin_factor),
        "visibilities": bin_visibilities(visibilities, bin_factor),
        "sigma": bin_sigma(sigma, bin_factor),
    }
    summary = {
        "bin_factor": bin_factor,
        "n_channels_in": n_chan,
        "n_channels_used": n_use,
        "n_channels_dropped": n_drop,
        "n_channels_out": len(binned["frequencies"]),
    }
    return binned, summary


def write_binned_fits(products, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, data in products.items():
        path = output_dir / f"{stem}_{name}.fits"
        fits.writeto(path, data=data, overwrite=True)
        paths[name] = path
    return paths


def bin_files_from_settings(
    input_dir,
    output_dir,
    uid,
    width,
    bin_factor=5,
    filename_suffix="_contsub",
    output_suffix="_binned",
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{uid}_width_{width}{filename_suffix}{output_suffix}"
    input_stem = f"{uid}_width_{width}{filename_suffix}"

    frequencies = load_exported_array(
        input_dir / f"frequencies_{input_stem}.fits"
    )
    uv_wavelengths = load_exported_array(
        input_dir / f"uv_wavelengths_{input_stem}.fits"
    )
    visibilities = load_exported_array(
        input_dir / f"visibilities_{input_stem}.fits"
    )
    sigma = load_exported_array(
        input_dir / f"sigma_statwt_{input_stem}.fits"
    )

    products, summary = bin_cube_products(
        frequencies=frequencies,
        uv_wavelengths=uv_wavelengths,
        visibilities=visibilities,
        sigma=sigma,
        bin_factor=bin_factor,
    )

    paths = {
        "frequencies": write_exported_array(
            output_dir / f"frequencies_{stem}", products["frequencies"]
        ),
        "uv_wavelengths": write_exported_array(
            output_dir / f"uv_wavelengths_{stem}", products["uv_wavelengths"]
        ),
        "visibilities": write_exported_array(
            output_dir / f"visibilities_{stem}", products["visibilities"]
        ),
        "sigma": write_exported_array(
            output_dir / f"sigma_statwt_{stem}", products["sigma"]
        ),
    }

    return paths, summary


def main():
    parser = argparse.ArgumentParser(
        description="Spectrally bin exported LensKin FITS cubes for faster test runs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path.home() / "Work/ALMA_SPT/SPT0538_CO9-8"),
        help="Directory containing exported frequencies/uv/vis/sigma FITS files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <input-dir>/binned_x<factor>).",
    )
    parser.add_argument("--uid", default="SPT0538_CO9-8")
    parser.add_argument("--width", default="30kms")
    parser.add_argument("--bin-factor", type=int, default=5)
    parser.add_argument("--filename-suffix", default="_contsub")
    parser.add_argument("--output-suffix", default="_binned")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(args.input_dir) / f"binned_x{args.bin_factor}"

    paths, summary = bin_files_from_settings(
        input_dir=args.input_dir,
        output_dir=output_dir,
        uid=args.uid,
        width=args.width,
        bin_factor=args.bin_factor,
        filename_suffix=args.filename_suffix,
        output_suffix=args.output_suffix,
    )

    print(f"Bin factor: {summary['bin_factor']}")
    print(
        "Channels: "
        f"{summary['n_channels_in']} -> {summary['n_channels_out']} "
        f"(used {summary['n_channels_used']}, dropped {summary['n_channels_dropped']})"
    )
    for name, path in paths.items():
        data = load_exported_array(path)
        print(f"  {name}: {path}  shape={data.shape}")


if __name__ == "__main__":
    main()
