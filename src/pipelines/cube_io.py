from pathlib import Path

import numpy as np


def _read_array(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    from astropy.io import fits

    return fits.getdata(path)


def exported_array_path(path):
    """
    Resolve an exported cube product path, preferring ``.fits`` over ``.npy``.
    """
    path = Path(path)
    if path.is_file():
        return path

    if path.suffix in {".fits", ".npy"}:
        stem = path.with_suffix("")
    else:
        stem = path

    candidates = [stem.with_suffix(".fits"), stem.with_suffix(".npy")]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find exported array for {path} (tried: {tried})"
    )


def exported_array_exists(path):
    try:
        exported_array_path(path)
        return True
    except FileNotFoundError:
        return False


def load_exported_array(path):
    return _read_array(exported_array_path(path))


def output_dir_from_settings(settings):
    return Path(settings["outputvis"]).resolve().parent


def export_stem(settings, prefix):
    uid = settings["uid"]
    width = settings["width"]
    filename_suffix = settings.get("filename_suffix", "_contsub")
    return f"{prefix}_{uid}_width_{width}{filename_suffix}"


def export_path(settings, prefix):
    return output_dir_from_settings(settings) / export_stem(settings, prefix)


def write_exported_array(filename_base, data):
    """
    Write a dataprep product as FITS when astropy is available, otherwise ``.npy``.
    """
    stem = Path(filename_base)
    stem.parent.mkdir(parents=True, exist_ok=True)
    filename_base = str(stem)
    try:
        from astropy.io import fits

        output_path = filename_base + ".fits"
        fits.writeto(output_path, data=data, overwrite=True)
        return output_path
    except ImportError:
        np.save(filename_base, data)
        output_path = filename_base if filename_base.endswith(".npy") else filename_base + ".npy"
        return output_path
