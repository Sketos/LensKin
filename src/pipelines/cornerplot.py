import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import corner
except ImportError as exc:
    raise ImportError(
        "corner is required for corner plots. Install it in your environment, "
        "e.g. `pip install corner`."
    ) from exc


METADATA_COLUMNS = (
    "log_likelihood",
    "log_prior",
    "log_posterior",
    "weight",
)


def resolve_run_directory(path):
    """
    Resolve a PyAutoFit run directory containing ``files/samples.csv``.

    Accepts the run hash folder, its ``files/`` subdirectory, or a direct path
    to ``samples.csv``.
    """
    path = Path(path).resolve()

    if path.is_file() and path.name == "samples.csv":
        return path.parent.parent

    if path.is_dir() and (path / "files" / "samples.csv").is_file():
        return path

    if path.is_dir() and (path / "samples.csv").is_file():
        return path.parent

    raise FileNotFoundError(
        f"Could not find samples.csv under {path}. "
        "Pass the PyAutoFit run directory (the hash folder) or a path to "
        "files/samples.csv."
    )


def _short_label(name):
    for prefix in ("galaxies.source.", "galaxies.lens."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("centre.centre_", "centre_")


PRIOR_TYPES = frozenset(
    {
        "Uniform",
        "LogUniform",
        "Gaussian",
        "LogGaussian",
        "Prior",
    }
)


def free_parameter_paths_from_model_json(model_json_path):
    """
    Return dotted parameter paths for free model parameters from ``model.json``.

    Fixed parameters appear as bare numeric values; free parameters have a
    prior ``type`` (``Uniform``, ``LogUniform``, etc.) or live under a
    ``tuple_prior``.
    """
    with open(model_json_path, encoding="utf-8") as handle:
        model = json.load(handle)

    paths = []

    def walk(node, path_parts):
        if not isinstance(node, dict):
            return

        node_type = node.get("type")

        if node_type == "tuple_prior":
            for name, child in node.get("arguments", {}).items():
                walk(child, path_parts + (name,))
            return

        if node_type in ("model", "collection"):
            for name, child in node.get("arguments", {}).items():
                walk(child, path_parts + (name,))
            return

        if node_type in PRIOR_TYPES or (
            "lower_limit" in node and "type" in node
        ):
            paths.append(".".join(path_parts))
            return

        if "arguments" in node:
            for name, child in node["arguments"].items():
                walk(child, path_parts + (name,))

    walk(model, ())
    return paths


def _filter_free_parameter_columns(parameter_columns, run_directory):
    """
    Keep only columns that correspond to free parameters in ``files/model.json``.
    """
    model_json_path = Path(run_directory) / "files" / "model.json"
    if not model_json_path.is_file():
        raise FileNotFoundError(
            f"Expected {model_json_path} to identify free parameters for the "
            "corner plot."
        )

    free_paths = {
        path.strip()
        for path in free_parameter_paths_from_model_json(model_json_path)
    }
    if not free_paths:
        raise ValueError(f"No free parameters found in {model_json_path}")

    filtered = [
        column
        for column in parameter_columns
        if column.strip() in free_paths
    ]
    if not filtered:
        raise ValueError(
            f"No samples.csv columns matched free parameters in {model_json_path}. "
            f"Expected one of: {sorted(free_paths)}"
        )
    return filtered


def load_samples(samples_csv, run_directory):
    """Load weighted posterior samples for the model's free parameters only."""
    samples_csv = Path(samples_csv)
    dataframe = pd.read_csv(samples_csv, skipinitialspace=True)

    if "weight" not in dataframe.columns:
        raise ValueError(f"No 'weight' column found in {samples_csv}")

    parameter_columns = [
        column
        for column in dataframe.columns
        if column.strip() not in METADATA_COLUMNS
    ]
    if not parameter_columns:
        raise ValueError(f"No parameter columns found in {samples_csv}")

    parameter_columns = _filter_free_parameter_columns(
        parameter_columns=parameter_columns,
        run_directory=run_directory,
    )

    weight_floor = 1e-10
    weights = dataframe["weight"].to_numpy(dtype=float)
    keep = weights >= weight_floor
    filtered = dataframe.loc[keep].copy()

    if len(filtered) == 0:
        filtered = dataframe.copy()
    weights = filtered["weight"].to_numpy(dtype=float)

    varying_columns = []
    for column in parameter_columns:
        values = filtered[column].to_numpy(dtype=float)
        if np.ptp(values) > 0.0:
            varying_columns.append(column)
    if not varying_columns:
        raise ValueError(f"No varying free parameters to plot in {samples_csv}")
    parameter_columns = varying_columns

    samples = filtered[parameter_columns].to_numpy(dtype=float)
    labels = [_short_label(column.strip()) for column in parameter_columns]
    return samples, weights, labels, parameter_columns


def _validate_corner_input(samples, labels, samples_csv):
    n_samples, n_params = samples.shape
    if n_params == 0:
        raise ValueError(f"No varying parameters to plot in {samples_csv}")

    if n_samples < n_params:
        raise ValueError(
            f"Cannot build a corner plot from {samples_csv}: "
            f"only {n_samples} sample(s) for {n_params} parameter(s). "
            "corner requires at least as many samples as parameters. "
            "The fit may still be running, may have failed early, or may "
            "need a longer search (increase n_live / maxiter)."
        )


def truths_from_samples_summary(run_directory, parameter_columns):
    """
    Read best-fit parameter values from ``files/samples_summary.json``.
    """
    summary_path = Path(run_directory) / "files" / "samples_summary.json"
    if not summary_path.is_file():
        return None

    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)

    sample = summary.get("arguments", {}).get("max_log_likelihood_sample")
    if sample is None:
        sample = summary.get("arguments", {}).get("median_pdf_sample")
    if sample is None:
        return None

    kwargs = sample.get("arguments", {}).get("kwargs", {}).get("arguments", {})
    truths = []
    for column in parameter_columns:
        key = column.strip()
        if key not in kwargs:
            return None
        truths.append(float(kwargs[key]))
    return np.asarray(truths, dtype=float)


def truths_from_max_likelihood(samples, weights):
    """Best-fit point as the sample with maximum weight."""
    if len(samples) == 0:
        return None
    return samples[int(np.argmax(weights))]


def make_cornerplot(run_directory):
    """
    Build a weighted corner plot of free parameters from a PyAutoFit run.

    Writes ``cornerplot.png`` in the run directory.
    """
    run_directory = resolve_run_directory(run_directory)
    samples_csv = run_directory / "files" / "samples.csv"
    samples, weights, labels, parameter_columns = load_samples(
        samples_csv=samples_csv,
        run_directory=run_directory,
    )
    _validate_corner_input(samples, labels, samples_csv)

    truths = truths_from_samples_summary(
        run_directory=run_directory,
        parameter_columns=parameter_columns,
    )
    if truths is None:
        truths = truths_from_max_likelihood(samples, weights)

    figure = corner.corner(
        samples,
        weights=weights,
        labels=labels,
        truths=truths,
        bins=50,
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
    )

    output_path = run_directory / "cornerplot.pdf"
    figure.savefig(output_path, dpi=150, bbox_inches="tight", format="pdf")
    plt.close(figure)
    return output_path
