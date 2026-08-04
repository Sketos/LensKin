from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import autolens_utils, plot_utils


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def plots_root_from_settings(settings):
    return Path(settings.get("plots_path", Path(settings["output_path"]) / "plots"))


def save_image(image, path, title=None, cmap="viridis", extent=None, vmin=None, vmax=None):
    _ensure_dir(Path(path).parent)
    figure, axis = plt.subplots(figsize=(6, 5))
    im = axis.imshow(
        image,
        origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
        cmap=cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    if title is not None:
        axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    if cmap == "RdBu_r" or vmin is not None or vmax is not None:
        figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def save_cube(image_cube, path, ncols=10, extent=None):
    _ensure_dir(Path(path).parent)
    figure, _ = plot_utils.plot_cube(cube=image_cube, ncols=ncols, extent=extent)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def save_fit_triplet(
    data,
    model,
    residuals,
    path,
    titles=("Data", "Model", "Residuals"),
    extent=None,
    scale_mode="sigma",
    residual_sigma=None,
):
    """
    Save a data / model / residual image triplet.

    ``scale_mode``:
      - ``sigma`` (**default**): data scale for data/model; residual panel shows
        ``residuals / residual_sigma`` with a fixed ±5σ colour bar.
        ``residual_sigma`` may be a scalar or an array broadcastable to
        ``residuals``. If omitted, falls back to ``residual`` scaling.
      - ``shared``: one color scale for all panels (legacy).
      - ``data``: data sets the scale; model uses the same (best for morphology).
      - ``residual``: data scale for data/model; symmetric absolute residual scale.
    """
    _ensure_dir(Path(path).parent)
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    if scale_mode == "sigma" and residual_sigma is None:
        scale_mode = "residual"

    if scale_mode == "shared":
        vmin = np.nanmin([data, model, residuals])
        vmax = np.nanmax([data, model, residuals])
        data_vmin, data_vmax = vmin, vmax
        model_vmin, model_vmax = vmin, vmax
        res_vmin, res_vmax = vmin, vmax
        residual_panel = residuals
        residual_cmap = "viridis"
    elif scale_mode == "data":
        data_vmin, data_vmax = np.nanmin(data), np.nanmax(data)
        model_vmin, model_vmax = data_vmin, data_vmax
        res_limit = np.nanmax(np.abs(residuals))
        res_vmin, res_vmax = -res_limit, res_limit
        residual_panel = residuals
        residual_cmap = "RdBu_r"
    elif scale_mode == "residual":
        data_vmin, data_vmax = np.nanmin(data), np.nanmax(data)
        model_vmin, model_vmax = data_vmin, data_vmax
        res_limit = np.nanmax(np.abs(residuals))
        res_vmin, res_vmax = -res_limit, res_limit
        residual_panel = residuals
        residual_cmap = "RdBu_r"
    elif scale_mode == "sigma":
        sigma = np.asarray(residual_sigma, dtype=float)
        safe = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, np.nan)
        residual_panel = residuals / safe
        data_vmin, data_vmax = np.nanmin(data), np.nanmax(data)
        model_vmin, model_vmax = data_vmin, data_vmax
        # Fixed ±5σ colour bar so noiseless systematics and noisy data share a scale.
        display_limit = 5.0
        res_vmin, res_vmax = -display_limit, display_limit
        residual_cmap = "RdBu_r"
        titles = list(titles)
        if len(titles) >= 3 and "σ" not in titles[2]:
            titles[2] = f"{titles[2]} / σ"
    else:
        raise ValueError(f"Unsupported scale_mode: {scale_mode!r}")

    images = (data, model, residual_panel)
    vmins = (data_vmin, model_vmin, res_vmin)
    vmaxs = (data_vmax, model_vmax, res_vmax)
    cmaps = ("viridis", "viridis", residual_cmap)

    residual_mappable = None
    for i, (axis, image, title, vmin, vmax, cmap) in enumerate(
        zip(axes, images, titles, vmins, vmaxs, cmaps)
    ):
        im = axis.imshow(
            image,
            origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        if i == 2:
            residual_mappable = im

    if scale_mode in ("residual", "sigma", "data") and residual_mappable is not None:
        cbar = figure.colorbar(
            residual_mappable,
            ax=axes[2],
            fraction=0.046,
            pad=0.04,
        )
        if scale_mode == "sigma":
            cbar.set_label(r"residual / $\sigma$")
            peak = (
                float(np.nanmax(np.abs(residual_panel)))
                if residual_panel.size
                else 0.0
            )
            finite = residual_panel[np.isfinite(residual_panel)]
            rmin = float(np.min(finite)) if finite.size else float("nan")
            rmax = float(np.max(finite)) if finite.size else float("nan")
            figure.suptitle(
                f"Residual / σ: colour bar ±{display_limit:.0f}σ; "
                f"map min={rmin:.2f}σ, max={rmax:.2f}σ, peak|r|={peak:.2f}σ",
                y=1.03,
                fontsize=10,
            )
        else:
            cbar.set_label("residual")

    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)


def plot_dirty_data(visibilities, transformers, output_dir, extent=None):
    """
    Save dirty images for each channel and the velocity-averaged map (Jy/pixel).
    """
    output_dir = Path(output_dir)
    dirty_cube = autolens_utils.dirty_cube_from(
        visibilities=visibilities,
        transformers=transformers,
    )
    dirty_mom0 = dirty_cube.mean(axis=0)

    save_cube(
        image_cube=dirty_cube,
        path=output_dir / "dirty_channels.png",
        extent=extent,
    )
    save_image(
        image=dirty_mom0,
        path=output_dir / "dirty_mom0.png",
        title="Dirty velocity-averaged (Jy/pixel)",
        extent=extent,
    )


def plot_phase1_fit(
    result,
    sb_map,
    output_dir,
    settings,
    sb_map_extent=None,
    *,
    sb_preview_n_pixels=30,
):
    """
    Save phase-1 velocity-averaged dirty data/model/residuals and the SB map.

    Residuals are shown in units of a Monte-Carlo dirty-image RMS estimated
    from the visibility noise map. Do **not** use Autolens
    ``fit.dirty_noise_map`` here — that is only an IFT of the σ visibilities,
    not an image-plane RMS (it can be negative / zero).

    Also writes the reconstruction interpolated onto a regular
    ``sb_preview_n_pixels``² grid (default 30²) over the source-mesh bounding
    box, plus a two-panel comparison figure.
    """
    from src.grid.grid import Grid3D
    from src.pipelines import reconstruction as reconstruction_mod

    output_dir = Path(output_dir)
    fit = result.max_log_likelihood_fit
    _, _, real_space_width = autolens_utils.source_grid_from_settings(settings)
    extent = autolens_utils.image_extent_arcsec(real_space_width)

    data = autolens_utils.array2d_to_numpy(fit.dirty_image)
    model = autolens_utils.array2d_to_numpy(fit.dirty_model_image)
    residuals = data - model
    residual_sigma = autolens_utils.dirty_noise_map_mc_from_fit(fit)

    save_fit_triplet(
        data=data,
        model=model,
        residuals=residuals,
        path=output_dir / "fit_triplet.png",
        titles=("Dirty data", "Dirty model", "Residuals"),
        extent=extent,
        scale_mode="sigma",
        residual_sigma=residual_sigma,
    )
    save_image(data, output_dir / "data.png", title="Dirty data", extent=extent)
    save_image(model, output_dir / "model.png", title="Dirty model", extent=extent)
    # Standalone residual map also in σ units.
    safe = np.where(
        np.isfinite(residual_sigma) & (residual_sigma > 0.0),
        residual_sigma,
        np.nan,
    )
    resid_sigma = residuals / safe
    lim = float(np.nanmax(np.abs(resid_sigma))) if np.any(np.isfinite(resid_sigma)) else 5.0
    lim = min(max(lim, 1.0), 5.0)
    save_image(
        resid_sigma,
        output_dir / "residuals.png",
        title="Residuals / σ",
        cmap="RdBu_r",
        extent=extent,
        vmin=-lim,
        vmax=lim,
    )
    save_image(
        sb_map,
        output_dir / "sb_map.png",
        title="Interpolated SB map (Jy/pixel)",
        extent=sb_map_extent,
    )

    # Preview on a fixed regular grid (default 30²) over the mesh bbox.
    # Grid3D currently requires n_channels > 1; only grid_2d is used for SB.
    n_preview = int(sb_preview_n_pixels)
    bbox = reconstruction_mod.source_plane_bounding_box_from_result(result)
    preview_grid = Grid3D.bounding_box(
        bounding_box=bbox,
        n_pixels=n_preview,
        n_channels=2,
    )
    sb_preview = reconstruction_mod.source_sb_from_fit(fit, preview_grid.grid_2d)
    preview_extent = autolens_utils.image_extent_from_bounding_box(bbox)
    save_image(
        sb_preview,
        output_dir / f"sb_map_{n_preview}x{n_preview}.png",
        title=f"SB on {n_preview}² regular grid (Jy/pixel)",
        extent=preview_extent,
    )

    # Side-by-side: caller SB map (usually fine KinMS grid) vs 30² preview.
    _ensure_dir(output_dir)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, image, title, ext in (
        (axes[0], sb_map, "SB (phase-2 / fine grid)", sb_map_extent),
        (
            axes[1],
            sb_preview,
            f"SB ({n_preview}² regular grid)",
            preview_extent,
        ),
    ):
        axis.imshow(
            image,
            origin=plot_utils.DEFAULT_IMAGE_ORIGIN,
            cmap="viridis",
            extent=ext,
        )
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output_dir / "sb_map_panels.png", bbox_inches="tight", dpi=150)
    plt.close(figure)


def plot_phase2_fit(analysis_instance, result, output_dir, settings):
    """
    Save phase-2 dirty data/model/residual cubes for the best-fit kinematic model.
    """
    output_dir = Path(output_dir)
    fit_dir = output_dir / "fit_dataset"
    _ensure_dir(fit_dir)

    analysis_instance.visualizer.directory = str(fit_dir)
    analysis_instance.visualizer.visualize_data()

    model_data = analysis_instance.model_data_from_instance(
        instance=result.max_log_likelihood_instance
    )
    analysis_instance.visualizer.visualize(
        model_data=model_data,
        during_analysis=False,
    )

    dirty_model_cube = autolens_utils.dirty_cube_from(
        visibilities=model_data,
        transformers=analysis_instance.transformers,
    )
    dirty_data_cube = analysis_instance.visualizer.dirty_cube
    z_step = analysis_instance.masked_dataset.z_step_kms
    dirty_mom0_data = dirty_data_cube.sum(axis=0) * z_step
    dirty_mom0_model = dirty_model_cube.sum(axis=0) * z_step
    _, _, real_space_width = autolens_utils.source_grid_from_settings(settings)
    kin_extent = autolens_utils.image_extent_arcsec(real_space_width)

    dirty_sigma = autolens_utils.dirty_noise_cube_from(
        noise_map=analysis_instance.masked_dataset.noise_map,
        transformers=analysis_instance.transformers,
        n_realizations=8,
        seed=0,
    )
    mom0_sigma = autolens_utils.dirty_mom0_noise_from_channel_noise(
        dirty_sigma, z_step
    )

    save_fit_triplet(
        data=dirty_mom0_data,
        model=dirty_mom0_model,
        residuals=dirty_mom0_data - dirty_mom0_model,
        path=output_dir / "fit_triplet_mom0.png",
        titles=("Dirty data (mom0)", "Dirty model (mom0)", "Residuals (mom0)"),
        extent=kin_extent,
        scale_mode="sigma",
        residual_sigma=mom0_sigma,
    )
