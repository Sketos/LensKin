"""
Helpers for standalone phase-1 pixelization tests.

Mirrors autolens_workspace/scripts/interferometer/features/pixelization/fit.py
using LensKin exported FITS cubes and ``lens_mass_model`` from settings JSON.
"""

from pathlib import Path

import autolens as al
import numpy as np

from src.pipelines import reconstruction
from src.pipelines.moment0_noise import moment0_settings_from_reconstruction
from src.pipelines.phase1_likelihood import (
    format_likelihood_breakdown,
    phase1_likelihood_breakdown,
)
from src.pipelines.runner import load_cube_data, load_cube_data_weights
from src.utils import autolens_utils


def fixed_lens_galaxy_from_settings(settings):
    """Build a fixed lens ``al.Galaxy`` from ``lens_mass_model`` in settings."""
    mass_cfg = settings["lens_mass_model"]
    mass = al.mp.PowerLaw(
        centre=(mass_cfg["centre_0"], mass_cfg["centre_1"]),
        ell_comps=(mass_cfg["elliptical_comps_0"], mass_cfg["elliptical_comps_1"]),
        einstein_radius=mass_cfg["einstein_radius"],
        slope=mass_cfg["slope"],
    )

    shear_swap = settings.get("swap_shear_components", True)
    gamma_1 = (
        mass_cfg["shear_elliptical_comps_1"]
        if shear_swap
        else mass_cfg["shear_elliptical_comps_0"]
    )
    gamma_2 = (
        mass_cfg["shear_elliptical_comps_0"]
        if shear_swap
        else mass_cfg["shear_elliptical_comps_1"]
    )
    shear = al.mp.ExternalShear(gamma_1=gamma_1, gamma_2=gamma_2)

    lens_kwargs = {
        "redshift": settings["redshift_lens"],
        "mass": mass,
        "shear": shear,
    }

    if (
        "multipole_m3_elliptical_comps_0" in mass_cfg
        and "multipole_m3_elliptical_comps_1" in mass_cfg
    ):
        lens_kwargs["multipole_m3"] = al.mp.PowerLawMultipole(
            m=3,
            centre=mass.centre,
            einstein_radius=mass.einstein_radius,
            slope=mass.slope,
            multipole_comps=(
                mass_cfg["multipole_m3_elliptical_comps_0"],
                mass_cfg["multipole_m3_elliptical_comps_1"],
            ),
        )

    if (
        "multipole_m4_elliptical_comps_0" in mass_cfg
        and "multipole_m4_elliptical_comps_1" in mass_cfg
    ):
        lens_kwargs["multipole_m4"] = al.mp.PowerLawMultipole(
            m=4,
            centre=mass.centre,
            einstein_radius=mass.einstein_radius,
            slope=mass.slope,
            multipole_comps=(
                mass_cfg["multipole_m4_elliptical_comps_0"],
                mass_cfg["multipole_m4_elliptical_comps_1"],
            ),
        )

    return al.Galaxy(**lens_kwargs)


def _transformer_class(name):
    if name in {None, "auto", "default"}:
        from src.utils.autolens_utils import default_transformer_class

        return default_transformer_class()
    if name == "dft":
        return al.TransformerDFT
    if name in {"nufft", "nufftax"}:
        return al.TransformerNUFFT
    if name in {"pynufft", "nufft_pynufft"}:
        return al.TransformerNUFFTPyNUFFT
    raise ValueError(
        f"Unsupported transformer: {name!r}. "
        "Expected 'auto', 'dft', 'nufft'/'nufftax', or 'pynufft'."
    )


def single_channel_dataset_from(
    uv_wavelengths,
    visibilities,
    sigma,
    mask_2d,
    channel=None,
    transformer_class=al.TransformerNUFFT,
):
    """Build a single-channel ``Interferometer`` dataset from a LensKin cube."""
    n_channels = visibilities.shape[0]
    if channel is None:
        channel = n_channels // 2

    vis_complex = visibilities[channel, :, 0] + 1j * visibilities[channel, :, 1]
    sigma_ch = sigma[channel]

    data = al.Visibilities(visibilities=vis_complex)
    noise_map = al.VisibilitiesNoiseMap(visibilities=sigma_ch)

    kwargs = {}
    if transformer_class is al.TransformerDFT:
        kwargs["raise_error_dft_visibilities_limit"] = False

    return al.Interferometer(
        data=data,
        noise_map=noise_map,
        uv_wavelengths=uv_wavelengths[channel],
        real_space_mask=mask_2d,
        transformer_class=transformer_class,
        **kwargs,
    )


def interferometer_dataset_from_settings(
    settings,
    dataset_kind="moment0",
    channel=None,
    transformer="auto",
):
    """
    Load LensKin cube data and return an ``Interferometer`` for phase-1 tests.

    ``dataset_kind`` is ``moment0`` (velocity-averaged, pipeline default) or
    ``channel`` (single spectral channel, workspace-style).
    """
    _, uv_wavelengths, visibilities, sigma = load_cube_data(settings)
    mask_2d = reconstruction.reconstruction_mask_from_settings(settings)
    # Prefer reconstruction.transformer / top-level transformer when present.
    if transformer in {None, "auto", "default"}:
        rec_t = settings.get("reconstruction", {}).get("transformer")
        top_t = settings.get("transformer")
        transformer = rec_t or top_t or "auto"
    transformer_class = _transformer_class(transformer)
    moment0_settings = moment0_settings_from_reconstruction(settings["reconstruction"])
    weights = load_cube_data_weights(settings)

    if dataset_kind == "moment0":
        return reconstruction.moment0_dataset_from(
            uv_wavelengths=uv_wavelengths,
            visibilities=visibilities,
            sigma=sigma,
            mask_2d=mask_2d,
            transformer_class=transformer_class,
            moment0_settings=moment0_settings,
            weights=weights,
        )
    if dataset_kind == "channel":
        return single_channel_dataset_from(
            uv_wavelengths=uv_wavelengths,
            visibilities=visibilities,
            sigma=sigma,
            mask_2d=mask_2d,
            channel=channel,
            transformer_class=transformer_class,
        )
    raise ValueError(
        f"Unsupported dataset_kind: {dataset_kind!r}. Expected 'moment0' or 'channel'."
    )


def pixelization_from_settings(
    settings,
    mask_2d,
    regularization_coefficient=1e6,
    mesh_type=None,
    regularization_overrides=None,
):
    """Build a workspace-style pixelization (rectangular or Delaunay)."""
    rec_cfg = settings["reconstruction"]
    mesh_type = mesh_type or rec_cfg.get("mesh_type", "rectangular_adapt_density")
    reg_cfg = rec_cfg.get("regularization", {})
    reg_type = reconstruction.regularization_type_from_settings(rec_cfg)
    reconstruction.validate_mesh_regularization_pair(mesh_type, reg_type)

    overrides = dict(regularization_overrides or {})
    if regularization_coefficient is not None:
        if reg_type in reconstruction._ADAPT_REGULARIZATION_TYPES:
            overrides.setdefault("outer_coefficient", regularization_coefficient)
            overrides.setdefault(
                "inner_coefficient", regularization_coefficient * 0.0026
            )
        else:
            overrides.setdefault("coefficient", regularization_coefficient)

    if mesh_type == "delaunay":
        image_plane_mesh_grid = reconstruction._image_plane_mesh_grid_from_settings(
            rec_cfg, mask_2d
        )
        edge_pixels = int(rec_cfg.get("delaunay_edge_pixels", 30))
        areas_factor = float(rec_cfg.get("delaunay_areas_factor", 0.5))
        mesh = al.mesh.Delaunay(
            pixels=image_plane_mesh_grid.shape[0],
            zeroed_pixels=edge_pixels,
            areas_factor=areas_factor,
        )
        if reg_type in reconstruction._ADAPT_REGULARIZATION_TYPES:
            values = reconstruction.fixed_regularization_values_from_settings(
                reg_cfg, reg_type, overrides=overrides
            )
            regularization = reconstruction._REGULARIZATION_CLASSES[reg_type](
                **values
            )
        elif reg_type == "constant_split":
            values = reconstruction.fixed_regularization_values_from_settings(
                reg_cfg, reg_type, overrides=overrides
            )
            regularization = al.reg.ConstantSplit(coefficient=values["coefficient"])
        else:
            values = reconstruction.fixed_regularization_values_from_settings(
                reg_cfg, reg_type, overrides=overrides
            )
            regularization = al.reg.Constant(coefficient=values["coefficient"])
        pixelization = al.Pixelization(mesh=mesh, regularization=regularization)
        return pixelization, image_plane_mesh_grid

    mesh_cls = reconstruction._MESH_MODEL_CLASSES.get(mesh_type)
    if mesh_cls is None:
        raise ValueError(
            f"Unsupported mesh_type for phase-1 test: {mesh_type!r}. "
            f"Choose 'delaunay' or one of {sorted(reconstruction._MESH_MODEL_CLASSES)}."
        )

    mesh_shape = tuple(rec_cfg.get("mesh_shape", [28, 28]))
    mesh = mesh_cls(shape=mesh_shape)
    if reg_type in reconstruction._ADAPT_REGULARIZATION_TYPES:
        values = reconstruction.fixed_regularization_values_from_settings(
            reg_cfg, reg_type, overrides=overrides
        )
        regularization = reconstruction._REGULARIZATION_CLASSES[reg_type](**values)
    else:
        values = reconstruction.fixed_regularization_values_from_settings(
            reg_cfg, reg_type, overrides=overrides
        )
        regularization = al.reg.Constant(coefficient=values["coefficient"])
    return al.Pixelization(mesh=mesh, regularization=regularization), None


def adapt_images_for_pixelization(source_galaxy, image_plane_mesh_grid, dataset=None, settings=None):
    """Return ``AdaptImages`` for Delaunay and/or adaptive regularization fits."""
    rec_cfg = (settings or {}).get("reconstruction", {})
    reg_type = reconstruction.regularization_type_from_settings(rec_cfg) if settings else "constant"
    mesh_type = rec_cfg.get("mesh_type", "rectangular_adapt_density")

    image_plane_mesh_grid_dict = {}
    image_dict = {}

    if image_plane_mesh_grid is not None:
        image_plane_mesh_grid_dict[source_galaxy] = image_plane_mesh_grid

    needs_dirty_image = (
        mesh_type == "rectangular_adapt_image"
        or reconstruction.regularization_needs_adapt_image(reg_type)
    )
    if needs_dirty_image:
        if dataset is None:
            raise ValueError(
                f"Adaptive regularization ({reg_type!r}) requires the dataset "
                "to build a dirty-image adapt map."
            )
        dirty_image = dataset.dirty_image
        if dirty_image is None:
            dirty_image = dataset.masked_dirty_image
        image_dict[source_galaxy] = dirty_image

    if not image_plane_mesh_grid_dict and not image_dict:
        return None

    kwargs = {}
    if image_dict:
        kwargs["galaxy_image_dict"] = image_dict
    if image_plane_mesh_grid_dict:
        kwargs["galaxy_image_plane_mesh_grid_dict"] = image_plane_mesh_grid_dict
    return al.AdaptImages(**kwargs)


def run_workspace_style_fit(
    settings,
    output_dir,
    dataset_kind="moment0",
    channel=None,
    transformer="auto",
    regularization_coefficient=None,
    mesh_type=None,
    regularization_overrides=None,
    use_jax=True,
    show_progress=True,
):
    """
    Fit a fixed-lens pixelized source model using ``FitInterferometer``.

    Returns the fit object and the output directory used for plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_2d = reconstruction.reconstruction_mask_from_settings(settings)

    dataset = interferometer_dataset_from_settings(
        settings=settings,
        dataset_kind=dataset_kind,
        channel=channel,
        transformer=transformer,
    )

    if use_jax:
        dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=show_progress)

    lens = fixed_lens_galaxy_from_settings(settings)
    pixelization, image_plane_mesh_grid = pixelization_from_settings(
        settings=settings,
        mask_2d=mask_2d,
        regularization_coefficient=regularization_coefficient,
        mesh_type=mesh_type,
        regularization_overrides=regularization_overrides,
    )
    source = al.Galaxy(
        redshift=settings["redshift_source"],
        pixelization=pixelization,
    )
    tracer = al.Tracer(galaxies=[lens, source])
    adapt_images = adapt_images_for_pixelization(
        source,
        image_plane_mesh_grid,
        dataset=dataset,
        settings=settings,
    )

    fit = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        adapt_images=adapt_images,
        settings=al.Settings(
            use_positive_only_solver=reconstruction.use_positive_only_solver_from_settings(
                settings
            )
        ),
    )

    save_workspace_style_plots(
        fit=fit,
        output_dir=output_dir,
        settings=settings,
        mesh_type=mesh_type or settings["reconstruction"].get("mesh_type"),
    )
    return fit, output_dir


def _mat_plots_for_output(output_dir, filename):
    """Build MatPlot wrappers that save PNGs (autolens_workspace style)."""
    import autolens.plot as aplt

    output = aplt.Output(path=str(output_dir), filename=filename, format="png")
    return aplt.MatPlot2D(output=output), aplt.MatPlot1D(output=output)


def save_workspace_style_plots(fit, output_dir, settings, mesh_type=None):
    """Save diagnostic plots analogous to the autolens_workspace example."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_type = mesh_type or settings["reconstruction"].get("mesh_type", "rectangular_adapt_density")

    try:
        import autolens.plot as aplt

        mat_plot_2d, mat_plot_1d = _mat_plots_for_output(output_dir, "dataset")
        dataset_plotter = aplt.InterferometerPlotter(
            dataset=fit.dataset,
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )
        dataset_plotter.subplot_dataset()

        mat_plot_2d, mat_plot_1d = _mat_plots_for_output(output_dir, "dataset_dirty")
        dataset_plotter = aplt.InterferometerPlotter(
            dataset=fit.dataset,
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )
        dataset_plotter.subplot_dirty_images()

        mat_plot_2d, mat_plot_1d = _mat_plots_for_output(output_dir, "fit")
        fit_plotter = aplt.FitInterferometerPlotter(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )
        fit_plotter.subplot_fit()

        mat_plot_2d, mat_plot_1d = _mat_plots_for_output(output_dir, "fit_dirty")
        fit_plotter = aplt.FitInterferometerPlotter(
            fit=fit,
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )
        fit_plotter.subplot_fit_dirty_images()

        mat_plot_2d, mat_plot_1d = _mat_plots_for_output(output_dir, "inversion_mapper")
        inversion_plotter = aplt.InversionPlotter(
            inversion=fit.inversion,
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )
        inversion_plotter.subplot_of_mapper(mapper_index=0)

        mat_plot_2d, mat_plot_1d = _mat_plots_for_output(output_dir, "inversion_mappings")
        inversion_plotter = aplt.InversionPlotter(
            inversion=fit.inversion,
            mat_plot_2d=mat_plot_2d,
            mat_plot_1d=mat_plot_1d,
        )
        inversion_plotter.subplot_mappings(pixelization_index=0)
        used_autolens_plot = True
    except Exception as exc:
        print(f"autolens.plot diagnostics unavailable ({exc}); using fallback plots.")
        used_autolens_plot = False

    from src.pipelines.pixelized_plots import save_fit_triplet, save_image

    data = autolens_utils.array2d_to_numpy(fit.dirty_image)
    model = autolens_utils.array2d_to_numpy(fit.dirty_model_image)
    residuals = data - model
    _, _, real_space_width = autolens_utils.source_grid_from_settings(settings)
    extent = autolens_utils.image_extent_arcsec(real_space_width)

    save_fit_triplet(
        data=data,
        model=model,
        residuals=residuals,
        path=output_dir / "fit_triplet.png",
        titles=("Dirty data", "Dirty model", "Residuals"),
        extent=extent,
        scale_mode="sigma",
        residual_sigma=autolens_utils.dirty_noise_map_mc_from_fit(fit),
    )

    inversion = fit.inversion
    mapper = inversion.linear_obj_list[0]
    reconstruction_values = np.asarray(getattr(inversion.reconstruction, "array", inversion.reconstruction))
    mesh_grid = np.asarray(
        getattr(mapper.source_plane_mesh_grid, "array", mapper.source_plane_mesh_grid)
    )

    np.savez(
        output_dir / "phase1_arrays.npz",
        reconstruction=reconstruction_values,
        source_plane_mesh_grid=mesh_grid,
        log_likelihood=fit.log_likelihood,
        chi_squared=fit.chi_squared,
    )

    with open(output_dir / "fit_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(f"mesh_type = {mesh_type}\n")
        reg = fit.tracer.galaxies[1].pixelization.regularization
        reg_type = reconstruction.regularization_type_from_settings(
            settings["reconstruction"]
        )
        if hasattr(reg, "coefficient"):
            handle.write(f"regularization_type = {reg_type}\n")
            handle.write(f"coefficient = {reg.coefficient}\n")
        else:
            handle.write(f"regularization_type = {reg_type}\n")
            handle.write(f"inner_coefficient = {reg.inner_coefficient}\n")
            handle.write(f"outer_coefficient = {reg.outer_coefficient}\n")
            handle.write(f"signal_scale = {reg.signal_scale}\n")
        breakdown = phase1_likelihood_breakdown(fit)
        handle.write(f"{format_likelihood_breakdown(breakdown)}\n")
        handle.write(
            "figure_of_merit_note = LBFGS maximizes log_evidence "
            "(figure_of_merit), not log_likelihood_data alone\n"
        )
        handle.write(f"used_autolens_plot = {used_autolens_plot}\n")
        mass_cfg = settings["lens_mass_model"]
        handle.write(
            "lens_mass_model centre = "
            f"({mass_cfg['centre_0']}, {mass_cfg['centre_1']})\n"
        )
        handle.write(f"einstein_radius = {mass_cfg['einstein_radius']}\n")

    return output_dir
