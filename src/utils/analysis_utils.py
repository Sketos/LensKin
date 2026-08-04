import numpy as np

# NOTE:
from scipy import (
    interpolate,
)
from scipy.ndimage import zoom

# NOTE:
import autolens as al


def pixel_area_arcsec2_from_grid_2d(grid_2d):
    """Solid angle of one pixel on a regular autolens ``Grid2D`` (arcsec²)."""
    shape = grid_2d.shape_native
    coords = np.asarray(grid_2d).reshape(shape + (2,))
    dy = abs(float(coords[1, 0, 0] - coords[0, 0, 0]))
    dx = abs(float(coords[0, 1, 1] - coords[0, 0, 1]))
    return dx * dy


def integrated_line_flux_jy_kms(cube, z_step_kms):
    """
    Velocity-integrated line flux (Jy km/s) for a cube in Jy/pixel/channel.

    Matches KinMS ``intFlux`` normalization: ``cube.sum() * z_step_kms``.
    """
    return float(np.sum(cube) * z_step_kms)


def flux_conservation_lensing_report(
    source_cube,
    lensed_cube,
    z_step_kms,
    source_grid_2d,
    image_grid_2d,
    traced_source_coords=None,
    lensed_cube_fine=None,
    fine_image_grid_2d=None,
):
    """
    Report flux metrics for source → lensed → (optional) rebinned cubes.

    Cubes use Jy/pixel per velocity channel. **Source and lensed totals are
    not expected to match** (lensing changes integrated flux). When a fine
    lensed cube is supplied, we check that rebinning to the image-plane grid
    conserved flux up to ``rtol``.
    """
    source_cube = np.asarray(source_cube, dtype=float)
    lensed_cube = np.asarray(lensed_cube, dtype=float)
    if source_cube.shape[0] != lensed_cube.shape[0]:
        raise ValueError(
            f"Channel mismatch: source {source_cube.shape[0]} vs lensed {lensed_cube.shape[0]}"
        )

    src_per_channel = source_cube.sum(axis=(1, 2))
    img_per_channel = lensed_cube.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_channel_ratio = np.where(
            src_per_channel > 0,
            img_per_channel / src_per_channel,
            np.nan,
        )

    src_total = float(src_per_channel.sum())
    img_total = float(img_per_channel.sum())
    src_line = integrated_line_flux_jy_kms(source_cube, z_step_kms)
    img_line = integrated_line_flux_jy_kms(lensed_cube, z_step_kms)

    omega_src = pixel_area_arcsec2_from_grid_2d(source_grid_2d)
    omega_img = pixel_area_arcsec2_from_grid_2d(image_grid_2d)
    n_src = int(np.prod(source_grid_2d.shape_native))
    n_img = int(np.prod(image_grid_2d.shape_native))
    naive_zoom_flux_ratio = (n_img / n_src) if n_src > 0 else float("nan")

    traced_inside_fraction = None
    if traced_source_coords is not None:
        traced = np.asarray(traced_source_coords, dtype=float)
        coords = np.asarray(source_grid_2d).reshape(source_grid_2d.shape_native + (2,))
        y_min, y_max = float(coords[:, 0, 0].min()), float(coords[:, 0, 0].max())
        x_min, x_max = float(coords[0, :, 1].min()), float(coords[0, :, 1].max())
        inside = (
            (traced[:, 0] >= y_min)
            & (traced[:, 0] <= y_max)
            & (traced[:, 1] >= x_min)
            & (traced[:, 1] <= x_max)
        )
        traced_inside_fraction = float(inside.mean())

    rebin_line_flux_ratio = None
    rebin_line_flux_relative_error = None
    if lensed_cube_fine is not None and fine_image_grid_2d is not None:
        fine_line = integrated_line_flux_jy_kms(lensed_cube_fine, z_step_kms)
        rebin_line_flux_ratio = img_line / fine_line if fine_line > 0 else float("nan")
        rebin_line_flux_relative_error = (
            (img_line - fine_line) / fine_line if fine_line > 0 else float("nan")
        )

    return {
        "z_step_kms": float(z_step_kms),
        "source_pixel_area_arcsec2": omega_src,
        "image_pixel_area_arcsec2": omega_img,
        "source_n_pixels": n_src,
        "image_n_pixels": n_img,
        "naive_bilinear_downsample_flux_ratio": naive_zoom_flux_ratio,
        "traced_inside_source_bbox_fraction": traced_inside_fraction,
        "source_sum_per_channel_jy": src_per_channel,
        "lensed_sum_per_channel_jy": img_per_channel,
        "per_channel_flux_ratio": per_channel_ratio,
        "source_spatial_sum_jy": src_total,
        "lensed_spatial_sum_jy": img_total,
        "spatial_sum_ratio": img_total / src_total if src_total > 0 else float("nan"),
        "source_line_flux_jy_kms": src_line,
        "lensed_line_flux_jy_kms": img_line,
        "line_flux_ratio": img_line / src_line if src_line > 0 else float("nan"),
        "line_flux_relative_error": (img_line - src_line) / src_line if src_line > 0 else float("nan"),
        "pixel_area_ratio_image_over_source": omega_img / omega_src if omega_src > 0 else float("nan"),
        "rebin_line_flux_ratio": rebin_line_flux_ratio,
        "rebin_line_flux_relative_error": rebin_line_flux_relative_error,
        "fine_lensed_line_flux_jy_kms": (
            integrated_line_flux_jy_kms(lensed_cube_fine, z_step_kms)
            if lensed_cube_fine is not None
            else None
        ),
    }


def resample_image_to_shape_flux_conserving(image, target_shape):
    """
    Resample a Jy/pixel image to ``target_shape`` preserving total flux.

    Bilinear ``zoom`` does not preserve the sum for non-integer size ratios;
    the output is scaled so ``out.sum() == image.sum()``.
    """
    image = np.asarray(image, dtype=float)
    if image.shape == target_shape:
        return image
    resampled = resample_image_to_shape(image, target_shape)
    in_sum = float(image.sum())
    out_sum = float(resampled.sum())
    if in_sum > 0.0 and out_sum > 0.0:
        resampled *= in_sum / out_sum
    return resampled


def resample_image_to_shape(image, target_shape):
    image = np.asarray(image, dtype=float)
    if image.shape == target_shape:
        return image
    sy = target_shape[0] / image.shape[0]
    sx = target_shape[1] / image.shape[1]
    return zoom(image, (sy, sx), order=1)


def source_plane_image_for_interp(image, source_grid_2d, *, flip_y=True):
    """
    Align a source-plane model image with an autolens ``Grid2D`` and return
    1D coordinate axes for ``RegularGridInterpolator``.

    KinMS cubes are converted to autolens ``(v, y, x)`` ordering in
    ``profiles.kinMS`` / ``kinMSPixelized``, but KinMS still uses image
    indexing with y increasing opposite to autolens ``Grid2D`` y. With
    ``flip_y=True`` (LensKin default), flip the y axis before interpolation.
    Do **not** mirror x here — that is handled implicitly by the traced-grid
    mapping.

    Set ``flip_y=False`` when the mock / data were generated without this flip
    (e.g. ``generate_kinms_lensed_cube.py`` before it matched LensKin).
    """
    shape_2d = source_grid_2d.shape_native
    image = resample_image_to_shape(image=image, target_shape=shape_2d)
    image = np.asarray(image)
    if flip_y:
        image = image[::-1, :]
    coords = np.asarray(source_grid_2d).reshape(shape_2d + (2,))
    y_interp = coords[:, 0, 0]
    x_interp = coords[0, :, 1]
    return image, y_interp, x_interp


# NOTE: PyAutoLens
def traced_grids_of_planes_from(
    tracer,
    grid,
):
    # NOTE: ...
    if True: # al.__version__ == ???
        return tracer.traced_grid_2d_list_from(
            grid=grid
        )
    else:
        raise NotImplementedError()


def ray_trace(
    traced_grids_of_planes,
    image: np.ndarray,
    source_grid_2d=None,
    output_shape=None,
    image_grid_2d=None,
    flip_kinms_y=True,
    #interpolator=interpolate.RegularGridInterpolator,
):

    traced_grid_j = np.asarray(traced_grids_of_planes[1])

    pixel_area_scale = 1.0
    if source_grid_2d is not None:
        image, y_interp, x_interp = source_plane_image_for_interp(
            image=image,
            source_grid_2d=source_grid_2d,
            flip_y=flip_kinms_y,
        )
        if output_shape is None:
            raise ValueError("output_shape is required when source_grid_2d is set")
        out_shape = output_shape
        if image_grid_2d is not None:
            omega_src = pixel_area_arcsec2_from_grid_2d(source_grid_2d)
            omega_img = pixel_area_arcsec2_from_grid_2d(image_grid_2d)
            if omega_src > 0.0:
                pixel_area_scale = omega_img / omega_src
    else:
        traced_grid_i = np.asarray(traced_grids_of_planes[0])
        x_interp = np.unique(traced_grid_i[:, 0])
        y_interp = np.unique(traced_grid_i[:, 1])
        out_shape = image.shape

    # NOTE:
    image_interp = interpolate.RegularGridInterpolator(
        points=(y_interp, x_interp),
        values=image,
        method="linear",
        bounds_error=False,
        fill_value=0.0
    )

    # NOTE:
    lensed_image = image_interp(
        traced_grid_j
    ).reshape(out_shape)

    lensed_image *= pixel_area_scale

    return lensed_image


def image_grid_2d_with_n_pixels(template_grid_2d, n_pixels: int):
    """
    Uniform ``Grid2D`` over the same sky extent as ``template_grid_2d``.

    Used to lens at fine image-plane resolution before flux-conserving rebin.
    """
    shape = template_grid_2d.shape_native
    coords = np.asarray(template_grid_2d).reshape(shape + (2,))
    dy = abs(float(coords[1, 0, 0] - coords[0, 0, 0]))
    dx = abs(float(coords[0, 1, 1] - coords[0, 0, 1]))
    n_y, n_x = shape
    return al.Grid2D.uniform(
        shape_native=(n_pixels, n_pixels),
        pixel_scales=(dy * n_y / n_pixels, dx * n_x / n_pixels),
    )


def lensed_cube_on_grid(
    cube,
    tracer,
    image_grid_2d,
    source_grid_2d,
    z_mask=None,
    flip_kinms_y=True,
):
    """Lens a source-plane cube onto ``image_grid_2d`` (one pixel per ray)."""
    traced_grids_of_planes = traced_grids_of_planes_from(
        tracer=tracer,
        grid=image_grid_2d,
    )
    output_shape = image_grid_2d.shape_native
    lensed_cube = np.zeros((cube.shape[0],) + tuple(output_shape))
    for i, image in enumerate(cube):
        lensed_cube[i] = ray_trace(
            traced_grids_of_planes=traced_grids_of_planes,
            image=image,
            source_grid_2d=source_grid_2d,
            output_shape=output_shape,
            image_grid_2d=image_grid_2d,
            flip_kinms_y=flip_kinms_y,
        )
    return lensed_cube


def lensed_cube_from_tracer(
    cube,
    tracer,
    grid, # NOTE: grid_3d.grid_2d
    z_mask=None,
    source_grid_2d=None,
    output_shape=None,
    interpolator=None,
    flip_kinms_y=True,
):
    if output_shape is None:
        if hasattr(grid, "shape_native"):
            output_shape = grid.shape_native
        elif source_grid_2d is not None:
            raise ValueError("output_shape is required when source_grid_2d is set")
    coarse_shape = tuple(output_shape)

    if source_grid_2d is not None:
        n_fine = int(source_grid_2d.shape_native[0])
        fine_image_grid = image_grid_2d_with_n_pixels(grid, n_fine)
        lensed_fine = lensed_cube_on_grid(
            cube=cube,
            tracer=tracer,
            image_grid_2d=fine_image_grid,
            source_grid_2d=source_grid_2d,
            z_mask=z_mask,
            flip_kinms_y=flip_kinms_y,
        )
        if fine_image_grid.shape_native == coarse_shape:
            lensed_coarse = lensed_fine
        else:
            lensed_coarse = np.stack(
                [
                    resample_image_to_shape_flux_conserving(
                        lensed_fine[i],
                        coarse_shape,
                    )
                    for i in range(lensed_fine.shape[0])
                ],
                axis=0,
            )
        return lensed_coarse

    traced_grids_of_planes = traced_grids_of_planes_from(
        tracer=tracer, grid=grid
    )
    lensed_cube = np.zeros(shape=cube.shape)
    for i, image in enumerate(cube):
        lensed_cube[i] = ray_trace(
            traced_grids_of_planes=traced_grids_of_planes,
            image=image,
            source_grid_2d=None,
            output_shape=None,
            image_grid_2d=grid,
            flip_kinms_y=flip_kinms_y,
        )
    return lensed_cube
