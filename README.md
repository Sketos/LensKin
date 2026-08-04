# LensKin

PyAutoLens/PyAutoFit workspace for fitting lensed ALMA uv spectral-line cubes with GalPaK or KinMS source models.

## Repository layout

```
LensKin/
├── config/                 # AutoFit / AutoLens YAML configuration
├── settings/
│   ├── runners/            # Fit-pipeline JSON settings
│   └── dataprep/           # Data-prep JSON settings
├── scripts/
│   ├── run_fit.py          # Generic fit entry point (all normalization modes)
│   ├── run_pixelized_fit.py # Legacy alias for two-phase KinMS fits
│   ├── run_dataprep.py     # Generic data-prep entry point
│   ├── make_cornerplot.py  # Corner plot from a completed run
│   ├── test_phase1_pixelization.py   # Phase-1 pixelized reconstruction tests
│   ├── profile_phase1_regularization.py  # Reg-coefficient / FoM diagnostics
│   ├── check_moment0_noise.py        # Moment-0 vs channel noise diagnostic
│   ├── generate_unlensed_mock_and_diagnose.py  # Unlensed KinMS self-mock + truth diagnostics
│   ├── smoke_unlensed_three_modes.py          # Lens-off smoke test (all normalization modes)
│   ├── generate_lensed_mock_and_diagnose.py    # Lensed KinMS self-mock + parametric truth diagnostics
│   ├── generate_lensed_mock_pixelized_and_diagnose.py  # Lensed mock + KinMSPixelized truth diagnostics
│   ├── trial_source_grid_regularization.py    # Phase-1 grid/reg scan vs pixelized truth floor
│   ├── diagnose_mode2_fit.py         # Post-fit dirty mom0 residual diagnostics
│   ├── runners/            # Target-specific fit wrappers
│   ├── dataprep/           # Target-specific data-prep wrappers
│   ├── plotting/           # Ad-hoc plotting scripts
│   ├── slurm/              # Slurm submit wrappers (_run_fit.sh, submit_*.sh)
│   └── tutorial.py         # Synthetic tutorial fit
├── src/
│   ├── pipelines/          # Shared runner / dataprep logic
│   ├── analysis/
│   ├── dataset/
│   ├── fit/
│   ├── grid/
│   ├── mask/
│   ├── model/
│   └── utils/
│       ├── ...
│       └── primary_beam.py    # Gaussian primary-beam attenuation (HPBW = 1.13λ/D)
├── tests/                  # Unit tests (pytest)
└── output/                 # Search results (generated)
```

## Running fits

From the repository root:

```bash
python scripts/run_fit.py --settings settings/runners/SPT0538_CO9-8.json
```

Or use a target wrapper (same behaviour, default settings baked in):

```bash
python scripts/runners/SPT0538_CO9-8.py
python scripts/runners/SPT0538_mockSMBH.py
python scripts/runners/HERMES_J021830.5-053124.py
python scripts/runners/SPT0538_CO9-8_pixelized.py
python scripts/runners/kinms_mock_pixelized.py
python scripts/runners/kinms_mock_parametric_flux.py
```

Override settings with `--settings /path/to/custom.json`.

`scripts/run_fit.py` dispatches automatically on `normalization_mode` in the settings file. `scripts/run_pixelized_fit.py` remains as a legacy entry point for two-phase KinMS fits.

Image-plane pixel scale defaults to Nyquist sampling of the longest baseline (`0.5 × λ/b_max`); see [Image-plane grid (Nyquist default)](#image-plane-grid-nyquist-default).

## KinMS source normalization modes

KinMS fits support three source normalization schemes, selected with the top-level `normalization_mode` key in the runner settings JSON.

| Mode | Settings value | Phase 1 | Source model | Flux / luminosity |
|------|----------------|---------|--------------|-------------------|
| 1 — Fully parametric | `"parametric"` | No | `KinMS` exponential disk (`effective_radius`) | `intensity` is a free fit parameter |
| 2 — Parametric shape, phase-1 flux | `"parametric_flux_from_phase1"` | Yes | `KinMS` exponential disk | `intensity` fixed to integrated flux from phase-1 pixelized reconstruction |
| 3 — Pixelized source | `"pixelized"` | Yes | `KinMSPixelized` cloudlets from phase-1 SB map | Spatial structure and total flux fixed from phase 1; only kinematics are fitted in phase 2 |

### Mode 1 — Fully parametric (`parametric`)

Single-phase fit. No `reconstruction` block is required.

- Surface brightness: parametric exponential disk (`effective_radius`)
- Intensity: free `intensity` prior
- Example: `settings/runners/SPT0538_mockSMBH.json`

```json
{
  "model_name": "KinMS",
  "normalization_mode": "parametric",
  "priors": {
    "intensity": {"type": "LogUniformPrior", "lower_limit": 0.001, "upper_limit": 0.1},
    "effective_radius": {"type": "LogUniformPrior", "lower_limit": 0.004, "upper_limit": 0.4}
  }
}
```

### Mode 2 — Parametric source, phase-1 flux normalization (`parametric_flux_from_phase1`)

Two-phase fit. Requires a `reconstruction` block (same phase-1 pixelized source reconstruction as mode 3).

- Phase 1: pixelized source reconstruction on velocity-averaged visibilities
- Phase 2: parametric `KinMS` / `GalPak` disk with free size and kinematic parameters
- Intensity: `intensity` is locked to the velocity-integrated flux of the phase-1 SB map (not fitted). By default that sum uses only pixels with reconstruction SNR ≥ `flux_snr_threshold` (see [Phase-1 flux SNR cut](#phase-1-flux-snr-cut) below)
- Example: `settings/runners/kinms_mock_parametric_flux.json`

```json
{
  "model_name": "KinMS",
  "normalization_mode": "parametric_flux_from_phase1",
  "reconstruction": {
    "mesh_type": "delaunay",
    "regularization": {"type": "constant_split", "prior_type": "fixed", "value": 1e5},
    "flux_snr_threshold": 0.5
  },
  "priors": {
    "effective_radius": {"type": "LogUniformPrior", "lower_limit": 0.03, "upper_limit": 0.07}
  }
}
```

Do not include an `intensity` prior; it is fixed automatically after phase 1.

### Mode 3 — Pixelized source (`pixelized`)

Two-phase fit. Requires a `reconstruction` block.

- Phase 1: pixelized source reconstruction on velocity-averaged visibilities
- Phase 2: `KinMSPixelized` — phase-1 SB map converted to KinMS `inClouds` / `flux_clouds`; total flux passed as fixed `intFlux`
- Only kinematic parameters and lens mass are fitted in phase 2 (no `intensity` or `effective_radius`)
- **Sky-plane `inClouds`:** phase-1 maps are already projected morphologies. Inclination/PA are applied only to LOS velocities (`vLOS_clouds`), not by re-projecting cloud positions (which would double-count \(\cos i\) and brighten peaks)
- **Cube axes:** KinMS returns `(x, y, v)`; LensKin stores `(v, y, x)` for autolens. Source centre placement uses `phaseCent=[x, y]` with optional `flip_kinms_y_before_lensing` for y-sense matching
- Example: `settings/runners/SPT0538_CO9-8_pixelized.json`

```json
{
  "model_name": "KinMSPixelized",
  "normalization_mode": "pixelized",
  "reconstruction": {
    "mesh_type": "delaunay",
    "regularization": {"type": "constant_split", "prior_type": "fixed", "value": 1e5},
    "clouds_per_pixel": 1024,
    "disk_scale_height_kpc": 0.1,
    "max_radius": null,
    "sb_input_units": "jy_per_pixel_per_channel",
    "flux_snr_threshold": 0.5
  },
  "priors": {
    "maximum_velocity": {"type": "UniformPrior", "lower_limit": 200.0, "upper_limit": 400.0}
  }
}
```

#### Cloud sampling (`KinMSPixelized`)

Phase-1 / truth SB maps on the KinMS grid are turned into cloudlets by `in_clouds_and_flux_from_sb_map`:

1. Optionally mask the map with `flux_snr_threshold` (default `0.5`) so noise pixels do not enter the cloudlets or total `intFlux`
2. Convert the map to velocity-integrated flux (`Jy km/s/pixel`) using `sb_input_units`
3. Spawn `clouds_per_pixel` clouds per lit pixel (uniform jitter in the pixel; optional exponential `z` scale height)
4. Pass relative `flux_clouds` weights plus total `intFlux` into KinMS (`cleanOut=True` → `cube.sum() * dv == intFlux`)

| Setting | Default | Notes |
|---------|---------|-------|
| `reconstruction.clouds_per_pixel` | `1024` | Higher density reduces spatial sampling speckles; total flux is conserved at any density |
| `reconstruction.disk_scale_height_kpc` | `0.1` | Converted to source-plane arcsec at `redshift_source` |
| `reconstruction.max_radius` | `null` | Optional arcsec clip of clouds about the phase centre; `null` = no clip |
| `reconstruction.sb_input_units` | `"jy_per_pixel_per_channel"` | Or `"jy_kms_per_pixel"` if the map is already moment-0 |
| `reconstruction.flux_snr_threshold` | `0.5` | See [Phase-1 flux SNR cut](#phase-1-flux-snr-cut) |

**Total flux** through the cloud step is conserved exactly. **Spatial** residuals vs a smooth truth map are a cloudlet / re-binning floor (typically \(\lesssim 1\sigma_{\mathrm{dirty}}\) at 1024 clouds on the wide-velocity mock).

For backward compatibility, `model_name: "KinMSPixelized"` without an explicit `normalization_mode` is treated as `"pixelized"`.

### GalPaK kinematic backend (modes 1 and 2)

Set `"model_name": "GalPak"` to build the source cube with GalPaK's `DiskModel._create_cube` instead of KinMS. Supported modes:

| Mode | Support |
|------|---------|
| `"parametric"` | Yes (existing) |
| `"parametric_flux_from_phase1"` | Yes — phase-1 SB still sets total flux |
| `"pixelized"` | Not yet |

**Flux units:** KinMS `intensity` is Jy km/s (`cube.sum() * dv`). GalPaK `intensity` normalizes so `cube.sum()` equals the flux parameter. Mode 2 therefore converts phase-1 `intFlux` as `intensity = intFlux / z_step_kms` before fixing the prior. The same `flux_snr_threshold` cut applies before that conversion.

Example runners:

- Mode 1: `settings/runners/galpak_mock_unlensed_parametric.json`
- Mode 2: `settings/runners/galpak_mock_unlensed_parametric_flux.json`
- Production-style mode 1: `settings/runners/SPT0538_CO9-8.json`

## Turning lensing off

Set a first-class flag in the runner JSON:

```json
"lensing": {
  "enabled": false
}
```

When `lensing.enabled` is `false`:

- The pipeline uses a fixed **identity** mass model (θ_E = 0, zero shear/multipoles) so the existing AutoLens tracer / regridding path still runs.
- `free_lens_centre` is forced off (an explicit `free_lens_centre: true` raises an error).
- Phase-1 mesh is forced to **`rectangular_uniform`** (Delaunay / density-adapt meshes are overridden — adaptive source meshes are not meaningful without magnification).
- Regularization may be **`constant`** or **`adapt`**. Delaunay-only `constant_split` / `adapt_split` are remapped to `constant` / `adapt`.

Missing `lensing.enabled` defaults to `true` so existing lensed runners are unchanged. Orientation keys (`flip_kinms_y_before_lensing`, etc.) are unchanged.

Unlensed three-mode runners (shared mock under `data/kinms_mock_unlensed/`):

| Mode | Settings |
|------|----------|
| parametric | `settings/runners/kinms_mock_unlensed_parametric.json` |
| flux-from-phase1 | `settings/runners/kinms_mock_unlensed_parametric_flux.json` |
| pixelized | `settings/runners/kinms_mock_unlensed_pixelized.json` |

```bash
# Generate / diagnose the shared unlensed mock
python scripts/generate_unlensed_mock_and_diagnose.py

# Short smoke test of all three modes (phase-1 + truth likelihood; no Dynesty)
# Writes dirty data/model/residual plots under output/kinms_mock_unlensed_smoke/plots/
# Regenerates the mock with Gaussian noise by default (needed for Autolens pixelizations)
python scripts/smoke_unlensed_three_modes.py

# Exact forward-model check without noise (diagnostic only)
python scripts/smoke_unlensed_three_modes.py --no-noise
```

Per-mode plot layout:

| Path | Content |
|------|---------|
| `plots/parametric/dirty_mom0_fit.png` | Mode 1 dirty mom0 data / model / residual |
| `plots/parametric_flux/phase1/` | Mode 2 phase-1 dirty fit + SB map |
| `plots/parametric_flux/phase2/dirty_mom0_fit.png` | Mode 2 kinematic dirty mom0 triplet |
| `plots/pixelized/phase1/` | Mode 3 phase-1 dirty fit + SB map |
| `plots/pixelized/phase2/dirty_mom0_fit.png` | Mode 3 kinematic dirty mom0 triplet |

## Phase-1 pixelized reconstruction

Modes 2 and 3 run a preliminary phase-1 fit before KinMS / GalPaK. Phase 1 builds a **moment-0** `Interferometer` dataset (complex mean over spectral channels), reconstructs the lensed source on the source plane, and passes the SB map (and optionally lens centre / flux) to phase 2.

### Phase-1 flux SNR cut

Summing the raw phase-1 reconstruction into a total flux (mode 2) or cloudlet weights (mode 3) includes noise: with `use_positive_only_solver: true`, faint positive noise biases the locked flux high; without it, negative bowls can bias it low.

By default LensKin therefore masks the source map with Autolens per-pixel reconstruction noise before locking flux:

```text
SNR = SB / inversion.reconstruction_noise_map
keep pixels with SNR ≥ flux_snr_threshold   (default 0.5)
```

| Setting | Default | Notes |
|---------|---------|-------|
| `reconstruction.flux_snr_threshold` | `0.5` | Applied in `runner_pixelized` for modes 2 and 3. Phase-1 diagnostic plots still show the **unmasked** SB map |
| | | Set to `null`, `false`, or `≤ 0` to disable and sum the full map |

```json
"reconstruction": {
  "use_positive_only_solver": true,
  "flux_snr_threshold": 0.5
}
```

On the GalPaK self-consistent unlensed mock, `positive_only` + `flux_snr_threshold: 0.5` recovered locked intensity to within ~2% of truth; with no cut the same run was ~34% too bright.

### Image-plane grid (Nyquist default)

The transformer / dirty-image grid is intentionally **coarse** (typically
`n_pixels: 40`). Refining that grid past the interferometer resolution does
not improve the fit: the longest baseline already sets the Nyquist limit.

**Default pixel scale** (when `"pixel_scale": "nyquist"` or when UV data are
loaded and no numeric scale is set):

\[
\Delta\theta \;=\; \tfrac{1}{2}\,\frac{\lambda}{b_{\max}}
\;=\; \frac{0.5}{u_{\max}}\quad\text{(radians)}
\]

where \(u_{\max}=\max\sqrt{u^2+v^2}\) is taken from the loaded
`uv_wavelengths` product (baselines in units of \(\lambda\)). The value stored
in settings is in **arcsec**.

```json
"n_pixels": 40,
"pixel_scale": "nyquist"
```

With that default:

| Quantity | Behaviour |
|----------|-----------|
| `n_pixels` | Kept as set (e.g. 40²) |
| `pixel_scale` | `0.5 × λ/b_max` in arcsec |
| Field of view | `n_pixels × pixel_scale` (declared `real_space_width` is **overridden**) |

For the `kinms_mock` ALMA UV coverage this is ≈ **0.157″/pixel** and FOV ≈
**6.27″** (vs the older fixed `5″/40 = 0.125″`).

**Overrides**

| Setting | Effect |
|---------|--------|
| `"pixel_scale": 0.1` (numeric) | Use that arcsec scale; FOV from `real_space_width` if set |
| `"pixel_scale_mode": "fov"` | Force legacy `real_space_width / n_pixels` even when UV is present |

Phase-1 `reconstruction.mask_n_pixels` may still **oversample the same FOV**
for the Autolens inversion (e.g. 128² over ~6″). That is separate from the
40² transformer grid. Source morphology is controlled by the **source mesh /
regularization**, not by refining the image-plane DFT/NUFFT grid.

This is resolved automatically when UV data are loaded (`run_fit`, phase 1,
mock generators): numeric `pixel_scale` and `real_space_width` are written
back into the in-memory settings for the rest of the run.

### Recommended settings (interferometer mock / ALMA cubes)

Validated on `kinms_mock` data:

| Setting | Recommended value |
|---------|-------------------|
| `mesh_type` | `"delaunay"` |
| `image_mesh_shape` | `[30, 30]` |
| `delaunay_edge_pixels` | `30` |
| `regularization.type` | `"adapt_split"` (default for Delaunay; less edge-pixel noise than `constant_split`) |
| `regularization` | free `inner_coefficient` (log-uniform); fixed `outer_coefficient` (~30) and `signal_scale` (~3) |
| `fix_lens` | `false` — free lens centre works well with fixed or optimised λ |
| `search.use_jax_gradient` | `false` (Delaunay triangulation is not JAX-differentiable) |

Example `reconstruction` block:

```json
"reconstruction": {
  "fix_lens": false,
  "mesh_type": "delaunay",
  "mask_n_pixels": 128,
  "mask_radius": 3.0,
  "image_mesh_shape": [30, 30],
  "delaunay_edge_pixels": 30,
  "clouds_per_pixel": 1024,
  "disk_scale_height_kpc": 0.1,
  "max_radius": null,
  "sb_input_units": "jy_per_pixel_per_channel",
  "flux_snr_threshold": 0.5,
  "moment0": {
    "sigma_mode": "independent_mean",
    "sigma_scale": 1.0,
    "uv_mode": "average"
  },
  "centre_prior": {"lower_limit": -0.5, "upper_limit": 0.5},
  "regularization": {
    "type": "adapt_split",
    "prior_type": "log_uniform",
    "inner_coefficient": {"lower_limit": 0.01, "upper_limit": 100.0},
    "outer_coefficient": {"prior_type": "fixed", "value": 30.0},
    "signal_scale": {"prior_type": "fixed", "value": 3.0}
  },
  "search": {
    "path_prefix": "kinms_mock_pixelized",
    "name": "reconstruction",
    "optimizer": "LBFGS",
    "use_jax_gradient": false,
    "number_of_cores": "auto",
    "maxiter": 1000,
    "figure_of_merit": "log_likelihood_with_regularization"
  }
}
```

Still available: `"type": "constant_split"` with a fixed or log-uniform `coefficient` if you want uniform smoothing.

### Mesh and regularization pairing

| Mesh | Regularization types |
|------|----------------------|
| `rectangular_adapt_density`, `rectangular_uniform`, `rectangular_adapt_image` | `constant`, `adapt` |
| `delaunay` | `constant_split`, `adapt_split` |

`adapt` / `adapt_split` are brightness-weighted (Nightingale+2018): higher smoothing in faint pixels (`outer_coefficient`), lower in bright pixels (`inner_coefficient`), with `signal_scale` controlling the transition. They need a dirty-image adapt map (built automatically from the phase-1 dataset). Coefficient scales differ strongly from a single `constant`/`constant_split` λ (~`1e5` for interferometer data).

**Lensed or unlensed:** both paths accept brightness-weighted regularization. Prefer rectangular + `adapt` when lensing is off (forced rectangular mesh). Prefer Delaunay + `adapt_split` for lensed production fits (same mesh pairing as `constant_split`).

Example (unlensed rectangular + Adapt):

```json
"mesh_type": "rectangular_uniform",
"mesh_shape": [20, 20],
"transformer": "dft",
"use_jax": false,
"regularization": {
  "type": "adapt",
  "prior_type": "log_uniform",
  "inner_coefficient": {"lower_limit": 0.01, "upper_limit": 100.0},
  "outer_coefficient": {"prior_type": "fixed", "value": 50.0},
  "signal_scale": {"prior_type": "fixed", "value": 3.0}
}
```

Example (lensed Delaunay + AdaptSplit):

```json
"mesh_type": "delaunay",
"image_mesh_shape": [30, 30],
"delaunay_edge_pixels": 30,
"regularization": {
  "type": "adapt_split",
  "prior_type": "log_uniform",
  "inner_coefficient": {"lower_limit": 0.01, "upper_limit": 100.0},
  "outer_coefficient": {"prior_type": "fixed", "value": 50.0},
  "signal_scale": {"prior_type": "fixed", "value": 3.0}
},
"search": {
  "figure_of_merit": "log_likelihood_with_regularization"
}
```

Tuned starting point from the unlensed pixelized mock: free `inner_coefficient` near ~1, fixed `outer_coefficient=50`, `signal_scale=3`. On large UV datasets with rectangular Adapt, prefer `transformer: "dft"` over NUFFT (NUFFT + Adapt can be memory-heavy). Neither `constant`/`adapt` nor their `*_split` variants enforce non-negative source pixels — set `"use_positive_only_solver": true` under `reconstruction` for Autolens' positive-only linear solver (recommended for Adapt; default in code is still `false` if omitted).

Example runners:

| Case | Settings |
|------|----------|
| Unlensed pixelized + Adapt | `settings/runners/kinms_mock_unlensed_pixelized.json` |
| Unlensed flux-from-phase1 + Adapt | `settings/runners/kinms_mock_unlensed_parametric_flux.json` |
| Lensed pixelized + AdaptSplit (Delaunay) | `settings/runners/kinms_mock_lensed_pixelized_adapt.json` |
| Lensed pixelized + `adapt_split` (Delaunay, default) | `settings/runners/kinms_mock_lensed_pixelized.json` |

### JAX gradients (`use_jax_gradient`)

Setting `"use_jax_gradient": true` in `reconstruction.search` selects `JAXLBFGS`, which passes analytical JAX gradients (`fitness.grad`) to scipy's L-BFGS-B. That avoids finite-difference stepping with a single `eps`, which is problematic when lens centres (~0.2″) and regularization coefficients (~`1e5`) are optimised together.

LensKin also sets `"use_jax": true` by default on the phase-1 dataset (JAX sparse UV operator). Both flags apply **only to phase-1 LBFGS**; phase-2 Nautilus uses a separate code path.

#### When JAX gradients work

| Requirement | Why |
|-------------|-----|
| **Rectangular mesh** (`rectangular_adapt_density`, `rectangular_uniform`, `rectangular_adapt_image`) | Mapper/interpolator is JAX-differentiable end-to-end |
| **`use_jax: true`** on the dataset (default) | Sparse operator and analysis run in JAX |
| **Phase-1 LBFGS** with mixed-scale free parameters | Main benefit: no shared `eps` across arcsec and coefficient scales |
| **`constant` or `adapt`** regularization on rectangular meshes | Standard schemes; no Delaunay triangulation callback |

Example (rectangular mesh, optimising λ and lens centre):

```json
"mesh_type": "rectangular_adapt_density",
"regularization": {
  "type": "constant",
  "prior_type": "log_uniform",
  "lower_limit": 1e5,
  "upper_limit": 1e7
},
"search": {
  "optimizer": "LBFGS",
  "use_jax_gradient": true
}
```

#### When JAX gradients are disabled or unavailable

| Condition | Reason |
|-----------|--------|
| **`mesh_type: delaunay`** | Triangulation uses `jax.pure_callback`, which has no JVP; LensKin **auto-disables** `use_jax_gradient` and `use_jax` |
| **`constant_split` / `adapt_split`** on Delaunay | Same non-differentiable triangulation (current validated mock setup) |
| **Phase-2 Nautilus** | `use_jax_gradient` does not apply |

#### Practical summary

| Setup | Use JAX gradient? |
|-------|-------------------|
| **Delaunay + `constant_split` @ `1e5`** (validated mock) | **No** — keep `use_jax_gradient: false` |
| **Rectangular + `constant`**, optimising λ and lens centre | **Yes** — primary use case |
| **Rectangular + `adapt`**, optimising reg params and centre | **Yes** (less tested than `constant`) |
| **Fixed λ, only lens centre free** on rectangular | Optional — finite-difference LBFGS is often sufficient |

#### What JAX gradients fix (and do not fix)

**Fix:** scipy's single `eps` across very different parameter scales during LBFGS.

**Do not fix:**

- `log_evidence` Cholesky failures during optimisation — set `search.figure_of_merit` to `"log_likelihood_with_regularization"` if needed (see diagnostics note below)
- Delaunay mesh limitations — use scipy LBFGS with `use_jax_gradient: false`
- Poor LBFGS landscapes — consider Nautilus on regularization parameters instead

### Phase-1 diagnostics

Test phase-1 in isolation (fixed lens from `lens_mass_model`):

```bash
python scripts/test_phase1_pixelization.py \
  --settings settings/runners/kinms_mock_pixelized.json

# Pipeline LBFGS path (same as run_fit.py phase 1)
python scripts/test_phase1_pixelization.py \
  --settings settings/runners/kinms_mock_pixelized.json \
  --mode pipeline
```

Scan regularization figures of merit vs coefficient:

```bash
python scripts/profile_phase1_regularization.py \
  --settings settings/runners/kinms_mock_pixelized.json --mode scan
```

Compare moment-0 and single-channel noise:

```bash
python scripts/check_moment0_noise.py \
  --settings settings/runners/kinms_mock_pixelized.json
```

Phase-1 LBFGS maximizes `figure_of_merit` from the analysis class. For Delaunay + `AdaptSplit` optimisation, set `search.figure_of_merit` to `"log_likelihood_with_regularization"` if `log_evidence` Cholesky factors fail. With `constant_split` and fixed λ, the default evidence-based metric is usually stable.

Phase-1 reconstruction uses `lens_mass_model` for the lens. With `"fix_lens": true`, only regularization is free in phase 1. With `"fix_lens": false`, the lens centre is also fitted (other mass parameters remain fixed from `lens_mass_model`).

Mock validation settings:

- `settings/runners/kinms_mock_pixelized.json` — mode 3 (`KinMSPixelized`)
- `settings/runners/kinms_mock_parametric_flux.json` — mode 2 (`parametric_flux_from_phase1`)
- `settings/runners/kinms_mock_lensed_pixelized.json` — lensed mock + pixelized diagnostics
- `settings/runners/kinms_mock_lensed_pixelized_widevel.json` — same with padded spectral axis (avoids \(v\sin i\) truncation)

Submit to Slurm:

```bash
bash scripts/slurm/submit_kinms_mock_pixelized.sh
bash scripts/slurm/submit_kinms_mock_parametric_flux.sh
```

## Mock validation (KinMS self-mocks)

Self-mocks reuse template ALMA UV coverage and `sigma_statwt` noise maps from an existing dataprep product, replace visibilities with a KinMS → lens → NUFFT model, and score truth (or phase-1) forward models.

### Lensed parametric mock

```bash
python scripts/generate_lensed_mock_and_diagnose.py \
  --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json
```

Ceiling check: lensing the frozen source cube and dirty-imaging should give χ² ≈ 0 on a noiseless mock (`--no-noise`). Production / phase-1 mocks should keep the default noise injection.

### Lensed pixelized-SB truth diagnostics

Fixes the SB map to the truth channel-mean cube, runs `KinMSPixelized` at truth kinematics, and writes source-plane + dirty residual diagnostics:

```bash
# Generate mock with noise (default; required for Autolens pixelizations) + diagnose
python scripts/generate_lensed_mock_pixelized_and_diagnose.py \
  --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json

# Re-run diagnostics only
python scripts/generate_lensed_mock_pixelized_and_diagnose.py \
  --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json \
  --skip-generate

# Optional: noiseless mock for exact forward-model checks only
python scripts/generate_lensed_mock_pixelized_and_diagnose.py \
  --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json \
  --no-noise
```

Useful outputs under `output/.../lensed_pixelized_truth_diagnostics/`:

| File | Content |
|------|---------|
| `pixelized_source_mom0.png` | Source-plane KinMS cube mom0 (should be a focussed disk, not a ring) |
| `source_plane_truth_vs_pixelized.png` | Truth vs cloudlet-sampled source mom0 |
| `dirty_mom0_*_over_sigma.png` | Dirty mom0 residuals in units of Monte-Carlo dirty-image σ |
| `channel_residuals_*_over_sigma.png` | Per-channel residuals / σ (±5σ colour bar) |

Visibility σ always comes from the template `sigma_statwt` (χ² weights). **Mocks inject Gaussian noise by default** (`N(0, σ)`); use `--no-noise` only for exact forward-model diagnostics — Autolens pixelized source solutions struggle without a noise floor. Residual `/σ` maps use a Monte-Carlo dirty-image noise cube from that same σ.

For high-\(v\sin i\) disks, set `mock_pad_channels_each_side` (widevel settings use `8`) so the spectral window is wider than the projected rotation; otherwise edge channels are truncated and pixelized SB underfills the line wings.

### Phase-1 grid / regularization trials

Score how phase-1 source-grid size and regularization set the residual floor relative to a frozen-SB baseline:

```bash
python scripts/trial_source_grid_regularization.py \
  --settings settings/runners/kinms_mock_lensed_pixelized_widevel.json \
  --source-n-pixels 128,256,512 \
  --reg 1e3,1e4,1e5,1e6 \
  --mesh-shapes 20x20,30x30 \
  --mesh rectangular
```

Writes `output/.../grid_reg_trials/trial_summary.csv`, heatmaps, and per-trial dirty `/σ` residual plots. Prefer `--mesh rectangular` for scans; Delaunay matches production but is heavier.

## Primary beam correction

LensKin supports an optional Gaussian primary-beam (PB) attenuation in the forward model, following Stacey et al. (2024, A&A, [arXiv:2403.04850](https://arxiv.org/abs/2403.04850)), §3.2. The PB is modelled as a Gaussian with half-power beam width HPBW = 1.13 λ/D (where D = 12 m for ALMA). The wavelength is computed from the mean of the input channel frequencies — no manual `wavelength_m` entry is needed.

When enabled, the PB map is applied as a diagonal image-plane operator: each real-space channel image is multiplied by the PB before the NUFFT (phase 2) or before the Autolens transformer (phase 1). Visibility noise maps are not modified. Dirty images remain in attenuated (observed) units.

### Settings

Add a `primary_beam` block to the runner settings JSON:

```json
"primary_beam": {
  "enabled": true,
  "dish_diameter_m": 12.0,
  "pointing_arcsec": [0.0, 0.0]
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable/disable PB correction |
| `dish_diameter_m` | `12.0` | Antenna diameter in metres |
| `pointing_arcsec` | `[0.0, 0.0]` | Pointing centre offset `[y, x]` in arcsec (default: phase centre) |

When `enabled` is `false` or the block is absent, the pipeline behaves identically to previous versions. At Band 7 (~350 GHz) with a 5″ field, the PB attenuation at the field edge is only a few percent; the correction matters more for wide-field or lower-frequency observations.

## Corner plots

After a completed Nautilus search:

```bash
python scripts/make_cornerplot.py /path/to/run_hash_directory
```

Writes `cornerplot.png` in the run directory, plotting free parameters only.

## Exporting uv FITS products (CASA)

Inside a CASA environment:

```bash
python scripts/run_dataprep.py --settings settings/dataprep/SPT0538_CO9-8.json
```

Or:

```bash
casa -c scripts/run_dataprep.py --settings settings/dataprep/SPT0538_CO9-8.json
```

## Dependencies

LOCAL

python == 3.8

pip install scipy == 1.10.1

pip install numpy == 1.24.3

pip install autofit == 2024.5.16.0

pip install autolens == 2024.5.16.0

pip install pynufft == 2024.1.2

pip install galpak == 1.34.0

pip install kinms == 3.0.7

COSMA

python == 3.9
