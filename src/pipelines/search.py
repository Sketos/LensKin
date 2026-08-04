import logging
import os

import autofit as af

logger = logging.getLogger(__name__)


def resolve_number_of_cores(search_cfg):
    """
    Resolve the worker count for Nautilus / Autofit multiprocessing.

    Priority:
      1. ``LENSKIN_CORES`` environment variable (``auto`` or an integer)
      2. ``search.number_of_cores`` in settings (``auto`` or an integer)
      3. default ``auto`` = all available CPUs minus one
    """
    env_cores = os.environ.get("LENSKIN_CORES")
    if env_cores is not None and str(env_cores).strip() != "":
        cores = str(env_cores).strip().lower()
    else:
        cores = search_cfg.get("number_of_cores", "auto")
        if cores is not None:
            cores = str(cores).strip().lower()

    if cores in (None, "", "auto"):
        available = os.cpu_count() or 1
        return max(1, available - 1)
    return max(1, int(cores))


def resolve_n_batch(search_cfg, number_of_cores):
    """
    Batch size for Nautilus likelihood evaluations.

    When not set explicitly, choose a default that is a multiple of the pool
    size, matching Nautilus's own default logic.
    """
    if "n_batch" in search_cfg:
        return search_cfg["n_batch"]
    base = 100
    return (base // number_of_cores + (base % number_of_cores != 0)) * number_of_cores


def build_search_from_settings(search_cfg):
    """
    Build a Nautilus nested sampler from a settings ``search`` block.

    Accepts ``n_live`` (preferred) or legacy ``nlive``. Dynesty-only keys such as
    ``sample`` are ignored.

    For non-JAX analyses (e.g. the KinMS Phase-2 fit), Nautilus parallelises
    likelihood calls across ``number_of_cores`` via a multiprocessing pool. For
    JAX analyses (e.g. the pixelized reconstruction), PyAutoFit disables that
    pool and instead batches evaluations with JAX ``vmap``; ``number_of_cores``
    does not apply there.
    """
    kwargs = {}

    n_live = search_cfg.get("n_live", search_cfg.get("nlive"))
    if n_live is not None:
        kwargs["n_live"] = n_live

    number_of_cores = resolve_number_of_cores(search_cfg)
    kwargs["n_batch"] = resolve_n_batch(search_cfg, number_of_cores)

    for key in (
        "iterations_per_quick_update",
        "iterations_per_full_update",
        "unique_tag",
        "use_jax_vmap",
        "enlarge_per_dim",
        "split_threshold",
    ):
        if key in search_cfg:
            kwargs[key] = search_cfg[key]

    logger.info(
        "Nautilus search: n_live=%s, number_of_cores=%s, n_batch=%s",
        kwargs.get("n_live"),
        number_of_cores,
        kwargs["n_batch"],
    )

    return af.Nautilus(
        path_prefix=search_cfg["path_prefix"],
        name=search_cfg["name"],
        number_of_cores=number_of_cores,
        **kwargs,
    )


_OPTIMIZER_CLASSES = {
    "LBFGS": af.LBFGS,
    "JAXLBFGS": None,  # resolved lazily below
    "BFGS": af.BFGS,
}


def _optimizer_class(name):
    if name == "JAXLBFGS":
        from src.pipelines.jax_lbfgs import JAXLBFGS

        return JAXLBFGS
    return _OPTIMIZER_CLASSES[name]


def build_optimizer_from_settings(search_cfg, mesh_type=None):
    """
    Build a maximum-likelihood optimizer from a settings ``search`` block.

    Supported optimizers: ``LBFGS`` (default), ``JAXLBFGS``, ``BFGS``.

    ``use_jax_gradient: true`` selects ``JAXLBFGS``, which passes analytical
    JAX gradients to scipy and avoids finite-difference ``eps`` stepping.

    Delaunay meshes use ``scipy.spatial.Delaunay`` via a non-differentiable
    JAX callback, so ``use_jax_gradient`` is ignored and standard ``LBFGS`` is
    used when ``mesh_type`` is ``delaunay``.

    For standard ``LBFGS``, ``eps`` (scipy finite-difference step, default
    ``1e-8`` from ``config/non_linear/optimize.yaml``) applies to every
    parameter in physical units. Values around ``10`` were tuned for
    regularization-only fits (~1e5) but break lens-centre parameters (~0.2).
    """
    search_cfg = dict(search_cfg)
    use_jax_gradient = search_cfg.get("use_jax_gradient", False)
    if mesh_type == "delaunay" and use_jax_gradient:
        logger.warning(
            "Delaunay mesh triangulation is not JAX-differentiable; "
            "disabling use_jax_gradient and using scipy LBFGS."
        )
        use_jax_gradient = False
        search_cfg["use_jax_gradient"] = False

    optimizer_name = search_cfg.get("optimizer", "LBFGS").upper()
    if use_jax_gradient and optimizer_name == "LBFGS":
        optimizer_name = "JAXLBFGS"

    try:
        optimizer_cls = _optimizer_class(optimizer_name)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Choose from {sorted(k for k in _OPTIMIZER_CLASSES if k != 'JAXLBFGS')} "
            "or set use_jax_gradient=true."
        ) from exc

    number_of_cores = resolve_number_of_cores(search_cfg)

    kwargs = {}
    for key in (
        "maxiter",
        "maxfun",
        "ftol",
        "gtol",
        "eps",
        "maxls",
        "visualize",
        "iterations_per_quick_update",
        "iterations_per_full_update",
        "unique_tag",
    ):
        if key in search_cfg:
            kwargs[key] = search_cfg[key]

    if optimizer_name == "LBFGS" and "eps" in kwargs:
        logger.warning(
            "LBFGS eps=%s applies in physical units to all free parameters. "
            "For mixed arcsec + regularization scales, prefer "
            "use_jax_gradient=true or prior_type=fixed for regularization.",
            kwargs["eps"],
        )
    elif optimizer_name == "JAXLBFGS" and "eps" in kwargs:
        logger.info(
            "JAXLBFGS uses analytical gradients; eps=%s is unused for jac.",
            kwargs["eps"],
        )

    logger.info(
        "%s optimization: number_of_cores=%s, kwargs=%s",
        optimizer_name,
        number_of_cores,
        kwargs or "(config defaults from config/non_linear/optimize.yaml)",
    )

    return optimizer_cls(
        path_prefix=search_cfg["path_prefix"],
        name=search_cfg["name"],
        number_of_cores=number_of_cores,
        **kwargs,
    )
