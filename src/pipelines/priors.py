import autofit as af
from autofit.mapper.prior.tuple_prior import TuplePrior

# Fixed order: Autofit TuplePrior slot 0 = centre_0 (x), slot 1 = centre_1 (y).
CENTRE_PRIOR_KEYS = ("centre_0", "centre_1")


def prior_from_cfg(prior_cfg):
    if prior_cfg["type"] == "UniformPrior":
        return af.UniformPrior(
            lower_limit=prior_cfg["lower_limit"],
            upper_limit=prior_cfg["upper_limit"],
        )
    if prior_cfg["type"] == "LogUniformPrior":
        return af.LogUniformPrior(
            lower_limit=prior_cfg["lower_limit"],
            upper_limit=prior_cfg["upper_limit"],
        )
    if prior_cfg["type"] == "fixed":
        return prior_cfg["value"]
    raise ValueError(f"Unsupported prior type: {prior_cfg['type']}")


def _centre_prior_from_cfg(priors_cfg):
    centre_kwargs = {
        key: prior_from_cfg(priors_cfg[key])
        for key in CENTRE_PRIOR_KEYS
        if key in priors_cfg
    }
    if not centre_kwargs:
        return (0.0, 0.0)
    return TuplePrior(**centre_kwargs)


def source_model_from_profile(profile, priors_cfg):
    """
    Build a source ``af.Model`` using priors from the settings JSON.

    Passing explicit priors avoids loading legacy default prior YAML entries
    (e.g. ``gaussian_limits``) that are incompatible with newer AutoFit builds.
    """
    scalar_kwargs = {
        name: prior_from_cfg(cfg)
        for name, cfg in priors_cfg.items()
        if name not in CENTRE_PRIOR_KEYS
    }
    return af.Model(
        profile,
        centre=_centre_prior_from_cfg(priors_cfg),
        **scalar_kwargs,
    )
