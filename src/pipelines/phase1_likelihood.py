"""
Likelihood term breakdown for phase-1 pixelized interferometer fits.

Autolens exposes several figures of merit for inversion fits:

- ``fit.log_likelihood``: data term only, ``-0.5 * (chi2 + noise_norm)``. Always
  favours lower regularization (more flexible reconstructions).
- ``fit.log_likelihood_with_regularization``: adds the smoothness penalty
  ``regularization_term`` but not Bayesian Occam factors.
- ``fit.figure_of_merit``: ``log_evidence`` when an inversion is present. This
  is what ``AnalysisInterferometer.log_likelihood_function`` returns to LBFGS.

Computing ``log_evidence`` requires Cholesky factors of the regularization
matrix. That can fail numerically for some coefficient values; this module
falls back to ``log_likelihood_with_regularization`` when that happens.
"""


def _as_float(value):
    try:
        return float(value)
    except TypeError:
        return float(getattr(value, "array", value))


def _safe_as_float(getter, default=None):
    try:
        return _as_float(getter())
    except Exception:
        return default


def phase1_likelihood_breakdown(fit):
    """
    Decompose phase-1 fit figures of merit for diagnostics.

    Returns a dict with ``chi_squared``, ``regularization_term`` (if inversion),
    ``log_likelihood_data``, ``log_likelihood_with_regularization``, and
    ``figure_of_merit_log_evidence`` (the quantity LBFGS maximizes when available).
    """
    breakdown = {
        "chi_squared": _as_float(fit.chi_squared),
        "noise_normalization": _as_float(fit.noise_normalization),
        "log_likelihood_data": _as_float(fit.log_likelihood),
        "regularization_term": None,
        "log_likelihood_with_regularization": None,
        "log_det_curvature_reg": None,
        "log_det_regularization": None,
        "figure_of_merit_log_evidence": _as_float(fit.log_likelihood),
        "figure_of_merit_note": None,
    }

    inversion = getattr(fit, "inversion", None)
    if inversion is None:
        return breakdown

    breakdown["regularization_term"] = _safe_as_float(
        lambda: inversion.regularization_term
    )
    breakdown["log_likelihood_with_regularization"] = _safe_as_float(
        lambda: fit.log_likelihood_with_regularization
    )
    breakdown["log_det_curvature_reg"] = _safe_as_float(
        lambda: inversion.log_det_curvature_reg_matrix_term
    )
    breakdown["log_det_regularization"] = _safe_as_float(
        lambda: inversion.log_det_regularization_matrix_term
    )

    log_evidence = _safe_as_float(lambda: fit.figure_of_merit)
    if log_evidence is not None:
        breakdown["figure_of_merit_log_evidence"] = log_evidence
    elif breakdown["log_likelihood_with_regularization"] is not None:
        breakdown["figure_of_merit_log_evidence"] = breakdown[
            "log_likelihood_with_regularization"
        ]
        breakdown["figure_of_merit_note"] = (
            "log_evidence unavailable (log-det Cholesky failed); "
            "using log_likelihood_with_regularization"
        )
    else:
        breakdown["figure_of_merit_note"] = (
            "log_evidence unavailable; using log_likelihood_data"
        )

    return breakdown


def format_likelihood_breakdown(breakdown):
    """Single-line summary for logging."""
    parts = [
        f"chi2={breakdown['chi_squared']:.2f}",
        f"logL_data={breakdown['log_likelihood_data']:.2f}",
    ]
    if breakdown["regularization_term"] is not None:
        parts.extend(
            [
                f"reg_term={breakdown['regularization_term']:.2e}",
                f"logL_reg={breakdown['log_likelihood_with_regularization']:.2f}",
                f"log_evidence={breakdown['figure_of_merit_log_evidence']:.2f}",
            ]
        )
    if breakdown.get("figure_of_merit_note"):
        parts.append(f"({breakdown['figure_of_merit_note']})")
    return ", ".join(parts)
