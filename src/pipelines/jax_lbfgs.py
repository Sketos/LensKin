"""
LBFGS wrapper that supplies JAX analytical gradients to scipy.

Autofit's default ``af.LBFGS`` calls ``scipy.optimize.minimize`` with only
``fun=fitness._jit``. Scipy then estimates gradients via finite differences using
``eps`` from ``config/non_linear/optimize.yaml``. That single scalar step size is
ill-suited to mixed-scale parameters (arcsec lens centres ~0.2 and
regularization coefficients ~1e5), which can send regularization to spuriously
high values.

When ``use_jax_gradient`` is enabled in the phase-1 search settings, this class
passes ``jac=fitness.grad`` so ``eps`` is not used for the gradient.
"""

import numpy as np
from autofit.non_linear.search.mle.bfgs.search import AbstractBFGS


class JAXLBFGS(AbstractBFGS):
    """L-BFGS-B with JAX ``grad`` passed to scipy ``minimize``."""

    method = "L-BFGS-B"

    @property
    def _class_config(self):
        """Reuse standard LBFGS config (no separate jaxlbfgs entry required)."""
        return self.config_type["LBFGS"]

    def _fit(self, model, analysis):
        from scipy import optimize

        from autofit.non_linear.fitness import Fitness

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=True,
            store_history=self.should_plot_start_point,
        )

        try:
            search_internal_dict = self.paths.load_search_internal()
            x0 = search_internal_dict["x0"]
            total_iterations = search_internal_dict["total_iterations"]
            self.logger.info(
                "Resuming JAXLBFGS non-linear search (previous samples found)."
            )
        except (FileNotFoundError, TypeError):
            (
                unit_parameter_lists,
                parameter_lists,
                log_posterior_list,
            ) = self.initializer.samples_from_model(
                total_points=1,
                model=model,
                fitness=fitness,
                paths=self.paths,
                n_cores=self.number_of_cores,
            )
            x0 = np.asarray(parameter_lists[0])
            total_iterations = 0
            self.logger.info(
                "Starting new JAXLBFGS non-linear search (no previous samples found)."
            )
            self.plot_start_point(
                parameter_vector=x0,
                model=model,
                analysis=analysis,
            )

        def fun(parameters):
            if analysis._use_jax:
                return float(fitness._jit(parameters))
            return float(fitness(parameters))

        def jac(parameters):
            return np.asarray(fitness.grad(parameters), dtype=float)

        while total_iterations < self.maxiter:
            iterations_remaining = self.maxiter - total_iterations
            iterations = self._steps_until_full_update(iterations_remaining)

            if iterations > 0:
                options = dict(self.options)
                options["maxiter"] = iterations

                search_internal = optimize.minimize(
                    fun=fun,
                    x0=x0,
                    jac=jac,
                    method=self.method,
                    options=options,
                    tol=self.tol,
                )

                total_iterations += search_internal.nit

                search_internal.log_posterior_list = -0.5 * fitness(
                    parameters=search_internal.x
                )

                if self.should_plot_start_point:
                    search_internal.parameters_history_list = (
                        fitness.parameters_history_list
                    )
                    search_internal.log_likelihood_history_list = (
                        fitness.log_likelihood_history_list
                    )

                self.paths.save_search_internal(obj=search_internal)
                x0 = search_internal.x

                if search_internal.nit < iterations:
                    return search_internal, fitness

                self.perform_update(
                    model=model,
                    analysis=analysis,
                    during_analysis=True,
                    fitness=fitness,
                    search_internal=search_internal,
                )

        self.logger.info("JAXLBFGS sampling complete.")
        return search_internal, fitness
