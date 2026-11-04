import logging
import warnings

import optuna


#: Accepted by ``--sampler``. The first is the default.
SAMPLER_CHOICES = ("tpe", "gp", "random")

SAMPLER_DESCRIPTIONS = {
    "tpe": "multivariate TPE (models the hifiasm parameters jointly)",
    "gp": "Gaussian-process Bayesian optimisation",
    "random": "random sampling (a baseline, not an optimiser)",
}


class _ParameterisedTrialsView:
    """
    A read-only view of a study that hides trials without parameters.

    Trial 0 is hifiasm with its own defaults and samples nothing, so its
    parameter set is empty. Optuna's intersection search space is the set of
    parameters *every* completed trial has, which an empty trial reduces to
    nothing -- and a sampler with an empty search space quietly samples every
    parameter at random. Hiding the baseline from the sampler's view of the
    study keeps the model on the trials that actually explored the space.
    Everything else is delegated to the real study.
    """

    def __init__(self, study):
        self._study = study

    def __getattr__(self, name):
        return getattr(self._study, name)

    def get_trials(self, deepcopy=True, states=None):
        return [
            t
            for t in self._study.get_trials(deepcopy=deepcopy, states=states)
            if t.params
        ]

    def _get_trials(self, deepcopy=True, states=None, use_cache=False):
        return [
            t
            for t in self._study._get_trials(
                deepcopy=deepcopy, states=states, use_cache=use_cache
            )
            if t.params
        ]


class BaselineAwareGPSampler(optuna.samplers.GPSampler):
    """
    ``GPSampler`` that fits its model on the parameterised trials only.

    Without this the parameter-free baseline trial collapses the GP's search
    space (see :class:`_ParameterisedTrialsView`) and every trial after it is
    drawn at random. Only the sampler's view of the study changes; trial 0
    stays in the study, in the score history and in the best-trial comparison.
    """

    def infer_relative_search_space(self, study, trial):
        return super().infer_relative_search_space(
            _ParameterisedTrialsView(study), trial
        )

    def sample_relative(self, study, trial, search_space):
        return super().sample_relative(
            _ParameterisedTrialsView(study), trial, search_space
        )


def build_sampler(name, multi_objective, seed):
    """
    The Optuna sampler for ``--sampler``.

    ``tpe`` is multivariate with ``group=True``. Multivariate because hifiasm's
    graph-cleaning parameters interact (``-x``/``-y``/``-s`` most of all), and
    independent TPE models each one alone. ``group=True`` because plain
    multivariate TPE also works on the intersection search space, which the
    parameter-free baseline trial empties, and it then falls back to
    independent sampling with a warning on every trial; the grouped search
    space ignores a trial with no parameters instead. TPE also handles
    multiple objectives, so it serves both modes.

    ``gp`` is single-objective only in the Optuna version hifimizer pins
    (4.2), and ignores pruned trials, so it can keep proposing regions the
    gfastats gate keeps rejecting; it tends to pay off on studies with few,
    mostly completing trials.

    Raises:
        ValueError: for an unknown name, or ``gp`` with multiple objectives.
    """
    name = (name or "tpe").lower()
    if name not in SAMPLER_CHOICES:
        raise ValueError(
            f"Unknown sampler '{name}'; choose one of {', '.join(SAMPLER_CHOICES)}."
        )

    # multivariate/group TPE and GPSampler are flagged experimental by Optuna;
    # the warning says nothing actionable to a hifimizer user.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        if name == "tpe":
            return optuna.samplers.TPESampler(
                seed=seed, multivariate=True, group=True
            )
        if name == "gp":
            if multi_objective:
                raise ValueError(
                    "--sampler gp supports a single objective only in the "
                    "Optuna version hifimizer uses. Use --sampler tpe with "
                    "--multi-objective, or drop --multi-objective."
                )
            return BaselineAwareGPSampler(seed=seed)

    logging.warning(
        "--sampler random: parameter sets are drawn at random and nothing is "
        "learned from earlier trials. This is mostly useful for generating "
        "baseline results to compare an optimiser against -- the best "
        "hyperparameter set found this way is not guaranteed to be anywhere "
        "near the best one available. Use --sampler tpe or gp to optimise."
    )
    return optuna.samplers.RandomSampler(seed=seed)
