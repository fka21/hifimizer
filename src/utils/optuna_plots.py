# utils/optuna_plots.py
"""
Parameter-importance plotting that survives the baseline trial.

Why this exists
---------------
Trial 0 runs hifiasm with default parameters and never calls
``trial.suggest_*``, so Optuna records it as COMPLETE with ``params == {}``.
``plot_param_importances`` resolves its search space as the *intersection* of
the search spaces of all completed trials, and intersecting anything with the
empty set yields the empty set -- which is why that one plot came out blank
while ``plot_slice`` / ``plot_contour`` / ``plot_parallel_coordinate`` (which
use the union, or per-trial params) all worked.

We want to keep trial 0 in the study: it is the reference point in the
optimisation history. So instead of removing it, we copy the parameterised
trials into a throwaway in-memory study and plot the importances from that.
"""

import logging
import warnings

import optuna
import optuna.visualization as vis


def _pick_evaluator(n_trials: int):
    """
    fANOVA (Optuna's default) needs a decent number of completed trials before
    its importances mean anything. With a convergence detector that can stop at
    ~15 trials over 7+ parameters, PED-ANOVA is the better-behaved choice.
    """
    if n_trials < 20:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return optuna.importance.PedAnovaImportanceEvaluator()
        except Exception:
            return None
    return None  # None -> Optuna's default evaluator (fANOVA)


def parameterised_trials(study):
    """Completed trials that actually sampled parameters (i.e. not trial 0)."""
    return [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.params
    ]


def _filtered_study(study, trials):
    """An in-memory clone of ``study`` holding only ``trials``."""
    clone = optuna.create_study(directions=study.directions)
    clone.add_trials(trials)
    return clone


def write_param_importances(
    study, out_dir, objectives=None, multi_objective=False
) -> bool:
    """
    Write parameter-importance plot(s) into ``out_dir``.

    For a multi-objective study ``target=None`` is invalid -- Optuna raises and
    asks for an explicit target -- so one plot per objective is produced.

    Returns True if at least one plot was written.
    """
    trials = parameterised_trials(study)

    if len(trials) < 2:
        logging.warning(
            "Parameter importance skipped: only %d completed trial(s) with sampled "
            "parameters (the baseline trial has none). At least 2 are required.",
            len(trials),
        )
        return False

    # A parameter that never varied carries no information and makes fANOVA
    # unhappy, so only the ones that actually moved are plotted.
    varying = [
        name
        for name in sorted({k for t in trials for k in t.params})
        if len({t.params.get(name) for t in trials if name in t.params}) > 1
    ]
    if not varying:
        logging.warning(
            "Parameter importance skipped: no parameter varied across the "
            "%d completed trials.",
            len(trials),
        )
        return False

    clone = _filtered_study(study, trials)
    evaluator = _pick_evaluator(len(trials))

    logging.info(
        "Computing parameter importances from %d parameterised trial(s) "
        "(baseline trial excluded) over %d varying parameter(s): %s",
        len(trials),
        len(varying),
        ", ".join(varying),
    )

    wrote = False
    out_dir.mkdir(parents=True, exist_ok=True)

    if multi_objective and objectives:
        for idx, metric in enumerate(objectives):
            metric_dir = out_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            try:
                fig = vis.plot_param_importances(
                    clone,
                    evaluator=evaluator,
                    params=varying,
                    # default-arg binding: without `i=idx` every lambda would
                    # close over the final value of idx.
                    target=lambda t, i=idx: t.values[i],
                    target_name=metric,
                )
                fig.write_html(metric_dir / "param_importance.html")
                wrote = True
            except Exception as e:
                logging.warning(
                    f"[{metric}] Failed to create param importance plot: {e}"
                )
    else:
        try:
            fig = vis.plot_param_importances(
                clone, evaluator=evaluator, params=varying
            )
            fig.write_html(out_dir / "optuna_param_importance.html")
            wrote = True
        except Exception as e:
            logging.warning(f"Failed to create param importance plot: {e}")

    return wrote