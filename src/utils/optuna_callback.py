import optuna
from typing import Optional
import logging
import numpy as np
from scipy import stats


class MultiCriteriaConvergenceDetector:
    def __init__(
        self,
        stagnation_patience=25,
        min_improvement=1e-4,
        threshold=0.001,
        patience=15,
        plateau_threshold=5e-1,
        min_plateau_length=15,
        window_size=20,
        significance_level=0.05,
        max_trials: Optional[int] = None,
        directions: Optional[list] = None,
    ):
        self.detectors = {
            "stagnation": StagnationDetector(
                patience=stagnation_patience, min_improvement=min_improvement
            ),
            "relative_improvement": RelativeImprovementDetector(
                threshold=threshold, patience=patience
            ),
            "plateau": PlateauDetector(
                plateau_threshold=plateau_threshold,
                min_plateau_length=min_plateau_length,
            ),
            "statistical": StatisticalConvergenceDetector(
                window_size=window_size, significance_level=significance_level
            ),
        }
        self.convergence_votes = {}
        self.max_trials = max_trials
        # Directions is a list like ["maximize","minimize",...]
        # used to aggregate multi-objective values into a scalar for detectors.
        self.directions = directions

    def update(self, current_value, trial_number):
        """Update all detectors and check for convergence"""
        if self.max_trials is not None and trial_number >= self.max_trials:
            self._last_result = (True, ["max_trials_reached"])
            return True, ["max_trials_reached"]

        if trial_number < 5 and not hasattr(self, "_last_result"):
            self._last_result = (False, [])
            return False, []
        else:
            # If current_value is a sequence (multi-objective), aggregate to a scalar
            scalar_value = current_value
            if isinstance(current_value, (list, tuple)):
                vals = list(current_value)
                if self.directions and len(self.directions) >= len(vals):
                    signs = [
                        1 if d == "maximize" else -1
                        for d in self.directions[: len(vals)]
                    ]
                else:
                    signs = [1] * len(vals)
                # Weighted average with signs so higher scalar_value means improvement
                scalar_value = sum(s * v for s, v in zip(signs, vals)) / max(
                    1, len(vals)
                )

            # Ensure scalar_value is numeric
            try:
                float(scalar_value)
            except Exception:
                scalar_value = 0.0

            results = {}
            for name, detector in self.detectors.items():
                results[name] = detector.update(scalar_value, trial_number)

            convergence_votes = sum(results.values())
            total_detectors = len(self.detectors)

            # Lower voting threshold: require at least floor(N/2) detectors
            # to agree (e.g., 2 of 4 -> faster convergence detection).
            has_converged = convergence_votes >= max(1, total_detectors // 2)
            converged_methods = [name for name, result in results.items() if result]

            # Store last convergence result for external query
            self._last_result = (has_converged, converged_methods)

            return has_converged, converged_methods

    def has_converged(self):
        """Return True if convergence detected based on last update or detector states."""
        if hasattr(self, "_last_result"):
            return self._last_result[0]

        votes = sum(getattr(det, "converged", False) for det in self.detectors.values())
        return votes >= 2


class PlateauDetector:
    def __init__(self, plateau_threshold=1e-4, min_plateau_length=8):
        self.plateau_threshold = plateau_threshold
        self.min_plateau_length = min_plateau_length
        self.history = []
        self.plateau_count = 0

    def update(self, current_value, trial_number):
        self.history.append(current_value)

        if len(self.history) >= self.min_plateau_length:
            recent_values = self.history[-self.min_plateau_length :]

            # Check if recent values are within plateau threshold
            value_range = max(recent_values) - min(recent_values)

            if value_range < self.plateau_threshold:
                self.plateau_count += 1
            else:
                self.plateau_count = 0

        # Converged if in plateau for sufficient time
        return self.plateau_count >= self.min_plateau_length


class StatisticalConvergenceDetector:
    def __init__(self, window_size=10, significance_level=0.05):
        self.window_size = window_size
        self.significance_level = significance_level
        self.history = []

    def update(self, current_value, trial_number):
        self.history.append(current_value)

        if len(self.history) >= 2 * self.window_size:
            # Compare recent window with older window
            recent_window = self.history[-self.window_size :]
            older_window = self.history[-2 * self.window_size : -self.window_size]

            # Use Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(
                recent_window, older_window, alternative="greater"
            )

            # Converged if no significant improvement
            return p_value > self.significance_level

        return False


class RelativeImprovementDetector:
    def __init__(self, threshold=0.001, patience=10):
        self.threshold = threshold
        self.patience = patience
        self.history = []
        self.poor_improvement_count = 0

    def update(self, current_value, trial_number):
        self.history.append(current_value)

        if len(self.history) >= 2:
            # Avoid division by zero
            if self.history[-2] == 0:
                relative_improvement = float("inf") if self.history[-1] > 0 else 0.0
            else:
                relative_improvement = (
                    self.history[-1] - self.history[-2]
                ) / self.history[-2]

            if relative_improvement < self.threshold:
                self.poor_improvement_count += 1
            else:
                self.poor_improvement_count = 0

        return self.poor_improvement_count >= self.patience


class StagnationDetector:
    def __init__(self, patience=10, min_improvement=0.1):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_value = None
        self.stagnation_count = 0
        self.convergence_history = []

    def update(self, current_value, trial_number):
        """Update convergence tracker with new value"""
        if self.best_value is None:
            self.best_value = current_value
            self.stagnation_count = 0
        elif current_value > self.best_value + self.min_improvement:
            # Significant improvement found
            self.best_value = current_value
            self.stagnation_count = 0
        else:
            # No significant improvement
            self.stagnation_count += 1
            # Update best_value if current is better (but not by min_improvement threshold)
            if current_value > self.best_value:
                self.best_value = current_value

        self.convergence_history.append(
            {
                "trial": trial_number,
                "value": current_value,
                "best_value": self.best_value,
                "stagnation_count": self.stagnation_count,
            }
        )

        return self.has_converged()

    def has_converged(self):
        return self.stagnation_count >= self.patience
