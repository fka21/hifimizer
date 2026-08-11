from typing import Optional

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
        min_trials: int = 5,
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
        self.min_trials = max(0, int(min_trials))
        # Directions is a list like ["maximize","minimize",...]
        # used to aggregate multi-objective values into a scalar for detectors.
        self.directions = directions
        self._last_result = (False, [])

    def _to_scalar(self, current_value):
        """Collapse a multi-objective value tuple into one 'higher is better' number."""
        if not isinstance(current_value, (list, tuple)):
            try:
                return float(current_value)
            except (TypeError, ValueError):
                return 0.0

        vals = list(current_value)
        if self.directions and len(self.directions) >= len(vals):
            signs = [
                1 if d == "maximize" else -1 for d in self.directions[: len(vals)]
            ]
        else:
            signs = [1] * len(vals)

        try:
            return sum(s * float(v) for s, v in zip(signs, vals)) / max(1, len(vals))
        except (TypeError, ValueError):
            return 0.0

    def update(self, current_value, trial_number):
        """Feed every detector one trial's score and take a vote."""
        scalar_value = self._to_scalar(current_value)

        # Warm-up: still feed the detectors so their histories are complete,
        # but never let them stop the study on the first few noisy trials.
        for detector in self.detectors.values():
            detector.update(scalar_value, trial_number)

        if trial_number < self.min_trials:
            self._last_result = (False, [])
            return self._last_result

        results = {
            name: detector.has_converged()
            for name, detector in self.detectors.items()
        }

        # Require at least floor(N/2) detectors to agree (2 of 4).
        has_converged = sum(results.values()) >= max(1, len(self.detectors) // 2)
        converged_methods = [name for name, result in results.items() if result]

        self._last_result = (has_converged, converged_methods)
        return self._last_result

    def has_converged(self):
        """Convergence verdict from the most recent :meth:`update`."""
        return self._last_result[0]


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

        return self.has_converged()

    def has_converged(self):
        """Converged once the values have sat inside the band long enough."""
        return self.plateau_count >= self.min_plateau_length


class StatisticalConvergenceDetector:
    def __init__(self, window_size=10, significance_level=0.05):
        self.window_size = window_size
        self.significance_level = significance_level
        self.history = []
        self.converged = False

    def update(self, current_value, trial_number):
        self.history.append(current_value)

        if len(self.history) >= 2 * self.window_size:
            recent_window = self.history[-self.window_size :]
            older_window = self.history[-2 * self.window_size : -self.window_size]

            try:
                # Mann-Whitney U (non-parametric): is the recent window
                # significantly better than the older one?
                _, p_value = stats.mannwhitneyu(
                    recent_window, older_window, alternative="greater"
                )
                # Converged if no significant improvement.
                self.converged = p_value > self.significance_level
            except ValueError:
                # scipy raises when all values are identical -- which is
                # itself about as converged as a study gets.
                self.converged = True
        else:
            self.converged = False

        return self.converged

    def has_converged(self):
        return self.converged


class RelativeImprovementDetector:
    def __init__(self, threshold=0.001, patience=10):
        self.threshold = threshold
        self.patience = patience
        self.history = []
        self.poor_improvement_count = 0

    def update(self, current_value, trial_number):
        self.history.append(current_value)

        if len(self.history) >= 2:
            previous = self.history[-2]
            delta = self.history[-1] - previous

            # abs() on the denominator: weighted scores can be negative, and
            # dividing by a negative previous value flipped the sign of the
            # improvement, so a run of worsening scores read as "improving"
            # and convergence was never detected.
            if previous == 0:
                relative_improvement = float("inf") if delta > 0 else 0.0
            else:
                relative_improvement = delta / abs(previous)

            if relative_improvement < self.threshold:
                self.poor_improvement_count += 1
            else:
                self.poor_improvement_count = 0

        return self.has_converged()

    def has_converged(self):
        return self.poor_improvement_count >= self.patience


class StagnationDetector:
    def __init__(self, patience=10, min_improvement=0.1):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_value = None
        self.stagnation_count = 0

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

        return self.has_converged()

    def has_converged(self):
        return self.stagnation_count >= self.patience