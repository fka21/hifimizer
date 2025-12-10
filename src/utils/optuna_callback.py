import optuna
from optuna.terminator import Terminator, EMMREvaluator
from typing import Optional
import logging
import numpy as np
from scipy import stats
import torch

class MultiCriteriaConvergenceDetector:
    def __init__(self, stagnation_patience=15, min_improvement=1e-4, threshold=0.001, patience=15, plateau_threshold=5e-1, min_plateau_length=15, window_size=10, significance_level=0.01, max_trials: Optional[int] = None):
        self.detectors = {
            'stagnation': StagnationDetector(patience=stagnation_patience, min_improvement=min_improvement),
            'relative_improvement': RelativeImprovementDetector(threshold=threshold, patience=patience),
            'plateau': PlateauDetector(plateau_threshold=plateau_threshold, min_plateau_length=min_plateau_length),
            'statistical': StatisticalConvergenceDetector(window_size=window_size, significance_level=significance_level)
        }
        self.convergence_votes = {}
        self.max_trials = max_trials
    
    def update(self, current_value, trial_number):
        """Update all detectors and check for convergence"""
        if self.max_trials is not None and trial_number >= self.max_trials:
            self._last_result = (True, ["max_trials_reached"])
            return True, ["max_trials_reached"]

        if trial_number < 15 and not hasattr(self, "_last_result"):
            self._last_result = (False, [])
            return False, []
        else:
            results = {}
            for name, detector in self.detectors.items():
                results[name] = detector.update(current_value, trial_number)

            convergence_votes = sum(results.values())
            total_detectors = len(self.detectors)

            has_converged = convergence_votes >= (total_detectors // 2 + 1)
            converged_methods = [name for name, result in results.items() if result]

            # Store last convergence result for external query
            self._last_result = (has_converged, converged_methods)
            
            return has_converged, converged_methods

    def has_converged(self):
        """Return True if convergence detected based on last update or detector states."""
        if hasattr(self, "_last_result"):
            return self._last_result[0]

        votes = sum(
            getattr(det, "converged", False)
            for det in self.detectors.values()
        )
        return votes >= (len(self.detectors) // 2 + 1)

class PlateauDetector:
    def __init__(self, plateau_threshold=1e-4, min_plateau_length=8):
        self.plateau_threshold = plateau_threshold
        self.min_plateau_length = min_plateau_length
        self.history = []
        self.plateau_count = 0
    
    def update(self, current_value, trial_number):
        self.history.append(current_value)
        
        if len(self.history) >= self.min_plateau_length:
            recent_values = self.history[-self.min_plateau_length:]
            
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
            recent_window = self.history[-self.window_size:]
            older_window = self.history[-2*self.window_size:-self.window_size]
            
            # Use Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(recent_window, older_window, alternative='less')
            
            # Converged if no significant improvement
            return p_value > self.significance_level
        
        return False


class RelativeImprovementDetector:
    def __init__(self, threshold=0.001, patience=15):
        self.threshold = threshold
        self.patience = patience
        self.history = []
        self.poor_improvement_count = 0
    
    def update(self, current_value, trial_number):
        self.history.append(current_value)
        
        if len(self.history) >= 2:
            relative_improvement = abs(self.history[-1] - self.history[-2]) / abs(self.history[-2])
            
            if relative_improvement < self.threshold:
                self.poor_improvement_count += 1
            else:
                self.poor_improvement_count = 0
        
        return self.poor_improvement_count >= self.patience


class StagnationDetector:
    def __init__(self, patience=15, min_improvement=1e-4):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_value = None
        self.stagnation_count = 0
        self.convergence_history = []
    
    def update(self, current_value, trial_number):
        """Update convergence tracker with new value"""
        if self.best_value is None or current_value < self.best_value - self.min_improvement:
            self.best_value = current_value
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        self.convergence_history.append({
            'trial': trial_number,
            'value': current_value,
            'best_value': self.best_value,
            'stagnation_count': self.stagnation_count
        })
        
        return self.has_converged()
    
    def has_converged(self):
        return self.stagnation_count >= self.patience




class EarlyStoppingCallback:
    """Callback to stop an Optuna study early if no improvement occurs.

    This callback monitors the optimization process and stops the study if no
    improvement is observed in the best value for a specified number of 
    consecutive trials.
    
    DEPRECATED: Use `ConvergenceCallback` instead for better convergence detection.

    Args:
        early_stopping_rounds (int): Number of non-improving trials to wait before stopping.
        direction (str, optional): Optimization direction ("minimize" or "maximize"). 
            Defaults to "minimize".
    """

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize", terminate_flag=None) -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0
        self.terminate_flag = terminate_flag
        
        if direction == "minimize":
            import operator
            self._operator = operator.lt
            self._score = float("inf")
        elif direction == "maximize":
            import operator
            self._operator = operator.gt
            self._score = float("-inf")
        else:
            raise ValueError(f"Invalid direction: {direction}. Use 'minimize' or 'maximize'.")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Check stopping condition and halt study if criteria are met.
        
        Args:
            study (optuna.Study): Current study object.
            trial (optuna.Trial): Current trial object.
        """
        if self.terminate_flag and self.terminate_flag():
            logging.info("EarlyStoppingCallback detected termination request. Stopping study.")
            study.stop()
            return

        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            logging.info("Early stopping triggered. Stopping study.")
            study.stop()
