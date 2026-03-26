"""
Unit tests for hifimizer.

Coverage:
  - argparser.parse_genome_size                (pure Python, real import)
  - AssemblyEvaluator                          (pure methods — no subprocess)
      parse_gfastats_output, parse_busco_results, parse_sniffles_vcf,
      calculate_weighted_sum, analyze_metric_contributions,
      _get_minimap2_preset, _get_current_stage
  - MultiCriteriaConvergenceDetector           (pure Python)
  - StagnationDetector, PlateauDetector,
    StatisticalConvergenceDetector,
    RelativeImprovementDetector                (pure Python)
  - ObjectiveBuilder                           (initialisation only)

Run from the repo root:
    pytest tests/test_hifimizer.py -v

The test suite patches SubprocessLogger so no log directories are created and
no real subprocesses are launched.
"""

import json
import sys
import os
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup — add src/ to sys.path so imports mirror the real project layout.
# Adjust if your structure differs.
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Patch SubprocessLogger before any module that uses it is imported, so that
# no log directories are created as a side effect of constructing objects.
# ---------------------------------------------------------------------------
_mock_subprocess_logger = MagicMock()
_mock_subprocess_logger.return_value = MagicMock()

with patch.dict("sys.modules", {"utils.subprocess_logger": MagicMock(SubprocessLogger=_mock_subprocess_logger)}):
    from utils.argparser import parse_genome_size
    from utils.assembly_eval import AssemblyEvaluator
    from utils.optuna_callback import (
        MultiCriteriaConvergenceDetector,
        StagnationDetector,
        PlateauDetector,
        StatisticalConvergenceDetector,
        RelativeImprovementDetector,
    )
    from utils.objective import ObjectiveBuilder


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def evaluator(tmp_path):
    """
    Construct a real AssemblyEvaluator with dummy arguments.
    SubprocessLogger is already globally patched so no logs/ dir is created.
    The busco_backend_cache.json is absent, so _load_backend_cache returns {}.
    weights.json is absent, so _load_weights falls back to built-in defaults.
    """
    with patch("utils.assembly_eval.SubprocessLogger", _mock_subprocess_logger):
        ev = AssemblyEvaluator(
            known_genome_size=3_000,          # 3 000 Mb (3 Gb)
            input_reads=tmp_path / "reads.fastq",
            trial_id=0,
            threads=8,
            logs_dir=tmp_path / "logs",
        )
    return ev


@pytest.fixture
def default_weights():
    return {
        "num_contigs":              -0.8,
        "length_diff":              -1.0,
        "n50":                       1.0,
        "single_copy":               1.0,
        "multi_copy":               -0.7,
        "fragmented":               -0.7,
        "missing":                  -1.0,
        "reads_mapped":              0.8,
        "error_rate":               -1.0,
        "num_sv":                   -0.5,
        "supplementary_alignments": -0.6,
    }


@pytest.fixture
def dummy_metrics():
    """Log-transformed metric values that mirror what the parsing methods return."""
    return {
        "num_contigs":              np.log(101),
        "length_diff":              np.log(1),        # perfect size match → log(0+1)=0
        "n50":                      np.log(1_000_001),
        "single_copy":              np.log(951),
        "multi_copy":               np.log(11),
        "fragmented":               np.log(21),
        "missing":                  np.log(31),
        "reads_mapped":             np.log(90_001),
        "error_rate":               np.log(1.001),
        "num_sv":                   np.log(51),
        "supplementary_alignments": np.log(501),
    }


# ===========================================================================
# 1. Genome-size parser (argparser.parse_genome_size)
# ===========================================================================

class TestParseGenomeSize:
    """Tests for the parse_genome_size argparse type function."""

    # --- gigabase variants ---
    @pytest.mark.parametrize("value,expected", [
        ("1G",    1_000),
        ("3G",    3_000),
        ("1.5G",  1_500),
        ("2gb",   2_000),
        ("1.5Gb", 1_500),
        ("0.5G",    500),
    ])
    def test_gigabases(self, value, expected):
        assert parse_genome_size(value) == expected

    # --- megabase variants ---
    @pytest.mark.parametrize("value,expected", [
        ("300M",  300),
        ("300Mb", 300),
        ("150mb", 150),
        ("300",   300),   # bare integer → Mb
        ("3000",  3_000),
    ])
    def test_megabases(self, value, expected):
        assert parse_genome_size(value) == expected

    # --- kilobase variants ---
    @pytest.mark.parametrize("value,expected", [
        ("2000K",  2),
        ("2000Kb", 2),
        ("1500kb", 2),    # 1.5 Mb → rounds to 2
        ("1000k",  1),
    ])
    def test_kilobases(self, value, expected):
        assert parse_genome_size(value) == expected

    def test_whitespace_is_stripped(self):
        assert parse_genome_size("  3G  ") == 3_000

    def test_case_insensitive(self):
        assert parse_genome_size("3g") == parse_genome_size("3G")

    def test_invalid_string_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_genome_size("notanumber")

    def test_zero_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_genome_size("0M")

    def test_negative_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            parse_genome_size("-1G")


# ===========================================================================
# 2. AssemblyEvaluator — parse_gfastats_output
# ===========================================================================

class TestParseGfastatsOutput:
    """
    parse_gfastats_output is pure Python (regex + numpy) — no subprocess needed.
    Values are log-transformed by the method before returning.
    """

    SAMPLE_OUTPUT = """\
    # contigs: 100
    Total contig length: 3000000000
    Contig N50: 1000000
    """

    def test_num_contigs_parsed(self, evaluator):
        metrics = evaluator.parse_gfastats_output(self.SAMPLE_OUTPUT)
        # num_contigs → log(100 + 1)
        assert "num_contigs" in metrics
        assert abs(metrics["num_contigs"] - np.log(101)) < 1e-9

    def test_n50_parsed(self, evaluator):
        metrics = evaluator.parse_gfastats_output(self.SAMPLE_OUTPUT)
        assert "n50" in metrics
        assert abs(metrics["n50"] - np.log(1_000_001)) < 1e-9

    def test_length_diff_is_log_transformed(self, evaluator):
        metrics = evaluator.parse_gfastats_output(self.SAMPLE_OUTPUT)
        assert "length_diff" in metrics
        # known_genome_size = 3_000 Mb; total contig length = 3_000_000_000 bytes
        # diff = |3_000_000_000 - 3_000| / 1_000_000 + 1 → log of that
        raw_diff = abs(3_000_000_000 - evaluator.known_genome_size) / 1_000_000
        expected = np.log(raw_diff + 1)
        assert abs(metrics["length_diff"] - expected) < 1e-6

    def test_missing_fields_absent_from_result(self, evaluator):
        output = "# contigs: 50\n"
        metrics = evaluator.parse_gfastats_output(output)
        assert "n50" not in metrics
        assert "length_diff" not in metrics

    def test_empty_output_returns_empty_dict(self, evaluator):
        assert evaluator.parse_gfastats_output("") == {}

    def test_all_three_keys_present(self, evaluator):
        metrics = evaluator.parse_gfastats_output(self.SAMPLE_OUTPUT)
        assert set(metrics.keys()) == {"num_contigs", "length_diff", "n50"}


# ===========================================================================
# 3. AssemblyEvaluator — parse_busco_results
# ===========================================================================

class TestParseBuscoResults:
    """parse_busco_results is a @staticmethod that reads a JSON file."""

    def _make_busco_json(self, tmp_path, single=950, multi=10, frag=20, miss=30):
        data = {
            "results": {
                "Single copy BUSCOs": single,
                "Multi copy BUSCOs":  multi,
                "Fragmented BUSCOs":  frag,
                "Missing BUSCOs":     miss,
            }
        }
        p = tmp_path / "busco_summary.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_single_copy_log_transformed(self, tmp_path):
        path = self._make_busco_json(tmp_path, single=950)
        m = AssemblyEvaluator.parse_busco_results(path)
        assert abs(m["single_copy"] - np.log(951)) < 1e-9

    def test_multi_copy_log_transformed(self, tmp_path):
        path = self._make_busco_json(tmp_path, multi=10)
        m = AssemblyEvaluator.parse_busco_results(path)
        assert abs(m["multi_copy"] - np.log(11)) < 1e-9

    def test_fragmented_log_transformed(self, tmp_path):
        path = self._make_busco_json(tmp_path, frag=20)
        m = AssemblyEvaluator.parse_busco_results(path)
        assert abs(m["fragmented"] - np.log(21)) < 1e-9

    def test_missing_log_transformed(self, tmp_path):
        path = self._make_busco_json(tmp_path, miss=30)
        m = AssemblyEvaluator.parse_busco_results(path)
        assert abs(m["missing"] - np.log(31)) < 1e-9

    def test_zero_values_give_log_one(self, tmp_path):
        path = self._make_busco_json(tmp_path, single=0, multi=0, frag=0, miss=0)
        m = AssemblyEvaluator.parse_busco_results(path)
        for key in ("single_copy", "multi_copy", "fragmented", "missing"):
            assert abs(m[key] - np.log(1)) < 1e-9

    def test_all_four_keys_returned(self, tmp_path):
        path = self._make_busco_json(tmp_path)
        m = AssemblyEvaluator.parse_busco_results(path)
        assert set(m.keys()) == {"single_copy", "multi_copy", "fragmented", "missing"}


# ===========================================================================
# 4. AssemblyEvaluator — parse_sniffles_vcf
# ===========================================================================

class TestParseSnifflesVcf:
    """parse_sniffles_vcf reads a VCF file and counts non-header lines."""

    def _write_vcf(self, tmp_path, num_sv=5):
        lines = ["##fileformat=VCFv4.2\n", "#CHROM\tPOS\tID\n"]
        for i in range(num_sv):
            lines.append(f"chr1\t{1000*i}\t.\tN\t<DEL>\t.\tPASS\t.\n")
        p = tmp_path / "out.vcf"
        p.write_text("".join(lines))
        return str(p)

    def test_sv_count_log_transformed(self, evaluator, tmp_path):
        path = self._write_vcf(tmp_path, num_sv=5)
        m = evaluator.parse_sniffles_vcf(path)
        assert abs(m["num_sv"] - np.log(6)) < 1e-9   # log(5 + 1)

    def test_zero_svs(self, evaluator, tmp_path):
        path = self._write_vcf(tmp_path, num_sv=0)
        m = evaluator.parse_sniffles_vcf(path)
        assert abs(m["num_sv"] - np.log(1)) < 1e-9   # log(0 + 1) = 0

    def test_missing_file_returns_zero(self, evaluator, tmp_path):
        m = evaluator.parse_sniffles_vcf(str(tmp_path / "nonexistent.vcf"))
        assert m["num_sv"] == 0

    def test_header_lines_not_counted(self, evaluator, tmp_path):
        # 10 header lines, 3 data lines
        lines = ["##header\n"] * 10 + ["#CHROM\n"] + ["chr1\t1\t.\n"] * 3
        p = tmp_path / "test.vcf"
        p.write_text("".join(lines))
        m = evaluator.parse_sniffles_vcf(str(p))
        assert abs(m["num_sv"] - np.log(4)) < 1e-9   # log(3 + 1)


# ===========================================================================
# 5. AssemblyEvaluator — calculate_weighted_sum
# ===========================================================================

class TestCalculateWeightedSum:
    """calculate_weighted_sum multiplies each log-metric by its weight and sums."""

    def test_all_zero_metrics_gives_zero(self, evaluator):
        metrics = {k: 0.0 for k in evaluator.weights}
        assert evaluator.calculate_weighted_sum(metrics) == 0.0

    def test_single_metric_contribution(self, evaluator):
        # Only n50 has a non-zero value; weight = 1.0
        metrics = {k: 0.0 for k in evaluator.weights}
        metrics["n50"] = 5.0
        result = evaluator.calculate_weighted_sum(metrics)
        assert abs(result - 5.0 * evaluator.weights["n50"]) < 1e-9

    def test_positive_weight_increases_score(self, evaluator):
        base = {k: 0.0 for k in evaluator.weights}
        with_n50 = {**base, "n50": 10.0}
        assert evaluator.calculate_weighted_sum(with_n50) > evaluator.calculate_weighted_sum(base)

    def test_negative_weight_decreases_score(self, evaluator):
        base = {k: 0.0 for k in evaluator.weights}
        with_missing = {**base, "missing": 10.0}
        assert evaluator.calculate_weighted_sum(with_missing) < evaluator.calculate_weighted_sum(base)

    def test_missing_metric_treated_as_zero(self, evaluator):
        result_full  = evaluator.calculate_weighted_sum({k: 0.0 for k in evaluator.weights})
        result_empty = evaluator.calculate_weighted_sum({})
        assert result_full == result_empty

    def test_full_dummy_metrics(self, evaluator, dummy_metrics):
        result = evaluator.calculate_weighted_sum(dummy_metrics)
        expected = sum(
            evaluator.weights[k] * dummy_metrics.get(k, 0.0)
            for k in evaluator.weights
        )
        assert abs(result - expected) < 1e-9


# ===========================================================================
# 6. AssemblyEvaluator — analyze_metric_contributions
# ===========================================================================

class TestAnalyzeMetricContributions:
    """analyze_metric_contributions returns a detailed breakdown dict."""

    def test_total_score_matches_weighted_sum(self, evaluator, dummy_metrics):
        analysis = evaluator.analyze_metric_contributions(dummy_metrics)
        expected = evaluator.calculate_weighted_sum(dummy_metrics)
        assert abs(analysis["total_score"] - expected) < 1e-9

    def test_positive_sum_is_non_negative(self, evaluator, dummy_metrics):
        analysis = evaluator.analyze_metric_contributions(dummy_metrics)
        assert analysis["positive_sum"] >= 0

    def test_negative_sum_is_non_negative(self, evaluator, dummy_metrics):
        # negative_sum stores the abs value of all negative contributions
        analysis = evaluator.analyze_metric_contributions(dummy_metrics)
        assert analysis["negative_sum"] >= 0

    def test_all_weight_keys_present_in_contributions(self, evaluator, dummy_metrics):
        analysis = evaluator.analyze_metric_contributions(dummy_metrics)
        for key in evaluator.weights:
            assert key in analysis["contributions"]

    def test_contribution_equals_weight_times_log_value(self, evaluator, dummy_metrics):
        analysis = evaluator.analyze_metric_contributions(dummy_metrics)
        for metric, data in analysis["contributions"].items():
            expected = evaluator.weights[metric] * dummy_metrics.get(metric, 0.0)
            assert abs(data["contribution"] - expected) < 1e-9

    def test_zero_metrics_gives_zero_totals(self, evaluator):
        zero_metrics = {k: 0.0 for k in evaluator.weights}
        analysis = evaluator.analyze_metric_contributions(zero_metrics)
        assert analysis["total_score"] == 0.0
        assert analysis["positive_sum"] == 0.0
        assert analysis["negative_sum"] == 0.0

    def test_proportion_sums_to_100_for_positive_contributions(self, evaluator):
        # Give only metrics with positive weights a non-zero value
        metrics = {k: 0.0 for k in evaluator.weights}
        metrics["n50"] = 5.0
        metrics["single_copy"] = 3.0
        metrics["reads_mapped"] = 2.0
        analysis = evaluator.analyze_metric_contributions(metrics)
        pos_props = [
            d["proportion"]
            for d in analysis["contributions"].values()
            if d["contribution"] > 0
        ]
        assert abs(sum(pos_props) - 100.0) < 1e-6


# ===========================================================================
# 7. AssemblyEvaluator — helpers
# ===========================================================================

class TestAssemblyEvaluatorHelpers:

    def test_minimap2_preset_hifi(self, evaluator):
        evaluator.ont = False
        assert evaluator._get_minimap2_preset() == "map-hifi"

    def test_minimap2_preset_ont(self, evaluator):
        evaluator.ont = True
        assert evaluator._get_minimap2_preset() == "map-ont"

    def test_get_current_stage_gfa(self, evaluator):
        stage = evaluator._get_current_stage(RuntimeError("gfa file missing"))
        assert "GFA" in stage or "conversion" in stage.lower()

    def test_get_current_stage_minimap2(self, evaluator):
        stage = evaluator._get_current_stage(RuntimeError("minimap2 failed"))
        assert "alignment" in stage.lower()

    def test_get_current_stage_busco(self, evaluator):
        stage = evaluator._get_current_stage(RuntimeError("busco error"))
        assert "BUSCO" in stage or "busco" in stage.lower()

    def test_get_current_stage_unknown(self, evaluator):
        stage = evaluator._get_current_stage(RuntimeError("totally unknown"))
        assert stage != ""

    def test_default_weights_loaded_when_no_json(self, evaluator):
        assert "n50" in evaluator.weights
        assert "single_copy" in evaluator.weights
        assert evaluator.weights["n50"] == 1.0
        assert evaluator.weights["missing"] == -1.0

    def test_patterns_compiled_on_init(self, evaluator):
        import re
        for key, pat in evaluator.gfastats_patterns.items():
            assert isinstance(pat, type(re.compile("")))
        for key, pat in evaluator.stats_patterns.items():
            assert isinstance(pat, type(re.compile("")))


# ===========================================================================
# 8. StagnationDetector
# ===========================================================================

class TestStagnationDetector:

    def test_not_converged_initially(self):
        d = StagnationDetector(patience=5, min_improvement=0.01)
        assert not d.update(0.5, 0)

    def test_converges_after_patience_trials_without_improvement(self):
        d = StagnationDetector(patience=5, min_improvement=0.1)
        d.update(1.0, 0)   # sets best_value = 1.0
        for i in range(1, 6):
            result = d.update(1.0, i)   # no improvement each time
        assert result is True

    def test_resets_on_significant_improvement(self):
        d = StagnationDetector(patience=5, min_improvement=0.1)
        d.update(1.0, 0)
        for i in range(1, 5):
            d.update(1.0, i)   # stagnating
        d.update(2.0, 5)       # big jump → reset
        assert d.stagnation_count == 0

    def test_best_value_tracks_maximum(self):
        d = StagnationDetector(patience=10, min_improvement=0.01)
        d.update(1.0, 0)
        d.update(1.5, 1)
        d.update(1.2, 2)
        assert d.best_value >= 1.5

    def test_history_recorded(self):
        d = StagnationDetector(patience=10, min_improvement=0.01)
        for i in range(5):
            d.update(float(i), i)
        assert len(d.convergence_history) == 5

    def test_has_converged_method(self):
        d = StagnationDetector(patience=3, min_improvement=0.1)
        d.update(1.0, 0)
        for i in range(1, 4):
            d.update(1.0, i)
        assert d.has_converged()


# ===========================================================================
# 9. PlateauDetector
# ===========================================================================

class TestPlateauDetector:

    def test_not_converged_with_few_values(self):
        d = PlateauDetector(plateau_threshold=0.01, min_plateau_length=8)
        for i in range(5):
            assert not d.update(1.0, i)

    def test_converges_on_flat_values(self):
        d = PlateauDetector(plateau_threshold=0.01, min_plateau_length=8)
        result = False
        for i in range(30):
            result = d.update(1.0, i)
        assert result is True

    def test_does_not_converge_with_varying_values(self):
        d = PlateauDetector(plateau_threshold=0.01, min_plateau_length=8)
        values = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.55]
        for i, v in enumerate(values):
            d.update(v, i)
        assert d.plateau_count == 0

    def test_plateau_count_resets_on_variation(self):
        d = PlateauDetector(plateau_threshold=0.01, min_plateau_length=5)
        for i in range(10):
            d.update(1.0, i)     # plateau building up
        d.update(5.0, 10)        # sudden change
        assert d.plateau_count == 0


# ===========================================================================
# 10. StatisticalConvergenceDetector
# ===========================================================================

class TestStatisticalConvergenceDetector:

    def test_returns_false_before_two_windows(self):
        d = StatisticalConvergenceDetector(window_size=5)
        for i in range(9):
            assert not d.update(float(i), i)

    def test_converges_when_windows_are_identical(self):
        d = StatisticalConvergenceDetector(window_size=5, significance_level=0.05)
        # Two windows of identical values → no significant difference → converged
        results = [d.update(1.0, i) for i in range(10)]
        assert results[-1] == True

    def test_does_not_converge_when_recent_window_is_clearly_better(self):
        d = StatisticalConvergenceDetector(window_size=5, significance_level=0.05)
        older  = [1.0] * 5
        recent = [100.0] * 5
        for i, v in enumerate(older + recent):
            result = d.update(v, i)
        assert not result


# ===========================================================================
# 11. RelativeImprovementDetector
# ===========================================================================

class TestRelativeImprovementDetector:

    def test_not_converged_initially(self):
        d = RelativeImprovementDetector(threshold=0.001, patience=5)
        assert not d.update(1.0, 0)

    def test_converges_after_patience_with_no_improvement(self):
        d = RelativeImprovementDetector(threshold=0.01, patience=5)
        d.update(1.0, 0)
        for i in range(1, 7):
            d.update(1.0, i)   # relative improvement = 0 every time
        assert d.poor_improvement_count >= 5

    def test_resets_count_on_good_improvement(self):
        d = RelativeImprovementDetector(threshold=0.01, patience=5)
        d.update(1.0, 0)
        for i in range(1, 4):
            d.update(1.0, i)
        d.update(2.0, 4)   # 100% relative improvement → reset
        assert d.poor_improvement_count == 0

    def test_handles_division_by_zero(self):
        d = RelativeImprovementDetector(threshold=0.01, patience=5)
        d.update(0.0, 0)
        # When previous value is 0, relative_improvement = inf → no increment
        d.update(1.0, 1)
        assert d.poor_improvement_count == 0


# ===========================================================================
# 12. MultiCriteriaConvergenceDetector
# ===========================================================================

class TestMultiCriteriaConvergenceDetector:

    def test_always_false_before_trial_5(self):
        d = MultiCriteriaConvergenceDetector()
        for i in range(5):
            converged, _ = d.update(0.85, i)
            assert not converged

    def test_max_trials_triggers_convergence(self):
        d = MultiCriteriaConvergenceDetector(max_trials=10)
        converged, methods = d.update(0.85, 10)
        assert converged
        assert "max_trials_reached" in methods

    def test_stable_sequence_eventually_converges(self):
        d = MultiCriteriaConvergenceDetector(
            stagnation_patience=5,
            min_improvement=0.1,
            threshold=0.001,
            patience=5,
            plateau_threshold=0.01,
            min_plateau_length=5,
            window_size=5,
        )
        converged = False
        for i in range(60):
            converged, _ = d.update(1.0, i)
            if converged:
                break
        assert converged

    def test_multi_objective_scalar_aggregation(self):
        d = MultiCriteriaConvergenceDetector(
            directions=["maximize", "minimize"],
            stagnation_patience=5,
            min_improvement=0.1,
            plateau_threshold=0.01,
            min_plateau_length=5,
            window_size=5,
        )
        # Should not raise; list values are aggregated to a scalar internally
        converged, _ = d.update([0.8, 0.2], 6)
        assert isinstance(converged, bool)

    def test_has_converged_reflects_last_update(self):
        d = MultiCriteriaConvergenceDetector(max_trials=5)
        d.update(0.85, 5)
        assert d.has_converged() is True

    def test_non_numeric_value_does_not_crash(self):
        d = MultiCriteriaConvergenceDetector()
        # Should silently coerce to 0.0 and not raise
        converged, _ = d.update("bad_value", 6)
        assert isinstance(converged, bool)

    def test_convergence_requires_majority_of_detectors(self):
        """At least floor(4/2) = 2 of the 4 detectors must agree."""
        d = MultiCriteriaConvergenceDetector()
        # Inject a known _last_result as if only 1 of 4 detectors voted
        d._last_result = (False, ["stagnation"])
        assert not d.has_converged()


# ===========================================================================
# 13. ObjectiveBuilder — initialisation
# ===========================================================================

class TestObjectiveBuilderInit:
    """
    Tests that ObjectiveBuilder stores constructor arguments correctly and
    that build_objective() returns a callable — without executing any trials.
    """

    @pytest.fixture
    def mock_evaluator(self, default_weights):
        ev = MagicMock()
        ev.weights = default_weights
        ev.known_genome_size = 3_000
        ev.input_reads = Path("/dummy/reads.fastq")
        return ev

    @pytest.fixture
    def builder(self, mock_evaluator, tmp_path):
        with patch("utils.objective.SubprocessLogger", _mock_subprocess_logger):
            return ObjectiveBuilder(
                evaluator=mock_evaluator,
                input_reads=Path("/dummy/reads.fastq"),
                haploid_genome_size=3_000,
                threads=8,
                output_dir=str(tmp_path),
            )

    def test_stores_genome_size(self, builder):
        assert builder.haploid_genome_size == 3_000

    def test_stores_threads(self, builder):
        assert builder.threads == 8

    def test_defaults_to_single_objective(self, builder):
        assert not builder.is_multi_objective

    def test_objectives_derived_from_evaluator_weights(self, builder, default_weights):
        assert set(builder.objectives) == set(default_weights.keys())

    def test_hic_defaults_to_none(self, builder):
        assert builder.hic1 is None
        assert builder.hic2 is None

    def test_ul_defaults_to_none(self, builder):
        assert builder.ul is None

    def test_sensitive_defaults_to_false(self, builder):
        assert not builder.sensitive

    def test_include_busco_defaults_to_true(self, builder):
        assert builder.include_busco

    def test_build_objective_returns_callable(self, builder):
        obj_fn = builder.build_objective()
        assert callable(obj_fn)

    def test_multi_objective_flag_stored(self, mock_evaluator, tmp_path):
        with patch("utils.objective.SubprocessLogger", _mock_subprocess_logger):
            builder = ObjectiveBuilder(
                evaluator=mock_evaluator,
                input_reads=Path("/dummy/reads.fastq"),
                haploid_genome_size=3_000,
                threads=8,
                is_multi_objective=True,
                output_dir=str(tmp_path),
            )
        assert builder.is_multi_objective

    def test_custom_objectives_override_weights(self, mock_evaluator, tmp_path):
        custom = ["n50", "single_copy"]
        with patch("utils.objective.SubprocessLogger", _mock_subprocess_logger):
            builder = ObjectiveBuilder(
                evaluator=mock_evaluator,
                input_reads=Path("/dummy/reads.fastq"),
                haploid_genome_size=3_000,
                threads=8,
                objectives=custom,
                output_dir=str(tmp_path),
            )
        assert builder.objectives == custom
