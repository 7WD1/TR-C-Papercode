"""Hand-calculated checks for offline metrics, not paper benchmarks."""
import numpy as np
import pytest

from xgdop.metrics import (
    counterfactual_minimality as _counterfactual_minimality,
    projected_cost_fidelity as _projected_cost_fidelity,
    relative_counterfactual_minimality as _relative_counterfactual_minimality,
    run_scenario_macro_mean,
    scenario_macro_mean,
)


def projected_cost_fidelity(baseline, *args, **options):
    """Primitive fixtures explicitly have one seed per scenario."""
    options.setdefault("seed_ids", [0] * len(baseline))
    return _projected_cost_fidelity(baseline, *args, **options)


def counterfactual_minimality(distances, *args, **options):
    options.setdefault("seed_ids", [0] * len(distances))
    return _counterfactual_minimality(distances, *args, **options)


def relative_counterfactual_minimality(distances, *args, **options):
    options.setdefault("seed_ids", [0] * len(distances))
    return _relative_counterfactual_minimality(distances, *args, **options)


def test_fidelity_uses_abs_contribution_selection_and_all_concept_denominator():
    # Contributions select concepts 0,1,2, even though concept3 changes cost most.
    result = projected_cost_fidelity(
        [10, 10],
        [[8, 14, 16, 30], [10, 10, 10, 10]],
        [[-9, 8, -7, 0.1], [0, 0, 0, 0]],
        ["a", "a"],
    )
    assert result.per_decision == pytest.approx((4 / (20 + 1e-6), 0))
    assert result.overall == pytest.approx(2 / (20 + 1e-6))
    assert result.eligible_decisions == 2
    # Candidate-local graph/feature element sets need not have equal sizes.
    ragged = _projected_cost_fidelity(
        [0, 0], [[1, 2, 3], [2, 4, 6, 100]],
        [[3, 2, 1], [3, 2, 1, 0]], ["a", "a"], seed_ids=[0, 1],
    )
    assert ragged.overall == pytest.approx((2 / (3 + 1e-6) + 4 / (100 + 1e-6)) / 2)


def test_fidelity_ties_use_lower_concept_index_and_macro_is_not_pooled():
    result = projected_cost_fidelity(
        [0, 0, 0, 0],
        [[1, 1, 1, 9], [1, 1, 1, 9], [1, 1, 1, 9], [9, 9, 9, 9]],
        [[2, -2, 2, -2]] * 4,
        ["three", "three", "three", "one"],
    )
    # Top3 ties choose0,1,2. Equal group weighting gives5/(9+eps),
    # while pooled weighting would incorrectly give3/(9+eps).
    assert result.overall == pytest.approx(5 / (9 + 1e-6))
    assert result.scenarios[0].eligible_decisions == 3
    assert result.scenarios[1].eligible_decisions == 1
    # Equal seed weighting must survive unequal numbers of decisions per run.
    seeded = _projected_cost_fidelity(
        [0] * 4, [[0, 0, 0]] * 3 + [[9, 9, 9]], [[3, 2, 1]] * 4,
        ["a"] * 4, seed_ids=[0, 0, 0, 1],
    )
    assert seeded.overall == pytest.approx(4.5 / (9 + 1e-6))


def test_counterfactual_valid_only_normalization_and_coverage():
    result = counterfactual_minimality(
        [1, 3, 1e9, 2, np.nan, np.inf],
        [True, True, False, True, False, False],
        ["a", "a", "a", "b", "empty", "empty"],
        reference_maxima={"a": 10, "b": 20},
    )
    # Common maxima include other compared methods, not just this call's values.
    expected_a = 2 / (10 + 1e-6)
    expected_b = 2 / (20 + 1e-6)
    assert result.overall == pytest.approx((expected_a + expected_b) / 2)
    assert result.per_decision[2] is None
    assert result.scenarios[2].mean is None
    assert result.included_scenarios == 2
    assert result.excluded_scenarios == 1
    assert result.eligible_decisions == 3
    assert result.total_decisions == 6
    assert result.validity_overall == pytest.approx((2 / 3 + 1 + 0) / 3)
    # A zero-valid run stays in validity, but never becomes minimality zero.
    seeded = _counterfactual_minimality(
        [0, 0, 0, 8, np.nan], [True, True, True, True, False], ["a"] * 5,
        seed_ids=[0, 0, 0, 1, 2], reference_maxima={"a": 10},
    )
    assert seeded.overall == pytest.approx(4 / (10 + 1e-6))
    assert seeded.scenarios[0].total_runs == 3 and seeded.scenarios[0].included_runs == 2
    assert seeded.runs[2].mean is None
    assert seeded.validity_overall == pytest.approx(2 / 3)


def test_safe_empty_zero_and_all_invalid_cases():
    for result in (
        scenario_macro_mean([], []),
        run_scenario_macro_mean([], [], []),
        projected_cost_fidelity([], [], [], []),
        projected_cost_fidelity([], np.empty((0, 7)), np.empty((0, 7)), []),
        counterfactual_minimality([], [], [], reference_maxima={}),
    ):
        assert result.overall is None
        assert result.total_decisions == result.included_scenarios == 0
    zeros = counterfactual_minimality([0, 0], [True, True], ["zero", "zero"], reference_maxima={"zero": 0})
    assert zeros.overall == 0
    missing = counterfactual_minimality([np.nan], [False], ["missing"], reference_maxima={})
    assert missing.overall is None and missing.excluded_scenarios == 1
    assert missing.validity_overall == 0
    with pytest.raises(ValueError):
        projected_cost_fidelity([0], [[]], [[]], ["a"])
    with pytest.raises(ValueError):
        _projected_cost_fidelity([0], [[1, 2, 3]], [[3, 2, 1]], ["a"], seed_ids=[])


def test_macro_preserves_explicit_scenario_seed_grouping_and_missing_counts():
    grouped = scenario_macro_mean(
        [0, 0, 8, 2, np.nan],
        [("a", 0), ("a", 0), ("a", 1), ("b", 0), ("b", 1)],
        valid=[True, True, True, True, False],
    )
    assert grouped.overall == pytest.approx(10 / 3)
    assert grouped.scenarios[0].scenario_id == ("a", 0)
    assert grouped.scenarios[-1].total_decisions == 1
    assert grouped.scenarios[-1].eligible_decisions == 0
    assert grouped.per_decision[-1] is None
    hierarchical = run_scenario_macro_mean(
        [0, 0, 0, 8, 2, np.nan, np.nan],
        ["a", "a", "a", "a", "b", "b", "c"], [0, 0, 0, 1, 0, 1, 0],
        valid=[True, True, True, True, True, False, False],
    )
    # a=(0+8)/2=4; b=2; c=NR. Overall(4+2)/2=3, not pooled10/5=2.
    assert hierarchical.overall == pytest.approx(3)
    assert hierarchical.validity_overall == pytest.approx((1 + 0.5 + 0) / 3)
    assert hierarchical.scenarios[0].mean == pytest.approx(4)
    assert hierarchical.scenarios[1].included_runs == 1
    assert hierarchical.scenarios[-1].mean is None


def test_common_cf_reference_retains_absolute_distance_comparison():
    # Method-specific maxima cannot reveal this100-fold distance difference.
    small = relative_counterfactual_minimality([1, 1], [True, True], ["a", "a"])
    large = relative_counterfactual_minimality([100, 100], [True, True], ["a", "a"])
    assert small.overall == pytest.approx(large.overall, abs=2e-6)
    assert small.normalization_scope == "within_input_group"
    common_small = counterfactual_minimality(
        [1, 1], [True, True], ["a", "a"], reference_maxima={"a": 100},
    )
    common_large = counterfactual_minimality(
        [100, 100], [True, True], ["a", "a"], reference_maxima={"a": 100},
    )
    assert common_small.overall * 100 == pytest.approx(common_large.overall)
    assert common_small.overall == pytest.approx(1 / (100 + 1e-6))
    assert common_large.overall == pytest.approx(100 / (100 + 1e-6))
    assert common_small.normalization_scope == "provided_group_reference"
    with pytest.raises(TypeError):
        counterfactual_minimality([1], [True], ["a"])
    with pytest.raises(ValueError):
        counterfactual_minimality([1], [True], ["a"], reference_maxima=None)
    with pytest.raises(ValueError):
        counterfactual_minimality([1], None, ["a"], reference_maxima={"a": 1})


@pytest.mark.parametrize(
    "call",
    [
        lambda: projected_cost_fidelity([0], [[1, 2, 3]], [[1, 2]], ["a"]),
        lambda: projected_cost_fidelity([0, 0], [[1, 2, 3]], [[1, 2, 3]], ["a", "b"]),
        lambda: projected_cost_fidelity([0], [[1, 2, 3]], [[1, 2, 3]], []),
        lambda: projected_cost_fidelity([0], [[1, 2]], [[1, 2]], ["a"]),
        lambda: projected_cost_fidelity([0], [[1, 2, 3]], [[1, 2, 3]], ["a"], top_k=0),
        lambda: projected_cost_fidelity([0], [[1, 2, 3]], [[1, 2, 3]], ["a"], top_k=1.5),
        lambda: projected_cost_fidelity([0], [[1, np.nan, 3]], [[1, 2, 3]], ["a"]),
        lambda: projected_cost_fidelity([0], [[1, 2, 3]], [[1, 2, 3]], ["a"], eps=0),
        lambda: counterfactual_minimality([1], [True, False], ["a"], reference_maxima={"a": 1}),
        lambda: counterfactual_minimality([1], [1], ["a"], reference_maxima={"a": 1}),
        lambda: counterfactual_minimality([-1], [True], ["a"], reference_maxima={"a": 1}),
        lambda: counterfactual_minimality([np.nan], [True], ["a"], reference_maxima={"a": 1}),
        lambda: counterfactual_minimality([1], [True], ["a"], reference_maxima={"a": 1}, eps=np.inf),
        lambda: counterfactual_minimality([1], [True], ["a"], reference_maxima={}),
        lambda: counterfactual_minimality([1], [True], ["a"], reference_maxima={"a": 0.5}),
        lambda: counterfactual_minimality([1], [True], ["a"], reference_maxima={"a": np.nan}),
        lambda: scenario_macro_mean([[1]], ["a"]),
        lambda: scenario_macro_mean([1], [["unhashable"]]),
        lambda: scenario_macro_mean([1], [np.nan]),
        lambda: scenario_macro_mean([1], [("a", None)]),
        lambda: scenario_macro_mean([1], "a"),
    ],
)
def test_reject_inconsistent_or_undefined_measurements(call):
    with pytest.raises(ValueError):
        call()
