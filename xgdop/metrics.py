"""Offline aggregation of measured explanation metrics.

This module accepts actual decision-level measurements. It does not generate
deletion costs, estimate feature scales, or reproduce any reported benchmark.
Comparative counterfactual statistics require a shared scenario denominator.

Comparative metrics average decisions within each scenario-seed run, then
eligible runs within each scenario, then scenarios equally. Pass scenario and
seed IDs explicitly; the number of rows does not determine this hierarchy.
"""
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Hashable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class ScenarioMean:
    """One group's mean and eligibility coverage, in first-seen order."""

    scenario_id: Hashable
    total_decisions: int
    eligible_decisions: int
    mean: float | None
    total_runs: int = 1
    included_runs: int = 1
    validity: float | None = None


@dataclass(frozen=True)
class RunMean:
    """One scenario-seed run; a zero-valid run has mean None and validity 0."""

    scenario_id: Hashable
    seed_id: Hashable
    total_decisions: int
    eligible_decisions: int
    mean: float | None
    validity: float


@dataclass(frozen=True)
class MacroMetric:
    """Equal-weight eligible-group mean; missing observations stay missing."""

    overall: float | None
    per_decision: tuple[float | None, ...]
    scenarios: tuple[ScenarioMean, ...]
    normalization_scope: str = "none"
    runs: tuple[RunMean, ...] = ()
    validity_overall: float | None = None

    @property
    def total_decisions(self) -> int:
        return len(self.per_decision)

    @property
    def eligible_decisions(self) -> int:
        return sum(group.eligible_decisions for group in self.scenarios)

    @property
    def included_scenarios(self) -> int:
        return sum(group.mean is not None for group in self.scenarios)

    @property
    def excluded_scenarios(self) -> int:
        return len(self.scenarios) - self.included_scenarios


def _vector(values, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _groups(scenario_ids: Iterable[Hashable], size: int) -> dict[Hashable, list[int]]:
    if isinstance(scenario_ids, (str, bytes)):
        raise ValueError("scenario_ids must contain one ID per decision")
    try:
        ids = list(scenario_ids)
    except TypeError as error:
        raise ValueError("scenario_ids must contain one ID per decision") from error
    if len(ids) != size:
        raise ValueError("scenario_ids length must match the decision count")
    groups: dict[Hashable, list[int]] = {}
    for index, scenario_id in enumerate(ids):
        try:
            hash(scenario_id)
        except TypeError as error:
            raise ValueError("each scenario ID must be hashable") from error
        # NaN-like labels would split one nominal group into multiple groups.
        if isinstance(scenario_id, Real) and not np.isfinite(scenario_id):
            raise ValueError("numeric scenario IDs must be finite")
        if scenario_id is None:
            raise ValueError("scenario IDs must not be missing")
        if isinstance(scenario_id, tuple):
            _groups(scenario_id, len(scenario_id))  # Validate composite labels too.
        groups.setdefault(scenario_id, []).append(index)
    return groups


def _mask(valid, size: int) -> np.ndarray:
    if valid is None:
        return np.ones(size, dtype=bool)
    array = np.asarray(valid)
    if array.ndim != 1 or array.size != size:
        raise ValueError("valid must be one-dimensional and match the decision count")
    if array.size and array.dtype.kind != "b":
        raise ValueError("valid must contain Boolean values")
    return array.astype(bool)


def _epsilon(eps: float) -> float:
    if isinstance(eps, bool) or not isinstance(eps, Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and strictly positive")
    return float(eps)


def _labels(ids, size: int, name: str) -> list[Hashable]:
    if isinstance(ids, (str, bytes)):
        raise ValueError(f"{name} must contain one ID per decision")
    try:
        labels = list(ids)
    except TypeError as error:
        raise ValueError(f"{name} must contain one ID per decision") from error
    _groups(labels, size)
    return labels


def _mean(values) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.sum(values / values.size)) if values.size else None
    if mean is not None and not np.isfinite(mean):
        raise ValueError("the mean is not finite")
    return mean


def _aggregate(values: np.ndarray, groups, valid: np.ndarray, *, normalization_scope="none") -> MacroMetric:
    if not np.isfinite(values[valid]).all():
        raise ValueError("eligible metric values must be finite")
    summaries = []
    for scenario_id, indices in groups.items():
        eligible = np.asarray(indices)[valid[indices]]
        # Division before summation prevents overflow for finite huge values.
        mean = _mean(values[eligible])
        summaries.append(ScenarioMean(scenario_id, len(indices), int(eligible.size), mean,
                                      included_runs=int(mean is not None),
                                      validity=float(eligible.size / len(indices))))
    means = [group.mean for group in summaries if group.mean is not None]
    overall = _mean(means)
    per_decision = tuple(float(value) if keep else None for value, keep in zip(values, valid))
    return MacroMetric(overall, per_decision, tuple(summaries), normalization_scope,
                       validity_overall=_mean([group.validity for group in summaries]))


def _run_aggregate(values, scenario_ids, seed_ids, valid, *, normalization_scope="none"):
    if not np.isfinite(values[valid]).all():
        raise ValueError("eligible metric values must be finite")
    run_groups = _groups(zip(scenario_ids, seed_ids), len(values))
    scenario_runs: dict[Hashable, list[RunMean]] = {}
    runs = []
    for (scenario_id, seed_id), indices in run_groups.items():
        eligible = np.asarray(indices)[valid[indices]]
        run = RunMean(scenario_id, seed_id, len(indices), int(eligible.size),
                      _mean(values[eligible]), float(eligible.size / len(indices)))
        runs.append(run)
        scenario_runs.setdefault(scenario_id, []).append(run)
    scenarios = []
    for scenario_id, entries in scenario_runs.items():
        included = [entry.mean for entry in entries if entry.mean is not None]
        scenarios.append(ScenarioMean(
            scenario_id, sum(entry.total_decisions for entry in entries),
            sum(entry.eligible_decisions for entry in entries), _mean(included),
            len(entries), len(included), _mean([entry.validity for entry in entries]),
        ))
    overall = _mean([group.mean for group in scenarios if group.mean is not None])
    per_decision = tuple(float(value) if keep else None for value, keep in zip(values, valid))
    return MacroMetric(overall, per_decision, tuple(scenarios), normalization_scope,
                       tuple(runs), _mean([group.validity for group in scenarios]))


def scenario_macro_mean(values, scenario_ids: Iterable[Hashable], *, valid=None) -> MacroMetric:
    """Average eligible decisions within each group, then weight groups equally.

    Invalid decisions and all-invalid groups stay ``None`` in the result.
    An all-invalid group contributes no artificial zero to the macro average.
    The returned counts make this exclusion visible. Empty input returns
    ``overall=None`` with zero counts. Only eligible values must be finite.
    """
    values = _vector(values, "values")
    groups = _groups(scenario_ids, len(values))
    return _aggregate(values, groups, _mask(valid, len(values)))


def run_scenario_macro_mean(values, scenario_ids, seed_ids, *, valid=None) -> MacroMetric:
    """Decision -> eligible run -> eligible scenario macro; expose coverage.

    Run values use eligible observations only. A zero-valid run has mean None,
    and a scenario without any eligible run has mean None. Such missing values
    are excluded from the conditional metric, never replaced with zero.
    Validity instead includes EVERY observed attempted decision and EVERY run,
    including zero-valid runs/scenarios, with equal run then scenario weights.
    Missing unattempted runs cannot be inferred; callers must report them in
    their experiment metadata. ``scenario_macro_mean`` is a simpler generic
    two-level helper and is not a substitute for this multi-seed protocol.
    """
    values = _vector(values, "values")
    scenarios = _labels(scenario_ids, len(values), "scenario_ids")
    seeds = _labels(seed_ids, len(values), "seed_ids")
    return _run_aggregate(values, scenarios, seeds, _mask(valid, len(values)))


def _element_rows(values, name: str) -> list[np.ndarray]:
    if isinstance(values, np.ndarray) and values.ndim != 2:
        raise ValueError(f"{name} must contain one explanation-element row per decision")
    try:
        rows = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must contain one explanation-element row per decision") from error
    return [_vector(row, f"{name} row") for row in rows]


def projected_cost_fidelity(
    baseline_costs,
    deleted_costs,
    contributions,
    scenario_ids: Iterable[Hashable],
    *,
    seed_ids: Iterable[Hashable],
    top_k: int = 3,
    eps: float = 1e-6,
) -> MacroMetric:
    """Aggregate top-contribution projected-cost fidelity over scenarios.

    ``baseline_costs`` has shape [D]. ``deleted_costs`` and signed
    ``contributions`` both have shape [D, E], with the same ordering of all
    declared explanation elements (concepts, edges, features, etc.). Rows may
    have different E, for example candidate-local graph edge sets. Every row
    needs at least top_k eligible elements; shorter/empty rows raise ValueError
    instead of creating an artificial zero. The comparative budget is top_k=3.
    Deletion costs must come from re-solving the declared projected problem
    after each element deletion, not from subtracting an additive score term.

    For decision d, select the top-k absolute contributions (ties use the lower
    element index). Let delta[d,e] = abs(deleted_cost[d,e] - baseline_cost[d]).
    The value is mean(delta[d, selected]) / (max_all_E delta[d,e] + eps).
    Average decisions within runs, eligible seeds within scenarios, then
    scenarios equally. Pass both ID arrays. Zero cost changes yield zero;
    no observations yield None. This function aggregates supplied measurements;
    it does not define a method's element deletion operator.
    """
    eps = _epsilon(eps)
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    baseline = _vector(baseline_costs, "baseline_costs")
    scenarios = _labels(scenario_ids, len(baseline), "scenario_ids")
    seeds = _labels(seed_ids, len(baseline), "seed_ids")
    deleted = _element_rows(deleted_costs, "deleted_costs")
    contributions = _element_rows(contributions, "contributions")
    if len(deleted) != len(baseline) or len(contributions) != len(baseline):
        raise ValueError("element row counts must match baseline_costs")
    if not np.isfinite(baseline).all():
        raise ValueError("baseline costs must be finite")
    values = []
    for cost, deletion, contribution in zip(baseline, deleted, contributions):
        if deletion.size != contribution.size:
            raise ValueError("each cost and contribution row must use identical element ordering and size")
        if top_k > deletion.size:
            raise ValueError("top_k must not exceed the number of explanation elements")
        if not (np.isfinite(deletion).all() and np.isfinite(contribution).all()):
            raise ValueError("costs and contributions must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            delta = np.abs(deletion - cost)
            denominator = np.max(delta) + eps
        if not np.isfinite(delta).all() or not np.isfinite(denominator):
            raise ValueError("absolute projected-cost changes and denominators must be finite")
        selected = np.argsort(-np.abs(contribution), kind="stable")[:top_k]
        values.append(float(np.sum((delta[selected] / denominator) / top_k)))
    return _run_aggregate(np.asarray(values), scenarios, seeds, np.ones(len(baseline), dtype=bool),
                          normalization_scope="per_decision_all_explanation_elements")


def counterfactual_minimality(
    distances,
    valid,
    scenario_ids: Iterable[Hashable],
    *,
    seed_ids: Iterable[Hashable],
    reference_maxima: Mapping[Hashable, float],
    eps: float = 1e-6,
) -> MacroMetric:
    """Aggregate comparative minimality using mandatory shared scenario maxima.

    The caller must compute distances with the declared feature scales and
    validate the counterfactual's target action, feasibility, and identity
    support. This function only aggregates those supplied measurements.
    ``reference_maxima`` must contain one common maximum valid distance per
    scenario, computed across all compared methods and seeds using the same
    declared feature scales. Pass the SAME mapping to every method's call.
    It must cover every eligible scenario and observed valid distance.
    Invalid distances are ignored, including NaN placeholders. An eligible
    distance is divided by its shared scenario maximum plus eps.
    Valid-only decision means are computed per run, then eligible run means
    per scenario, then eligible scenario means equally. Zero-valid runs or
    scenarios have mean None with coverage counts. Validity includes all
    attempts/runs/scenarios, even when their conditional metric is missing.
    All-zero valid
    distances yield zero. Within one scenario, the common scale preserves the
    ordering of standardized L1 distances. The cross-scenario macro measures
    normalized distances, not the raw-L1 mean. Measurement generation and
    numerical paper results are outside this aggregation function's scope.
    """
    if not isinstance(reference_maxima, Mapping):
        raise ValueError("reference_maxima must map scenario IDs to shared comparison maxima")
    return _counterfactual_statistic(distances, valid, scenario_ids, seed_ids, eps=eps,
                                    reference_maxima=reference_maxima)


def relative_counterfactual_minimality(
    distances,
    valid,
    scenario_ids: Iterable[Hashable],
    *,
    seed_ids: Iterable[Hashable],
    eps: float = 1e-6,
) -> MacroMetric:
    """Single-input diagnostic with each group's own valid-distance maximum.

    This relative statistic is approximately invariant to rescaling all
    distances within a group. It is NOT the comparative metric protocol and
    must not establish absolute minimality differences between methods.
    Use ``counterfactual_minimality`` for shared-scenario-scale comparisons;
    its cross-scenario macro summarizes normalized rather than raw distances.
    Valid-only means, missing groups and coverage use the same accounting.
    """
    return _counterfactual_statistic(distances, valid, scenario_ids, seed_ids, eps=eps,
                                    reference_maxima=None)


def _counterfactual_statistic(distances, valid, scenario_ids, seed_ids, *, eps, reference_maxima):
    eps = _epsilon(eps)
    distances = _vector(distances, "distances")
    scenarios = _labels(scenario_ids, len(distances), "scenario_ids")
    seeds = _labels(seed_ids, len(distances), "seed_ids")
    groups = _groups(scenarios, len(distances))
    if valid is None:
        raise ValueError("counterfactual validity must be supplied as a Boolean vector")
    mask = _mask(valid, len(distances))
    if not np.isfinite(distances[mask]).all() or np.any(distances[mask] < 0):
        raise ValueError("valid distances must be finite and nonnegative")
    values = np.full(len(distances), np.nan)
    for scenario_id, indices in groups.items():
        eligible = np.asarray(indices)[mask[indices]]
        if eligible.size:
            maximum = float(np.max(distances[eligible]))
            if reference_maxima is not None:
                if scenario_id not in reference_maxima:
                    raise ValueError("reference_maxima must cover every eligible group")
                reference = reference_maxima[scenario_id]
                if (isinstance(reference, bool) or not isinstance(reference, Real)
                        or not np.isfinite(reference) or reference < 0):
                    raise ValueError("reference maxima must be finite and nonnegative")
                if reference < maximum:
                    raise ValueError("reference maxima must cover the observed valid distances")
                maximum = float(reference)
            denominator = maximum + eps
            if not np.isfinite(denominator):
                raise ValueError("normalization denominators must be finite")
            values[eligible] = distances[eligible] / denominator
    scope = "provided_group_reference" if reference_maxima is not None else "within_input_group"
    return _run_aggregate(values, scenarios, seeds, mask, normalization_scope=scope)
