"""One affine constraint builder shared by SCIP deployment and OSQP training.

SCIP solves the actual mixed binary/continuous quadratic objective via an exact
convex epigraph. OSQP solves the continuous relaxation. The backward pass is a
fixed-active-set KKT derivative to the proposal only; predictor coefficients and
constraint right hand sides are held fixed for this local training derivative.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.linalg import qr
import torch
import osqp



@dataclass
class ProjectionResult:
    decision: np.ndarray
    full: np.ndarray
    status: str
    fallback: bool
    objective: float
    gap: float | None
    certificate: dict


class FleetProblem:
    def __init__(self, state, output, cfg, fallback=False):
        self.state, self.cfg, self.actions = state, cfg, output["actions"]
        self.time = output["time"].detach().numpy().astype(float)
        self.energy = output["energy"].detach().numpy().astype(float)
        self.fallback = fallback
        count = len(self.actions)
        charges = [i for i, a in enumerate(self.actions) if a.kind == "chg"]
        self.charge_index = {i: count + k for k, i in enumerate(charges)}
        self.reject_start = count + len(charges)
        self.dimension = self.reject_start + len(state.request_ids)
        active_zones = sorted(set(state.request_zone.tolist()))
        self.active_zones = active_zones
        self.aux_start = self.dimension
        self.slack_start = self.dimension + len(active_zones)
        self.total_dimension = self.slack_start + (2 * len(state.zone_xy) if fallback else 0)
        n = self.total_dimension
        self.qbar = {}
        for i in charges:
            a = self.actions[i]
            self.qbar[i] = max(0, min(state.battery[a.vehicle] * (1-state.soc[a.vehicle]) / cfg["charging_efficiency"],
                                  cfg["pair_power_kw"] * max(0, cfg["interval_min"]-self.time[i]) / 60))
        self.weight = np.ones(self.dimension)
        for i, qi in self.charge_index.items():
            self.weight[qi] = max(self.qbar[i], 1e-6) ** -2
        self.lower = np.zeros(n)
        self.upper = np.ones(n)
        for i, a in enumerate(self.actions):
            if a.kind == "idle":
                self.upper[i] = 0  # idle is unused mass, not an independent deployed decision.
        for i, qi in self.charge_index.items():
            self.upper[qi] = self.qbar[i]
        if fallback:
            self.upper[self.slack_start:] = np.inf
        self.binary_indices = list(range(count)) + list(range(self.reject_start, self.dimension))
        rows, lower, upper, names = [], [], [], []

        def add(name, coefficients, lo=-np.inf, hi=np.inf):
            row = np.zeros(n)
            for j, value in coefficients.items():
                row[j] += value
            rows.append(row)
            lower.append(lo)
            upper.append(hi)
            names.append(name)

        for v in np.flatnonzero(state.busy <= 0):
            ids = [i for i, a in enumerate(self.actions) if a.vehicle == v and a.kind != "idle"]
            add(f"vehicle_exclusivity[{v}]", dict.fromkeys(ids, 1), hi=1)
            coeff = {i: -self.energy[i] / state.battery[v] for i in ids}
            for i in charges:
                if self.actions[i].vehicle == v:
                    coeff[self.charge_index[i]] = cfg["charging_efficiency"] / state.battery[v]
            add(f"soc_bounds[{v}]", coeff, cfg["minimum_soc"]-state.soc[v], 1-state.soc[v])
        for r in range(len(state.request_ids)):
            coeff = {i: 1 for i, a in enumerate(self.actions) if a.kind == "serv" and a.target == r}
            coeff[self.reject_start + r] = 1
            add(f"request_coverage[{r}]", coeff, 1, 1)
        for i, qi in self.charge_index.items():
            add(f"charging_link[{i}]", {qi: 1, i: -self.qbar[i]}, hi=0)
        for s in range(len(state.station_xy)):
            at_station = [i for i in charges if self.actions[i].target == s]
            add(f"station_ports[{s}]", dict.fromkeys(at_station, 1), hi=state.ports[s]-state.occupied[s])
            add(f"station_power[{s}]", {self.charge_index[i]: 1 for i in at_station},
                hi=state.power[s]*cfg["interval_min"]/60-state.committed_kwh[s])
        for z in range(len(state.zone_xy)):
            coeff = {}
            for i, a in enumerate(self.actions):
                if a.kind == "reb" and a.destination_zone != int(state.vehicle_zone[a.vehicle]):
                    value = int(a.destination_zone == z)-int(state.vehicle_zone[a.vehicle] == z)
                    if value:
                        coeff[i] = value
            if fallback:
                coeff[self.slack_start + z] = -1
                coeff[self.slack_start + len(state.zone_xy) + z] = 1
            add(f"net_flow[{z}]", coeff, state.net[z], state.net[z])
        # Exact epigraph of the active-zone mean absolute service-rate disparity.
        service_rates = []
        for z in active_zones:
            denominator = max(1, int(np.sum(state.request_zone == z)))
            service_rates.append({i: 1 / denominator for i, a in enumerate(self.actions)
                                  if a.kind == "serv" and int(state.request_zone[a.target]) == z})
        for k, z in enumerate(active_zones):
            deviation = np.zeros(n)
            for j, rates in enumerate(service_rates):
                for i, value in rates.items():
                    deviation[i] += value * ((1 if j == k else 0) - 1/len(active_zones))
            for sign in (1, -1):
                coeff = {i: sign * value for i, value in enumerate(deviation) if value}
                coeff[self.aux_start + k] = -1
                add(f"fairness_epigraph[{z},{sign}]", coeff, hi=0)
        self.A = np.asarray(rows)
        self.l, self.u, self.names = np.asarray(lower), np.asarray(upper), names
        self.cost_components = np.zeros((7, n))
        requests = max(1, len(state.request_ids))
        vehicles = max(1, len(state.soc))
        for i, a in enumerate(self.actions):
            if a.kind == "serv":
                self.cost_components[0, i] = self.time[i]/requests
            if a.kind != "idle":
                self.cost_components[2, i] = a.distance/vehicles
                self.cost_components[5, i] = 0.001*self.energy[i]/state.battery[a.vehicle]/vehicles
        for r in range(len(state.request_ids)):
            self.cost_components[0, self.reject_start+r] = (cfg["max_wait_min"]-state.wait[r])/requests
            self.cost_components[1, self.reject_start+r] = 1/requests
        for i, qi in self.charge_index.items():
            a = self.actions[i]
            self.cost_components[3, qi] = state.price[a.target]
            self.cost_components[4, qi] = state.carbon[a.target]
            self.cost_components[5, qi] = 0.001*cfg["charging_efficiency"]/state.battery[a.vehicle]/vehicles
        if active_zones:
            self.cost_components[6, self.aux_start:self.slack_start] = 1/len(active_zones)
        maxima = np.asarray(cfg["normalization_max"])
        self.cost_vector = np.asarray(cfg["alpha"]) @ (self.cost_components / (maxima[:, None] + 1e-6*np.maximum(1,maxima[:, None])))
        self.cost_constant = cfg["alpha"][0]*float(np.mean(state.wait) if len(state.wait) else 0)/(maxima[0]+1e-6*max(1,maxima[0]))
        self.output = output

    def operating_cost(self, full):
        if isinstance(full, torch.Tensor):
            cost = torch.as_tensor(self.cost_vector, dtype=full.dtype, device=full.device)
            return full @ cost + self.cost_constant
        return float(np.asarray(full) @ self.cost_vector + self.cost_constant)

    def violation_surrogate(self, proposal):
        """Fairness epigraph is an objective device, not an operational constraint."""
        full = torch.cat((proposal, proposal.new_zeros(self.total_dimension-self.dimension)))
        A = torch.as_tensor(self.A, dtype=full.dtype)
        values = A @ full
        terms = []
        for i, name in enumerate(self.names):
            if name.startswith("fairness"):
                continue
            if np.isfinite(self.l[i]):
                terms.append(torch.relu(self.l[i]-values[i]))
            if np.isfinite(self.u[i]):
                terms.append(torch.relu(values[i]-self.u[i]))
        return torch.stack(terms).sum()

    def certificate(self, full):
        values = self.A @ full
        residual = np.maximum(np.maximum(self.l-values, values-self.u), 0)
        bounds = np.maximum(np.maximum(self.lower-full, full-self.upper), 0)
        integrality = max((abs(full[i]-round(full[i])) for i in self.binary_indices), default=0)
        unchanged = [i for i, name in enumerate(self.names) if not name.startswith("net_flow")]
        original_flow = []
        for z in range(len(self.state.zone_xy)):
            net = 0
            for i, a in enumerate(self.actions):
                origin = int(self.state.vehicle_zone[a.vehicle])
                if a.kind == "reb" and a.destination_zone != origin:
                    net += full[i]*(int(a.destination_zone == z)-int(origin == z))
            original_flow.append(float(net-self.state.net[z]))
        soc = self.state.soc.copy()
        for i, a in enumerate(self.actions):
            soc[a.vehicle] -= self.energy[i]*full[i]/self.state.battery[a.vehicle]
            if i in self.charge_index:
                soc[a.vehicle] += self.cfg["charging_efficiency"]*full[self.charge_index[i]]/self.state.battery[a.vehicle]
        result = {"maximum_modeled_constraint_violation": float(max(np.max(residual, initial=0), np.max(bounds, initial=0))),
                  "maximum_unchanged_constraint_violation": float(max(np.max(residual[unchanged], initial=0), np.max(bounds, initial=0))),
                  "integrality_error": float(integrality), "minimum_predicted_soc": float(min(soc)),
                  "original_net_flow_residual": original_flow,
                  "constraints": {name: {"value": float(value), "violation": float(error)}
                                  for name, value, error in zip(self.names, values, residual)}}
        if self.fallback:
            zones = len(self.state.zone_xy)
            result["net_slack_positive"] = full[self.slack_start:self.slack_start+zones].tolist()
            result["net_slack_negative"] = full[self.slack_start+zones:].tolist()
        return result

    def matrices(self, proposal, beta=None):
        beta = self.cfg["projection_beta"] if beta is None else beta
        H = np.zeros(self.total_dimension)
        H[:self.dimension] = 2*self.weight
        linear = beta*self.cost_vector.copy()
        linear[:self.dimension] -= 2*self.weight*np.asarray(proposal)
        if self.fallback:
            linear[self.slack_start:] += self.cfg["fallback_penalty"]
        A = np.vstack((self.A, np.eye(self.total_dimension)))
        return H, linear, A, np.r_[self.l,self.lower], np.r_[self.u,self.upper]

    def solve_relaxed(self, proposal, beta=None):
        H, linear, A, lower, upper = self.matrices(proposal, beta)
        solver = osqp.OSQP()
        solver.setup(P=sp.diags(H, format="csc"), q=linear, A=sp.csc_matrix(A), l=lower, u=upper,
                     eps_abs=1e-8, eps_rel=1e-8, max_iter=100000, polishing=True, verbose=False)
        result = solver.solve(raise_error=False)
        if result.info.status_val not in (1, 2):
            raise RuntimeError(f"Continuous projection failed: {result.info.status}")
        return result.x, (H, A, lower, upper, result.y)

    def solve_integer(self, proposal, teacher=False, time_limit=None):
        """SCIP solves the MIQP in its own process (Windows OpenMP isolation)."""
        import json
        import subprocess
        import sys
        from pathlib import Path
        initial = np.zeros(self.total_dimension)
        initial[self.reject_start:self.dimension] = 1
        if self.fallback:
            zones = len(self.state.zone_xy)
            initial[self.slack_start:self.slack_start+zones] = np.maximum(-self.state.net,0)
            initial[self.slack_start+zones:] = np.maximum(self.state.net,0)
        solver_cfg = self.cfg.copy()
        if time_limit is not None:
            solver_cfg["fallback_time_seconds" if self.fallback else "solver_time_seconds"] = max(.001,time_limit)
        payload = dict(cfg=solver_cfg, lower=self.lower.tolist(), upper=self.upper.tolist(),
                       binary=self.binary_indices, A=self.A.tolist(), l=self.l.tolist(), u=self.u.tolist(),
                       names=self.names, cost=self.cost_vector.tolist(), weight=self.weight.tolist(),
                       proposal=np.asarray(proposal).tolist(), dimension=self.dimension, action_count=len(self.actions),
                       fallback=self.fallback, teacher=teacher, slack_start=self.slack_start, initial=initial.tolist())
        command = [sys.executable,"-m","xgdop.scip_worker"]
        process = subprocess.run(command,input=json.dumps(payload),text=True,capture_output=True,
                                 cwd=Path(__file__).resolve().parent.parent,timeout=30+self.cfg["solver_time_seconds"])
        if process.returncode:
            raise RuntimeError(f"SCIP worker failed: {process.stderr[-2000:]}")
        record = json.loads(process.stdout)
        if record["solution"] is None:
            return None
        full = np.asarray(record["solution"])
        certificate = self.certificate(full)
        certificate["lexicographic_complete"] = record["lex_complete"]
        if certificate["maximum_modeled_constraint_violation"]>2e-6 or certificate["integrality_error"]>2e-6:
            raise RuntimeError("SCIP incumbent failed the independent feasibility audit.")
        return ProjectionResult(full[:self.dimension],full,record["status"],self.fallback,
                                record["objective"],record["gap"],certificate)
    def project(self, proposal):
        proposal = np.asarray(proposal, dtype=float)
        result = self.solve_integer(proposal)
        if result is not None:
            return result, self
        fallback = FleetProblem(self.state, self.output, self.cfg, fallback=True)
        result = fallback.solve_integer(proposal)
        if result is None:
            raise RuntimeError("Fallback invariant failed: check initial SOC and committed charging capacities.")
        return result, fallback


class _RelaxedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proposal, problem):
        solution, (H, A, lower, upper, dual) = problem.solve_relaxed(proposal.detach().double().cpu().numpy())
        values = A @ solution
        equality = np.isfinite(lower) & np.isfinite(upper) & (np.abs(lower-upper) < 1e-10)
        binding = (np.isclose(values,lower,atol=2e-6,rtol=0) | np.isclose(values,upper,atol=2e-6,rtol=0)) & (np.abs(dual)>1e-7)
        active = A[equality | binding]
        if len(active):
            _, R, permutation = qr(active.T, pivoting=True, mode="economic")
            rank = int(np.sum(np.abs(np.diag(R)) > 1e-9))
            active = active[permutation[:rank]]
        n, m = len(H), len(active)
        kkt = np.block([[np.diag(H), active.T], [active, np.zeros((m,m))]])
        rhs = np.zeros((n+m, problem.dimension))
        rhs[:problem.dimension] = np.diag(2*problem.weight)
        # Least squares handles redundant/weak epigraph degeneracy; the local derivative
        # is asserted only for stable active sets and is checked by finite differences.
        jacobian = np.linalg.lstsq(kkt, rhs, rcond=1e-10)[0][:n]
        ctx.save_for_backward(torch.as_tensor(jacobian, dtype=proposal.dtype, device=proposal.device))
        return torch.as_tensor(solution, dtype=proposal.dtype, device=proposal.device)

    @staticmethod
    def backward(ctx, gradient):
        (jacobian,) = ctx.saved_tensors
        return jacobian.T @ gradient, None


def relaxed_projection(proposal, problem):
    return _RelaxedProjection.apply(proposal, problem)

