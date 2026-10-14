"""Small stochastic EV operating environment, with explicit physical units.

This is a Euclidean synthetic simulator, not SUMO or an observed city dataset.
Forecast supervision comes from its known generating process. Vehicles in a trip
are excluded from dispatchable nodes and enter aggregate zone supply instead.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
import torch


RELATIONS = ("road", "association", "access", "grid")
TYPES = ("zone", "vehicle", "request", "station")


@dataclass
class State:
    xy: np.ndarray
    zone_xy: np.ndarray
    station_xy: np.ndarray
    soc: np.ndarray
    battery: np.ndarray
    vehicle_zone: np.ndarray
    busy: np.ndarray
    request_xy: np.ndarray
    request_dest: np.ndarray
    request_zone: np.ndarray
    request_dest_zone: np.ndarray
    wait: np.ndarray
    request_ids: np.ndarray
    ports: np.ndarray
    occupied: np.ndarray
    committed_kwh: np.ndarray
    power: np.ndarray
    queue: np.ndarray
    price: np.ndarray
    carbon: np.ndarray
    net: np.ndarray
    step: int = 0
    demand_scale: float = 1.0
    travel_scale: float = 1.0
    energy_scale: float = 1.0
    history: list = field(default_factory=list)

    def clone(self):
        return deepcopy(self)


@dataclass(frozen=True)
class Action:
    kind: str
    vehicle: int
    target: int
    origin_zone: int
    destination_zone: int
    affected_zone: int
    distance: float
    travel_min: float
    energy_kwh: float


def graph_frame(state: State, cfg):
    """Directed typed edges; both directions share one relation family."""
    features, kinds, keys, index = [], [], [], {}
    zcount = len(state.zone_xy)
    demand = np.bincount(state.request_zone, minlength=zcount).astype(float)
    supply = np.bincount(state.vehicle_zone[state.busy <= 0], minlength=zcount).astype(float)
    committed = np.bincount(state.vehicle_zone[state.busy > 0], minlength=zcount).astype(float)

    def node(kind, ident, values):
        key = (kind, int(ident))
        index[key] = len(features)
        row = np.zeros(10, dtype=np.float32)
        row[:len(values)] = values
        features.append(row)
        kinds.append(TYPES.index(kind))
        keys.append(key)

    for z, xy in enumerate(state.zone_xy):
        node("zone", z, [*xy / 10, demand[z] / 10, supply[z] / 10, state.net[z] / 10, committed[z] / 10])
    for v in np.flatnonzero(state.busy <= 0):
        node("vehicle", v, [*state.xy[v] / 10, state.soc[v], state.battery[v] / 100])
    for r, xy in enumerate(state.request_xy):
        node("request", state.request_ids[r], [*xy / 10, *state.request_dest[r] / 10,
                                               state.wait[r] / cfg["max_wait_min"], 1])
    for s, xy in enumerate(state.station_xy):
        node("station", s, [*xy / 10, state.ports[s] / 10, state.occupied[s] / 10,
                            state.queue[s] / 10, state.power[s] / 350,
                            state.price[s], state.carbon[s] / 1000])
    edges = {rel: [] for rel in RELATIONS}

    def edge(rel, a, b):
        edges[rel].extend([(index[a], index[b]), (index[b], index[a])])

    for z in range(zcount):
        if z + 1 < zcount:
            edge("road", ("zone", z), ("zone", z + 1))
    for v in np.flatnonzero(state.busy <= 0):
        edge("association", ("vehicle", int(v)), ("zone", int(state.vehicle_zone[v])))
    for r, ident in enumerate(state.request_ids):
        for z in sorted({int(state.request_zone[r]), int(state.request_dest_zone[r])}):
            edge("association", ("request", int(ident)), ("zone", z))
    for s, xy in enumerate(state.station_xy):
        z = int(np.argmin(np.linalg.norm(state.zone_xy - xy, axis=1)))
        edge("access", ("zone", z), ("station", s))
        if s + 1 < len(state.station_xy):
            edge("grid", ("station", s), ("station", s + 1))
    return {"features": torch.tensor(np.asarray(features)), "kinds": torch.tensor(kinds),
            "keys": keys, "index": index,
            "edges": {rel: torch.tensor(values, dtype=torch.long).reshape(-1, 2).T
                      for rel, values in edges.items()}}


def geometry_candidates(state, cfg, quota=None):
    """Balanced spatial tree queries run before any learned prediction."""
    from scipy.spatial import cKDTree
    actions = []
    families = (("serv", state.request_xy), ("reb", state.zone_xy), ("chg", state.station_xy))
    trees = {kind: cKDTree(points) for kind, points in families if len(points)}
    for v in np.flatnonzero(state.busy <= 0):
        origin = int(state.vehicle_zone[v])
        for kind, points in families:
            if not len(points):
                continue
            count = len(points) if quota is None else min(quota, len(points))
            _, near = trees[kind].query(state.xy[v], k=count)
            targets = set(np.atleast_1d(near).astype(int).tolist())
            if kind == "reb":
                targets.update(np.flatnonzero(state.net != 0).tolist())
            for target in sorted(targets):
                empty = float(np.linalg.norm(state.xy[v] - points[target]))
                trip = float(np.linalg.norm(state.request_xy[target] - state.request_dest[target])) if kind == "serv" else 0
                travel = empty / cfg["speed_km_per_min"] * state.travel_scale
                energy = (empty + trip) * cfg["energy_kwh_per_km"] * state.energy_scale
                oz = int(state.request_zone[target]) if kind == "serv" else origin
                dz = int(state.request_dest_zone[target]) if kind == "serv" else (
                    target if kind == "reb" else int(np.argmin(np.linalg.norm(state.zone_xy - points[target], axis=1))))
                affected = oz if kind == "serv" else dz
                actions.append(Action(kind, int(v), target, oz, dz, affected, empty, travel, energy))
        actions.append(Action("idle", int(v), -1, origin, origin, origin, 0, 0, 0))
    return actions


def action_features(actions, state, cfg):
    rows = []
    for a in actions:
        indicator = [float(a.kind == kind) for kind in ("serv", "reb", "chg", "idle")]
        row = indicator + [a.distance / 10, a.travel_min / 10, a.energy_kwh / 10,
                           state.soc[a.vehicle], state.battery[a.vehicle] / 100,
                           state.wait[a.target] / cfg["max_wait_min"] if a.kind == "serv" else 0]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 10)


class FleetEnvironment:
    def __init__(self, cfg, seed=None, randomized=False):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
        self.randomized = randomized
        self.state = None

    def _requests(self, step, zone_xy, demand_scale):
        n = max(1, int(round(self.cfg["requests_per_step"] * demand_scale)))
        origin = self.rng.integers(0, len(zone_xy), n)
        dest = self.rng.integers(0, len(zone_xy), n)
        xy = zone_xy[origin] + self.rng.normal(0, 0.2, (n, 2))
        dxy = zone_xy[dest] + self.rng.normal(0, 0.2, (n, 2))
        return xy, dxy, origin, dest, step * 1000 + np.arange(n)

    def reset(self):
        cfg = self.cfg
        zxy = np.column_stack((np.linspace(0, 4, cfg["zones"]), np.zeros(cfg["zones"])))
        sxy = zxy[np.arange(cfg["stations"]) % cfg["zones"]] + [0.1, 0.1]
        zones = np.arange(cfg["vehicles"]) % cfg["zones"]
        dr = cfg["domain_randomization"]
        demand = self.rng.uniform(*dr["demand_scale"]) if self.randomized else 1
        travel = self.rng.uniform(*dr["travel_scale"]) if self.randomized else 1
        energy = self.rng.uniform(*dr["energy_scale"]) if self.randomized else 1
        xy, dxy, oz, dz, ids = self._requests(0, zxy, demand)
        ports = np.full(cfg["stations"], cfg["charger_ports"], dtype=int)
        if self.randomized:
            probability = self.rng.uniform(*dr["station_failure_probability"])
            ports[self.rng.random(len(ports)) < probability] = 0
        self.state = State(zxy[zones] + self.rng.normal(0, 0.1, (len(zones), 2)), zxy, sxy,
                           self.rng.uniform(0.25, 0.8, len(zones)),
                           np.resize(np.array(cfg["battery_kwh"]), len(zones)), zones,
                           np.zeros(len(zones)), xy, dxy, oz, dz, np.zeros(len(ids)), ids,
                           ports, np.zeros(len(ports)), np.zeros(len(ports)),
                           np.full(len(ports), cfg["station_power_kw"]), np.zeros(len(ports)),
                           np.linspace(0.2, 0.4, len(ports)), np.linspace(250, 600, len(ports)),
                           np.zeros(cfg["zones"]), demand_scale=demand, travel_scale=travel, energy_scale=energy)
        frame = graph_frame(self.state, cfg)
        self.state.history = [frame] * cfg["history"]
        return self.state.clone()

    def step(self, problem, decision):
        """Execute an integer projection, move state, and draw new demand."""
        state, cfg = self.state, self.cfg
        state.busy = np.maximum(0, state.busy - cfg["interval_min"])
        for i, a in enumerate(problem.actions):
            if a.kind == "idle" or decision[i] < 0.5:
                continue
            v = a.vehicle
            # Evaluation energy comes from the physical simulator, not a neural coefficient.
            state.soc[v] -= a.energy_kwh / state.battery[v]
            if a.kind == "serv":
                state.xy[v] = state.request_dest[a.target]
                state.vehicle_zone[v] = state.request_dest_zone[a.target]
                trip = np.linalg.norm(state.request_dest[a.target] - state.request_xy[a.target])
                state.busy[v] = max(0, a.travel_min + trip / cfg["speed_km_per_min"] - cfg["interval_min"])
            elif a.kind == "reb":
                state.xy[v] = state.zone_xy[a.target]
                state.vehicle_zone[v] = a.target
            else:
                q = decision[problem.charge_index[i]]
                state.soc[v] += cfg["charging_efficiency"] * q / state.battery[v]
                state.xy[v] = state.station_xy[a.target]
                state.vehicle_zone[v] = a.destination_zone
        state.step += 1
        xy, dxy, oz, dz, ids = self._requests(state.step, state.zone_xy, state.demand_scale)
        state.request_xy, state.request_dest = xy, dxy
        state.request_zone, state.request_dest_zone, state.request_ids = oz, dz, ids
        state.wait = np.zeros(len(ids))
        state.history = (state.history + [graph_frame(state, cfg)])[-cfg["history"]:]
        return state.clone()
