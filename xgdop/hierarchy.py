"""Nested geometry queries, per-zone budgets, score retention and master solve."""
import numpy as np
import time
from .projection import FleetProblem


def hierarchical_project(model,state):
    cfg = model.cfg
    retained = set()
    rounds = []
    primary_deadline = time.perf_counter()+cfg["solver_time_seconds"]
    for q in range(cfg["expansion_rounds"]+1):
        final = q==cfg["expansion_rounds"]
        quota = None if final else cfg["geometry_quota"]*2**q
        output = model(state,quota=quota)
        actions = output["actions"]
        if final:
            retained.update(actions)
        else:
            scores = output["score"].detach().numpy()
            probabilities = output["probability"].detach().numpy()
            # Zone budgets use rounded soft action mass and grow on infeasibility.
            budgets = {}
            for kind in ("serv","reb","chg"):
                for z in range(len(state.zone_xy)):
                    ids = [i for i,a in enumerate(actions) if a.kind==kind and a.affected_zone==z]
                    mass = float(probabilities[ids].sum())
                    budgets[(kind,z)] = min(len(ids),max(1,int(round(mass))*2**q))
            for vehicle in sorted({a.vehicle for a in actions}):
                for kind in ("serv","reb","chg"):
                    ids = [i for i,a in enumerate(actions) if a.vehicle==vehicle and a.kind==kind]
                    ids.sort(key=lambda i:(-scores[i],actions[i].target))
                    for i in ids[:cfg["score_quota"]*2**q]:
                        a = actions[i]
                        if budgets[(kind,a.affected_zone)]>0:
                            retained.add(a)
                            budgets[(kind,a.affected_zone)]-=1
            # Geometry-only flow-support rebalancing arcs always survive scoring
            # if the physical predictor screening admits them. This covers all
            # target zones; the final round guarantees full support recovery.
            for a in actions:
                if a.kind=="idle" or (a.kind=="reb" and (state.net[a.destination_zone]!=0 or state.net[state.vehicle_zone[a.vehicle]]!=0)):
                    retained.add(a)
        # Recompute softmax on cumulative retained support, never reuse an old
        # denominator. Retained actions were geometrically available in this round.
        support = [a for a in actions if a in retained]
        output = model(state,quota=quota,support=support)
        problem = FleetProblem(state,output,cfg)
        proposal = model.decode(output,problem).detach().numpy()
        remaining = primary_deadline-time.perf_counter()
        result = problem.solve_integer(proposal,time_limit=remaining) if remaining>.01 else None
        rounds.append({"round":q,"geometry_count":len(output["geometry"]),
                       "retained_count":len(output["actions"]),"feasible_incumbent":result is not None})
        if result is not None:
            return result,problem,output,rounds
        if time.perf_counter()>=primary_deadline:
            break
    # No repeated final standard solve: all sparse rounds share one primary
    # budget. Only the separately reserved fallback budget remains.
    problem = FleetProblem(state,output,cfg,fallback=True)
    result = problem.solve_integer(proposal)
    if result is None:
        raise RuntimeError("Fallback invariant failed under the cached sparse support.")
    return result,problem,output,rounds
