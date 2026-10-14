"""SCIP-only worker: isolate incompatible Windows OpenMP DLLs from PyTorch.

JSON in/out over a local pipe. No shell interpolation or temporary credentials.
"""
import json
import sys
import time
import numpy as np
from pyscipopt import Model, quicksum


def solve(data):
    cfg = data["cfg"]
    model = Model("XG-DOP mixed integer quadratic projection")
    model.hideOutput()
    model.setIntParam("parallel/maxnthreads",1)
    model.setIntParam("randomization/randomseedshift",cfg["seed"])
    budget = cfg["fallback_time_seconds"] if data["fallback"] else cfg["solver_time_seconds"]
    model.setRealParam("limits/time",budget)
    model.setRealParam("limits/gap",cfg["solver_relative_gap"])
    model.setRealParam("numerics/feastol",1e-8)
    lower,upper = data["lower"],data["upper"]
    binary = set(data["binary"])
    variables = [model.addVar(name=f"a_{i:04d}",vtype="B" if i in binary else "C",
                              lb=float(lower[i]),ub=float(upper[i]) if np.isfinite(upper[i]) else None)
                 for i in range(len(lower))]
    for row,lo,hi,name in zip(data["A"],data["l"],data["u"],data["names"]):
        expression = quicksum(float(c)*variables[i] for i,c in enumerate(row) if c)
        if lo==hi:
            model.addCons(expression==float(lo),name)
        else:
            if np.isfinite(lo):
                model.addCons(expression>=float(lo),name+"_lower")
            if np.isfinite(hi):
                model.addCons(expression<=float(hi),name+"_upper")
    cost = quicksum(float(c)*variables[i] for i,c in enumerate(data["cost"]) if c)
    if data["teacher"]:
        expression = cost
    else:
        terms = []
        for i in range(data["dimension"]):
            w,p = float(data["weight"][i]),float(data["proposal"][i])
            # b^2=b exactly for a binary coordinate. Charging stays quadratic.
            terms.append(w*((1-2*p)*variables[i]+p*p) if i in binary else w*(variables[i]-p)**2)
        expression = quicksum(terms)+cfg["projection_beta"]*cost
    if data["fallback"]:
        expression += cfg["fallback_penalty"]*quicksum(variables[data["slack_start"]:])
    epigraph = model.addVar(name="quadratic_epigraph",lb=-model.infinity())
    model.addCons(epigraph>=expression,"exact_quadratic_epigraph")
    model.setObjective(epigraph,"minimize")
    if data["fallback"]:
        initial = model.createSol()
        vector = np.asarray(data["initial"])
        for i,var in enumerate(variables):
            model.setSolVal(initial,var,float(vector[i]))
        objective = float(np.dot(data["cost"],vector)) if data["teacher"] else float(np.sum(np.asarray(data["weight"])*(vector[:data["dimension"]]-data["proposal"])**2)+cfg["projection_beta"]*np.dot(data["cost"],vector))
        objective += cfg["fallback_penalty"]*float(vector[data["slack_start"]:].sum())
        model.setSolVal(initial,epigraph,objective+1e-5)
        model.addSol(initial)
    start = time.perf_counter()
    model.optimize()
    if model.getNSols()==0:
        return {"solution":None,"status":str(model.getStatus())}
    best = model.getBestSol()
    full = [float(model.getSolVal(best,var)) for var in variables]
    objective,gap,status = float(model.getObjVal()),float(model.getGap()),str(model.getStatus())
    # If the global primary optimum was proved, resolve binary ties in coordinate
    # order without changing it beyond solver tolerance. Charging is strictly
    # convex once binary assignments are fixed; rejection is fixed by coverage.
    lex_complete = False
    if status=="optimal" and not data["teacher"] and cfg.get("lexicographic",True):
        model.freeTransform()
        model.addCons(epigraph<=objective+1e-8,"fix_primary_optimum")
        lex_complete = True
        for i in range(data["action_count"]):
            remaining = budget-(time.perf_counter()-start)
            if remaining<=0.01:
                lex_complete=False
                break
            model.setRealParam("limits/time",remaining)
            model.setObjective(variables[i],"minimize")
            model.optimize()
            if str(model.getStatus())!="optimal" or model.getNSols()==0:
                lex_complete=False
                break
            best = model.getBestSol()
            full = [float(model.getSolVal(best,var)) for var in variables]
            chosen = round(full[i])
            model.freeTransform()
            model.addCons(variables[i]==chosen,f"lex_fix_{i}")
    return {"solution":full,"objective":objective,"gap":gap,"status":status,"lex_complete":lex_complete}


if __name__=="__main__":
    payload = json.load(sys.stdin)
    print(json.dumps(solve(payload)))
