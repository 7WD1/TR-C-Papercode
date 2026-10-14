"""Intrinsic contributions, optimized edge-mask audit and finite-grid counterfactual.

Edge-mask attribution is an explicit GNNExplainer-style audit. It is not a claim
to reproduce the manuscript's PGExplainer/SHAP comparison benchmarks.
"""
import numpy as np
import torch
from .projection import FleetProblem

CONCEPTS = ("Demand pressure","Service gap","SOC risk","Charger congestion",
            "Pickup delay","Traction energy","Grid carbon")


def fixed_score_shap(output,selected=None):
    """Exact 2^7-coalition Shapley audit for the fixed-weight additive score.

    The game fixes the candidate, learned weights and residual. This audit is
    deliberately local to its score; projected-action SHAP requires rerunning the
    complete optimizer and is not equated with these seven numbers.
    """
    from math import factorial
    selected = int(output["score"].detach().argmax()) if selected is None else selected
    contribution = (output["concept"][selected]*output["weights"][selected]).detach().numpy()
    baseline = float(output["residual"][selected].detach())
    games = {mask:baseline+sum(contribution[k] for k in range(7) if mask & (1<<k)) for mask in range(128)}
    shapley = []
    for k in range(7):
        value = 0.0
        for mask in range(128):
            if mask & (1<<k):
                continue
            size = mask.bit_count()
            weight = factorial(size)*factorial(6-size)/factorial(7)
            value += weight*(games[mask | (1<<k)]-games[mask])
        shapley.append(float(value))
    return {"baseline":baseline,"values":dict(zip(CONCEPTS,shapley)),"coalitions":128}


class ParametricEdgeExplainer(torch.nn.Module):
    """PGExplainer-style learned edge gate with a proposal-fidelity objective."""
    def __init__(self,width):
        super().__init__()
        self.gate = torch.nn.Sequential(torch.nn.Linear(2*width+4,width),torch.nn.ReLU(),torch.nn.Linear(width,1))

    def masks(self,model,state):
        from .environment import RELATIONS
        with torch.no_grad():
            h,_,_,_ = model.encode(state.history)
        masks = {}
        for k,relation in enumerate(RELATIONS):
            source,target = state.history[-1]["edges"][relation]
            indicator = h.new_zeros(len(source),4)
            indicator[:,k] = 1
            pairs = torch.cat((h[source],h[target],indicator),-1)
            masks[relation] = self.gate(pairs).squeeze(-1).sigmoid()
        return masks

    def fit(self,model,states,epochs=10):
        optimizer = torch.optim.Adam(self.parameters(),lr=0.01)
        parameters = list(model.parameters())
        flags = [p.requires_grad for p in parameters]
        for parameter in parameters:
            parameter.requires_grad_(False)
        try:
            for _ in range(epochs):
                optimizer.zero_grad()
                losses = []
                for state in states:
                    original = model(state)
                    masks = self.masks(model,state)
                    explained = model(state,masks=masks,support=original["actions"])
                    gates = torch.cat(list(masks.values())).clamp(1e-6,1-1e-6)
                    fidelity = ((explained["probability"]-original["probability"].detach())**2).mean()
                    entropy = -(gates*gates.log()+(1-gates)*(1-gates).log()).mean()
                    losses.append(fidelity+0.005*gates.mean()+0.001*entropy)
                torch.stack(losses).mean().backward()
                optimizer.step()
        finally:
            for parameter,flag in zip(parameters,flags):
                parameter.requires_grad_(flag)
        return self


def intrinsic(output,selected=None):
    selected = int(output["score"].detach().argmax()) if selected is None else selected
    contribution = output["concept"][selected]*output["weights"][selected]
    a = output["actions"][selected]
    return {"candidate":{"type":a.kind,"vehicle":a.vehicle,"target":a.target},
            "concepts":dict(zip(CONCEPTS,contribution.detach().tolist())),
            "residual":float(output["residual"][selected].detach()),
            "score":float(output["score"][selected].detach()),
            "deletion_identity_max_error":float(((output["score"][selected]-
                (output["score"][selected]-contribution))-contribution).abs().max().detach())}


def edge_mask_audit(model,state,iterations=12):
    original = model(state)
    reference = original["probability"].detach()
    edges = state.history[-1]["edges"]
    logits = {name:torch.nn.Parameter(torch.full((edge.shape[1],),2.0)) for name,edge in edges.items()}
    optimizer = torch.optim.Adam(list(logits.values()),lr=0.1)
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    try:
        for _ in range(iterations):
            optimizer.zero_grad()
            masks = {name:value.sigmoid() for name,value in logits.items()}
            output = model(state,masks=masks,support=original["actions"])
            error = ((output["probability"]-reference)**2).mean()
            sparse = torch.cat(list(masks.values())).mean()
            loss = error+0.005*sparse
            loss.backward()
            optimizer.step()
        records = []
        for relation,edge in edges.items():
            for k,(source,target) in enumerate(edge.T.tolist()):
                records.append({"relation":relation,"source":state.history[-1]["keys"][source],
                                "target":state.history[-1]["keys"][target],
                                "mask":float(logits[relation][k].sigmoid().detach())})
        return sorted(records,key=lambda row:-row["mask"])
    finally:
        for p,flag in zip(parameters,flags):
            p.requires_grad_(flag)


def counterfactual_grid(model,state,target_binary,fields=None,charge_target=None,charge_tolerance=0.1):
    """Exact minimum among an explicit finite admissible intervention grid.

    Default operator-controllable field: prescribed zone net-rebalancing target.
    Identity/topology are fixed. This routine does not claim a continuous global
    counterfactual optimum. The grid, standardizing scales and feasibility status
    are returned so the result can be audited.
    """
    cfg = model.cfg
    if fields is None:
        fields = [("net",np.zeros(len(state.zone_xy)))]
    best = None
    with torch.no_grad():
        for name,value in fields:
            if name not in ("net","price","power"):
                raise ValueError("Only declared operator-controllable fields are admissible.")
            candidate = state.clone()
            original = np.asarray(getattr(candidate,name))
            value = np.asarray(value,dtype=float)
            if value.shape != original.shape or (name in ("price","power") and np.any(value<0)):
                continue
            setattr(candidate,name,value)
            # Rebuild the last feature frame when the intervention affects the graph.
            from .environment import graph_frame
            candidate.history[-1] = graph_frame(candidate,cfg)
            output = model(candidate)
            problem = FleetProblem(candidate,output,cfg)
            proposal = model.decode(output,problem).numpy()
            result,executed = problem.project(proposal)
            binary = np.rint(result.decision[executed.binary_indices]).astype(int)
            if not np.array_equal(binary,target_binary):
                continue
            q = result.decision[list(executed.charge_index.values())]
            if charge_target is not None and (q.shape != np.shape(charge_target) or np.max(np.abs(q-charge_target),initial=0)>charge_tolerance):
                continue
            # Demonstration standardizing scales are fixed and exposed, not fitted.
            scale = {"net":1.0,"price":0.2,"power":100.0}[name]
            distance = float(np.abs(value-original).sum()/scale)
            if best is None or distance<best["standardized_l1"]:
                best = {"field":name,"value":value.tolist(),"standardized_l1":distance,
                        "scale":scale,"fallback":result.fallback,"valid":True}
    return best or {"valid":False,"reason":"No valid counterfactual in the declared finite grid."}
