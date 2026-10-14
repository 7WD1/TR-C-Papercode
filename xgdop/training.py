"""Prediction/teacher IL/DFL stages and terminal-aware hybrid-proposal PPO."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from .projection import FleetProblem, relaxed_projection
from .environment import FleetEnvironment


def pinball(prediction, target):
    quantiles = prediction.new_tensor([0.1,0.25,0.5,0.75,0.9])
    residual = target[..., None]-prediction
    return torch.maximum(quantiles*residual, (quantiles-1)*residual).sum()


def prediction_loss(output, state, cfg):
    action_target = torch.tensor([[a.travel_min,a.energy_kwh] for a in output["geometry"]],dtype=torch.float32)
    demand = np.bincount(state.request_zone,minlength=len(state.zone_xy)).astype(float)
    supply = np.bincount(state.vehicle_zone[state.busy <= 0],minlength=len(state.zone_xy)).astype(float)
    targets = np.tile(np.stack((demand,supply),-1)[:,None,:],(1,cfg["prediction_horizon"]+1,1))
    # Future labels in the runnable fixture are generator expectations. The current
    # interval uses observed counts. They are not observed real-city trajectories.
    targets[:,1:,0] = cfg["requests_per_step"]*state.demand_scale/len(state.zone_xy)
    return pinball(output["te"],action_target)+pinball(output["ds"],torch.tensor(targets,dtype=torch.float32))


def method_losses(model, state, stage="decision"):
    cfg = model.cfg
    output = model(state,quota=None)  # complete support Q for every learning update.
    problem = FleetProblem(state,output,cfg)
    proposal = model.decode(output,problem)
    losses = {"prediction":prediction_loss(output,state,cfg),
              "concept":F.mse_loss(output["concept"],output["labels"],reduction="sum")/len(output["actions"])}
    zero = proposal.sum()*0
    for name in ("regret","imitation","sparse","stable","faith","violation"):
        losses[name] = zero
    if stage != "prediction":
        teacher = problem.solve_integer(np.zeros(problem.dimension),teacher=True)
        # Fallback states never contribute IL, DFL, faithfulness or projection gradients.
        if teacher is not None:
            target = torch.tensor(teacher.decision,dtype=proposal.dtype)
            ids = problem.binary_indices
            losses["imitation"] = F.binary_cross_entropy(proposal[ids].clamp(1e-6,1-1e-6),target[ids])
            qids = list(problem.charge_index.values())
            if qids:
                losses["imitation"] += F.mse_loss(proposal[qids],target[qids])
            if stage == "decision":
                relaxed = relaxed_projection(proposal,problem)
                # Exact one-step relaxed operating-cost reference, with no distance penalty.
                reference = solve_relaxed_reference(problem)
                losses["regret"] = problem.operating_cost(relaxed)-problem.operating_cost(reference)
                selected = int(output["score"].detach().argmax())
                losses["sparse"] = output["weights"][selected].abs().sum()
                radius = cfg["stability_radius"]
                features = state.history[-1]["features"]
                noise = torch.randn_like(features)
                # Uniform radius in a d-dimensional ball, not a Gaussian perturbation.
                noise = noise/noise.norm().clamp_min(1e-12)*(torch.rand(())**(1/features.numel()))*radius
                shifted = model(state,quota=None,feature_noise=noise)
                if shifted["actions"] == output["actions"]:
                    losses["stable"] = ((output["weights"][selected]-shifted["weights"][selected])**2).sum()
                contributions = (output["weights"][selected]*output["concept"][selected]).abs()
                top = torch.argsort(contributions.detach(),descending=True,stable=True)[:3]
                faith = []
                for k in top:
                    new_score = output["score"].clone()
                    new_score[selected] -= output["weights"][selected,k]*output["concept"][selected,k]
                    deleted = model.decode(output,problem,scores=new_score)
                    changed = relaxed_projection(deleted,problem)
                    delta = (problem.operating_cost(changed)-problem.operating_cost(relaxed)).abs()
                    faith.append(torch.relu(cfg["faith_margin"]-delta))
                losses["faith"] = torch.stack(faith).mean()
                losses["violation"] = problem.violation_surrogate(proposal)
    weights = cfg["loss_weights"]
    total = sum(weights[name]*loss for name,loss in losses.items())
    return total,losses,output,problem


def solve_relaxed_reference(problem):
    from scipy.optimize import linprog
    equality = np.isfinite(problem.l)&np.isfinite(problem.u)&(problem.l==problem.u)
    inequalities, rhs = [], []
    for i in np.flatnonzero(~equality):
        if np.isfinite(problem.u[i]):
            inequalities.append(problem.A[i]); rhs.append(problem.u[i])
        if np.isfinite(problem.l[i]):
            inequalities.append(-problem.A[i]); rhs.append(-problem.l[i])
    result = linprog(problem.cost_vector,A_ub=np.asarray(inequalities),b_ub=np.asarray(rhs),
                     A_eq=problem.A[equality],b_eq=problem.l[equality],
                     bounds=list(zip(problem.lower,problem.upper)),method="highs")
    if not result.success:
        raise RuntimeError(f"Relaxed cost reference failed: {result.message}")
    return torch.tensor(result.x,dtype=torch.float32)


def generalized_advantage(rewards,values,terminals,bootstrap,gamma=0.99,lam=0.95):
    """A terminal episode never bootstraps from its next episode."""
    advantage = np.zeros(len(rewards),dtype=float)
    running = 0.0
    for i in range(len(rewards)-1,-1,-1):
        next_value = bootstrap if i==len(rewards)-1 else values[i+1]
        mask = 1-float(terminals[i])
        delta = rewards[i]+gamma*mask*next_value-values[i]
        running = delta+gamma*lam*mask*running
        advantage[i] = running
    return advantage,advantage+np.asarray(values)


@dataclass
class RolloutItem:
    state: object
    latent: dict
    old_logprob: torch.Tensor
    old_value: float
    reward: float
    terminal: bool
    support: tuple
    projection_context: object


def collect_rollout(model,environment,steps):
    cfg = model.cfg
    if cfg["loss_weights"]["rl"] == 0:
        return [],None
    state = environment.reset()
    records = []
    with torch.no_grad():
        for i in range(steps):
            output = model(state,quota=None)
            problem = FleetProblem(state,output,cfg)
            latent,logprob = model.sample(output)
            proposal = model.decode(output,problem,latent).numpy()
            result,executed_problem = problem.project(proposal)
            reward = -executed_problem.operating_cost(result.full)
            next_state = environment.step(executed_problem,result.decision)
            terminal = i==steps-1 or not np.any(next_state.busy <= 0)
            stored = {key:([a.detach().clone() for a in value] if isinstance(value,list) else value.detach().clone())
                      for key,value in latent.items()}
            records.append(RolloutItem(state.clone(),stored,logprob.detach().clone(),float(output["value"]),
                                       reward,terminal,tuple(output["actions"]),problem))
            state = environment.reset() if terminal and i<steps-1 else next_state
        bootstrap = 0.0 if records[-1].terminal else float(model(state,quota=None)["value"])
    advantage,returns = generalized_advantage([r.reward for r in records],[r.old_value for r in records],
                                               [r.terminal for r in records],bootstrap,cfg["gamma"],cfg["gae_lambda"])
    return records,(torch.tensor(advantage,dtype=torch.float32),torch.tensor(returns,dtype=torch.float32))


def ppo_loss(model,records,targets):
    cfg = model.cfg
    if cfg["loss_weights"]["rl"] == 0:
        return next(model.parameters()).sum()*0,{}
    advantages,returns = targets
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    actor,value,entropy,ratios = [],[],[],[]
    for i,record in enumerate(records):
        output = model(record.state,quota=None,support=record.support)
        if tuple(output["actions"]) != record.support:
            raise RuntimeError("PPO candidate support changed. Freeze predictors/encoder for rollout epochs or recollect.")
        logprob,ent = model.logprob(output,record.latent)
        ratio = torch.exp(logprob-record.old_logprob)
        clipped = ratio.clamp(1-cfg["ppo_clip"],1+cfg["ppo_clip"])
        actor.append(-torch.minimum(ratio*advantages[i],clipped*advantages[i]))
        value.append((output["value"]-returns[i])**2)
        entropy.append(ent)
        ratios.append(ratio)
    actor,value,entropy = torch.stack(actor).mean(),torch.stack(value).mean(),torch.stack(entropy).mean()
    loss = cfg["loss_weights"]["rl"]*(actor+cfg["value_weight"]*value-cfg["entropy_weight"]*entropy)
    return loss,{"actor":float(actor.detach()),"value":float(value.detach()),
                 "entropy":float(entropy.detach()),"ratio_mean":float(torch.stack(ratios).mean().detach())}


def train_smoke(model,optimizer):
    """Run all four stages once; this verifies code paths, not convergence."""
    cfg,history = model.cfg,[]
    for stage in ("prediction","imitation","decision"):
        for iteration in range(cfg["smoke_stage_steps"]):
            state = FleetEnvironment(cfg,seed=cfg["seed"]+iteration).reset()
            optimizer.zero_grad()
            loss,terms,_,_ = method_losses(model,state,stage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            history.append({"stage":stage,"loss":float(loss.detach()),
                            "terms":{k:float(v.detach()) for k,v in terms.items()}})
    for iteration in range(cfg["smoke_stage_steps"]):
        environment = FleetEnvironment(cfg,seed=cfg["seed"]+100+iteration,randomized=True)
        state = environment.reset()
        # Alternate auxiliary DFL and PPO cycles. No predictor/constraint update is
        # interleaved into a frozen PPO cycle; fresh rollout follows this update.
        optimizer.zero_grad(set_to_none=True)
        dfl,terms,_,_ = method_losses(model,state,"decision")
        dfl.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        history.append({"stage":"domain_randomized_auxiliary_dfl","loss":float(dfl.detach())})
        records,targets = collect_rollout(model,environment,cfg["rollout_steps"])
        for epoch in range(cfg["ppo_epochs"] if records else 0):
            optimizer.zero_grad()
            rl,ppo = ppo_loss(model,records,targets)
            loss = rl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            history.append({"stage":"domain_randomized_ppo_fixed_projection_context","loss":float(loss.detach()),"ppo":ppo})
    return history
