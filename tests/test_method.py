"""Focused tests for feasibility, projection derivatives and PPO accounting."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from xgdop.environment import FleetEnvironment
from xgdop.model import ActorCritic
from xgdop.projection import FleetProblem,relaxed_projection
from xgdop.training import generalized_advantage,collect_rollout,ppo_loss,method_losses
from xgdop.explanations import fixed_score_shap
from xgdop.hierarchy import hierarchical_project


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/demo.json").read_text())
    state = FleetEnvironment(cfg).reset()
    model = ActorCritic(cfg)
    output = model(state)
    problem = FleetProblem(state,output,cfg)
    return cfg,state,model,output,problem


def test_actual_miqp_enforces_joint_constraints(setup):
    cfg,state,model,output,problem = setup
    # Deliberately propose conflicts, excess station energy and no rejections.
    proposal = np.ones(problem.dimension)
    proposal[problem.reject_start:] = 0
    for i,qi in problem.charge_index.items():
        proposal[qi] = problem.qbar[i]
    result,executed = problem.project(proposal)
    cert = result.certificate
    assert not result.fallback
    assert cert["maximum_modeled_constraint_violation"]<2e-6
    assert cert["integrality_error"]<2e-6
    assert cert["minimum_predicted_soc"]>=cfg["minimum_soc"]-2e-6
    assert max(map(abs,cert["original_net_flow_residual"]))<2e-6
    # An independent audit of request coverage and per-vehicle exclusivity.
    for r in range(len(state.request_ids)):
        served = sum(result.decision[i] for i,a in enumerate(executed.actions) if a.kind=="serv" and a.target==r)
        assert served+result.decision[executed.reject_start+r]==pytest.approx(1,abs=2e-6)
    for v in range(len(state.soc)):
        assert sum(result.decision[i] for i,a in enumerate(executed.actions) if a.vehicle==v)<=1+2e-6


def test_only_net_targets_are_relaxed_in_fallback(setup):
    cfg,state,model,_,_ = setup
    state.net[:] = [1,0,0]  # Impossible because net rebalancing sums to zero.
    output = model(state)
    problem = FleetProblem(state,output,cfg)
    result,executed = problem.project(model.decode(output,problem).detach().numpy())
    assert result.fallback
    assert result.certificate["maximum_unchanged_constraint_violation"]<2e-6
    assert abs(sum(result.certificate["original_net_flow_residual"]))==pytest.approx(1,abs=2e-6)
    pos,neg = result.certificate["net_slack_positive"],result.certificate["net_slack_negative"]
    assert sum(pos)+sum(neg)>=1-2e-6
    assert executed.total_dimension-executed.slack_start==2*len(state.zone_xy)


def test_relaxed_kkt_gradient_matches_finite_difference(setup):
    _,_,_,_,problem = setup
    rng = np.random.default_rng(8)
    raw = rng.uniform(0.15,0.75,problem.dimension)
    for i,qi in problem.charge_index.items():
        raw[qi] *= problem.qbar[i]
    proposal = torch.tensor(raw,dtype=torch.float64,requires_grad=True)
    direction = rng.normal(size=problem.dimension)
    direction /= np.linalg.norm(direction)
    objective = rng.normal(size=problem.total_dimension)
    result = relaxed_projection(proposal,problem)
    loss = result @ torch.tensor(objective,dtype=torch.float64)
    loss.backward()
    analytic = float(proposal.grad.numpy() @ direction)
    eps = 1e-4
    plus,_ = problem.solve_relaxed(raw+eps*direction)
    minus,_ = problem.solve_relaxed(raw-eps*direction)
    numerical = float(objective @ (plus-minus)/(2*eps))
    assert analytic==pytest.approx(numerical,abs=1e-4,rel=1e-4)


def test_quantiles_and_typed_gradients(setup):
    _,_,model,output,_ = setup
    assert torch.all(torch.diff(output["te"],dim=-1)>0)
    assert torch.all(torch.diff(output["ds"],dim=-1)>0)
    assert output["concept"].shape[1]==7
    assert output["residual_vector"].shape[1]==4
    for i,a in enumerate(output["actions"]):
        if a.kind!="chg":
            assert output["concept"][i,3]==0 and output["concept"][i,6]==0
        if a.kind!="serv":
            assert output["concept"][i,4]==0
    output["score"].sum().backward()
    for relation in ("road","association","access","grid"):
        assert model.layers[0].transforms[relation].weight.grad is not None
    assert model.gru.weight_hh.grad is not None


def test_terminal_gae_does_not_cross_episode():
    advantage,returns = generalized_advantage([1,2,3],[0.5,0.7,0.9],[False,True,False],0.2,gamma=.99,lam=.95)
    last = 3+.99*.2-.9
    middle = 2-.7  # terminal, so no next-value or carried advantage.
    first = 1+.99*.7-.5+.99*.95*middle
    assert advantage==pytest.approx([first,middle,last])
    assert returns==pytest.approx(advantage+np.array([.5,.7,.9]))


def test_ppo_joint_proposal_ratio_and_loss_sign(setup):
    cfg,_,model,_,_ = setup
    records,targets = collect_rollout(model,FleetEnvironment(cfg),2)
    assert records[-1].terminal
    cached = records[0].projection_context.A.copy()
    loss,terms = ppo_loss(model,records,targets)
    assert terms["ratio_mean"]==pytest.approx(1,abs=1e-6)
    assert terms["actor"]==pytest.approx(0,abs=1e-6)
    assert float(loss.detach())==pytest.approx(.5*terms["value"]-.01*terms["entropy"],abs=2e-6)
    loss.backward()
    assert model.critic[-1].weight.grad is not None
    assert model.concept_weights[0][-1].weight.grad is not None
    assert np.array_equal(cached,records[0].projection_context.A)
    # The saved likelihood is from the sampled latent actor, not the executed MIQP.
    output = model(records[0].state,support=records[0].support)
    logprob,_ = model.logprob(output,records[0].latent)
    assert torch.allclose(logprob.detach(),records[0].old_logprob,atol=1e-6)
    assert len(records[0].latent["fraction"])==len(records[0].projection_context.charge_index)


def test_no_rl_skips_rollout_critic_and_gae(setup):
    cfg,state,model,_,_ = setup
    cfg["loss_weights"]["rl"] = 0
    environment = FleetEnvironment(cfg)
    records,targets = collect_rollout(model,environment,4)
    assert records==[] and targets is None and environment.state is None
    loss,terms = ppo_loss(model,[],None)
    assert float(loss)==0 and terms=={}
    auxiliary,_,_,_ = method_losses(model,state,"prediction")
    auxiliary.backward()
    assert all(parameter.grad is None for parameter in model.critic.parameters())


def test_exhaustive_shap_audits_local_additive_score(setup):
    _,_,_,output,_ = setup
    selected = int(output["score"].argmax())
    result = fixed_score_shap(output,selected)
    assert result["coalitions"]==128
    values = np.array(list(result["values"].values()))
    contributions = (output["concept"][selected]*output["weights"][selected]).detach().numpy()
    assert values==pytest.approx(contributions,abs=1e-8)
    assert result["baseline"]+values.sum()==pytest.approx(float(output["score"][selected]),abs=1e-6)


def test_hierarchy_retains_flow_support_and_feasibility(setup):
    cfg,state,model,_,_ = setup
    cfg["geometry_quota"],cfg["score_quota"] = 1,1
    state.net[:] = [1,-1,0]
    result,problem,output,rounds = hierarchical_project(model,state)
    assert not result.fallback
    assert result.certificate["maximum_modeled_constraint_violation"]<2e-6
    assert rounds[-1]["feasible_incumbent"]


def test_beta_latent_draw_is_not_artificially_clamped(setup,monkeypatch):
    _,_,model,output,_ = setup
    from torch.distributions import Beta
    monkeypatch.setattr(Beta,"sample",lambda self,sample_shape=torch.Size(): self.concentration1.new_full(self.concentration1.shape,1e-8))
    latent,logprob = model.sample(output)
    assert torch.all(latent["fraction"]==1e-8)
    assert torch.isfinite(logprob)
    repeated,_ = model.logprob(output,latent)
    assert torch.equal(logprob,repeated)


def test_empty_request_set_remains_runnable(setup):
    cfg,state,model,_,_ = setup
    from xgdop.environment import graph_frame
    for name in ("request_xy","request_dest"):
        setattr(state,name,np.empty((0,2)))
    for name in ("request_zone","request_dest_zone","request_ids"):
        setattr(state,name,np.empty(0,dtype=int))
    state.wait = np.empty(0)
    state.history[-1] = graph_frame(state,cfg)
    output = model(state)
    assert output["reject_probability"].shape==(0,)
    problem = FleetProblem(state,output,cfg)
    proposal = model.decode(output,problem)
    result,executed = problem.project(proposal.detach().numpy())
    assert not result.fallback
    assert result.certificate["maximum_modeled_constraint_violation"]<2e-6


def test_counterfactual_rejects_equal_length_different_candidate_identities(setup):
    cfg,state,_,output,_ = setup
    from xgdop.explanations import counterfactual_grid
    def service(v,r):
        return next(a for a in output["actions"] if a.kind=="serv" and a.vehicle==v and a.target==r)
    def idle(v):
        return next(a for a in output["actions"] if a.kind=="idle" and a.vehicle==v)
    baseline = [service(0,0),service(1,1),idle(0),idle(1)]
    changed = [service(0,1),service(1,0),idle(0),idle(1)]

    class ScreenedModel:
        def __init__(self):
            self.cfg = cfg
        def __call__(self,candidate):
            support = changed if candidate.price[0]>state.price[0] else baseline
            return {"actions":support,"time":torch.tensor([1.,1.,0.,0.]),
                    "energy":torch.tensor([.1,.1,0.,0.])}
        def decode(self,result,problem):
            proposal = torch.zeros(problem.dimension)
            proposal[:2] = 1
            proposal[problem.reject_start+2:] = 1
            return proposal

    model = ScreenedModel()
    changed_state = state.clone()
    changed_state.price[0] += .1
    def project(candidate):
        candidate_output = model(candidate)
        problem = FleetProblem(candidate,candidate_output,cfg)
        result,executed = problem.project(model.decode(candidate_output,problem).numpy())
        return result,executed
    original_result,original_problem = project(state)
    changed_result,changed_problem = project(changed_state)
    target = np.rint(original_result.decision[original_problem.binary_indices]).astype(int)
    alternative = np.rint(changed_result.decision[changed_problem.binary_indices]).astype(int)
    # Both vectors are genuinely feasible and numerically equal, but the request
    # assigned to each vehicle is different. A length/raw-vector check fails.
    assert np.array_equal(target,alternative)
    assert baseline[0].target!=changed[0].target
    assert original_result.certificate["maximum_modeled_constraint_violation"]<2e-6
    assert changed_result.certificate["maximum_modeled_constraint_violation"]<2e-6
    result = counterfactual_grid(model,state,target,fields=[("price",changed_state.price)],
                                 target_support=baseline)
    assert not result["valid"]
    assert result["candidate_identity_policy"]=="matched_ordered_support"
    assert result["skipped_support_mismatches"]==1
