"""One-command CPU execution, with plots and auditable JSON outputs."""
import argparse
import csv
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .environment import FleetEnvironment
from .model import ActorCritic
from .projection import FleetProblem
from .training import train_smoke
from .explanations import intrinsic,edge_mask_audit,fixed_score_shap,ParametricEdgeExplainer,counterfactual_grid
from .hierarchy import hierarchical_project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/demo.json")
    parser.add_argument("--output",default="outputs/smoke")
    parser.add_argument("--no-train",action="store_true")
    parser.add_argument("--no-rl",action="store_true")
    parser.add_argument("--hierarchical",action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.no_rl:
        cfg["loss_weights"]["rl"] = 0.0
    torch.set_num_threads(1)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True,exist_ok=True)
    model = ActorCritic(cfg)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    started = time.perf_counter()
    history = [] if args.no_train else train_smoke(model,optimizer)
    print(f"Completed training code paths: {len(history)} updates.",flush=True)
    environment = FleetEnvironment(cfg)
    state = environment.reset()
    snapshots,reports = [],[]
    with torch.no_grad():
        for step in range(4):
            if not np.any(state.busy <= 0):
                state = environment.reset()
            if args.hierarchical:
                result,problem,output,rounds = hierarchical_project(model,state)
            else:
                output = model(state)
                problem = FleetProblem(state,output,cfg)
                proposal = model.decode(output,problem).numpy()
                result,problem = problem.project(proposal)
                rounds = []
            snapshots.append((state.clone(),result,problem))
            reports.append({"step":step,"solver":result.status,"gap":result.gap,"fallback":result.fallback,
                            "operating_cost":problem.operating_cost(result.full),
                            "certificate":result.certificate,"candidate_rounds":rounds,
                            "intrinsic":intrinsic(output),"fixed_score_shap":fixed_score_shap(output),
                            **serialize_execution(state,result,problem)})
            state = environment.step(problem,result.decision)
            for vehicle in reports[-1]["vehicle_soc"]:
                vehicle["physical_after"] = float(state.soc[vehicle["vehicle"]])
    first_state,first_result,first_problem = snapshots[0]
    mask = edge_mask_audit(model,first_state,iterations=8)
    pg = ParametricEdgeExplainer(cfg["hidden_dim"]).fit(model,[first_state],epochs=3)
    pg_masks = {name:value.detach().tolist() for name,value in pg.masks(model,first_state).items()}
    alternative = first_state.clone()
    alternative.net[:] = 0
    alternative.net[0],alternative.net[1] = 1,-1
    from .environment import graph_frame
    alternative.history[-1] = graph_frame(alternative,cfg)
    with torch.no_grad():
        alternative_output = model(alternative)
        alternative_problem = FleetProblem(alternative,alternative_output,cfg)
        alternative_result,alternative_problem = alternative_problem.project(model.decode(alternative_output,alternative_problem).numpy())
    target = np.rint(alternative_result.decision[alternative_problem.binary_indices]).astype(int)
    counterfactual = counterfactual_grid(model,first_state,target,
                        fields=[("net",first_state.net.copy()),("net",alternative.net.copy()),("net",-alternative.net.copy())],
                        charge_target=alternative_result.decision[list(alternative_problem.charge_index.values())],
                        target_support=alternative_problem.actions)
    # Separate intentionally inconsistent net target demonstrates reported fallback.
    bad = first_state.clone()
    bad.net[0] = 1  # Sum of net flow must be zero; this target is impossible.
    with torch.no_grad():
        bad_output = model(bad)
        bad_problem = FleetProblem(bad,bad_output,cfg)
        fallback,fbproblem = bad_problem.project(model.decode(bad_output,bad_problem).numpy())
    report = {"scope":"Synthetic method smoke run; no manuscript benchmark numbers.",
              "versions":{"python":platform.python_version(),"torch":torch.__version__},
              "config":cfg,"training":history,"closed_loop":reports,"edge_mask_audit":mask,
              "parametric_edge_mask":pg_masks,"counterfactual_grid_check":counterfactual,
              "intentional_fallback":{"active":fallback.fallback,"certificate":fallback.certificate},
              "wall_time_seconds":time.perf_counter()-started}
    (output_dir/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (output_dir/"training.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=["update","stage","loss"])
        writer.writeheader()
        for i,row in enumerate(history):
            writer.writerow({"update":i+1,"stage":row["stage"],"loss":row["loss"]})
    torch.save({"state_dict":model.state_dict(),"config":cfg},output_dir/"checkpoint.pt")
    make_plots(snapshots,reports,output_dir)
    print(json.dumps({"output":str(output_dir.resolve()),"closed_loop_steps":len(reports),
                      "maximum_modeled_violation":max(r["certificate"]["maximum_modeled_constraint_violation"] for r in reports),
                      "fallback_active":fallback.fallback,"rl_enabled":cfg["loss_weights"]["rl"]>0,
                      "wall_time_seconds":round(report["wall_time_seconds"],2)},indent=2))


def serialize_execution(state,result,problem):
    """Persist the executed action and state accounting without changing it."""
    decision = result.decision
    predicted_soc = state.soc.copy()
    actions = []
    assigned = set()
    for i,action in enumerate(problem.actions):
        if action.kind=="idle" or decision[i]<0.5:
            continue
        vehicle = action.vehicle
        assigned.add(vehicle)
        grid_energy = float(decision[problem.charge_index[i]]) if i in problem.charge_index else 0.0
        predicted_soc[vehicle] -= problem.energy[i]*decision[i]/state.battery[vehicle]
        predicted_soc[vehicle] += problem.cfg["charging_efficiency"]*grid_energy/state.battery[vehicle]
        actions.append({"type":action.kind,"vehicle":vehicle,"target_index":action.target,
                        "request_id":int(state.request_ids[action.target]) if action.kind=="serv" else None,
                        "origin_zone":action.origin_zone,"destination_zone":action.destination_zone,
                        "predicted_travel_min":float(problem.time[i]),
                        "predicted_traction_energy_kwh":float(problem.energy[i]),
                        "physical_traction_energy_kwh":float(action.energy_kwh),
                        "grid_energy_kwh":grid_energy})
    station_usage = []
    for station in range(len(state.station_xy)):
        ids = [i for i in problem.charge_index if problem.actions[i].target==station]
        new_ports = float(sum(decision[i] for i in ids))
        new_energy = float(sum(decision[problem.charge_index[i]] for i in ids))
        total_energy = new_energy+float(state.committed_kwh[station])
        station_usage.append({"station":station,"new_ports_used":new_ports,
                              "committed_ports":float(state.occupied[station]),
                              "total_ports_used":new_ports+float(state.occupied[station]),
                              "port_capacity":int(state.ports[station]),
                              "new_grid_energy_kwh":new_energy,
                              "committed_grid_energy_kwh":float(state.committed_kwh[station]),
                              "total_grid_energy_kwh":total_energy,
                              "average_grid_power_kw":total_energy/(problem.cfg["interval_min"]/60),
                              "power_capacity_kw":float(state.power[station])})
    coordinates = [{"index":i,"type":action.kind,"vehicle":action.vehicle,"target_index":action.target}
                   for i,action in enumerate(problem.actions)]
    coordinates.extend({"index":qi,"type":"grid_energy_kwh","vehicle":problem.actions[i].vehicle,
                        "target_index":problem.actions[i].target} for i,qi in problem.charge_index.items())
    coordinates.extend({"index":problem.reject_start+r,"type":"rejection","request_id":int(ident)}
                        for r,ident in enumerate(state.request_ids))
    return {"executed_decision_vector":decision.tolist(),"decision_coordinates":coordinates,
            "executed_actions":actions,
            "idle_dispatchable_vehicles":[int(v) for v in np.flatnonzero(state.busy<=0) if int(v) not in assigned],
            "request_rejections":[{"request_id":int(ident),"rejected":int(round(decision[problem.reject_start+r]))}
                                  for r,ident in enumerate(state.request_ids)],
            "charging_energy_by_arc":[{"vehicle":problem.actions[i].vehicle,"station":problem.actions[i].target,
                                       "grid_energy_kwh":float(decision[qi])} for i,qi in problem.charge_index.items()],
            "vehicle_soc":[{"vehicle":v,"before":float(state.soc[v]),"predicted_after":float(predicted_soc[v])}
                            for v in range(len(state.soc))],"station_usage":station_usage}


def make_plots(snapshots,reports,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    state,result,problem = snapshots[0]
    ax = axes[0]
    ax.scatter(*state.zone_xy.T,s=350,marker="s",color="#e5edf5",edgecolor="#9aabc0",label="Zone")
    ax.scatter(*state.station_xy.T,s=100,marker="P",color="#13aa8b",label="Charger")
    ax.scatter(*state.request_xy.T,s=50,marker="*",color="#f0a53a",label="Request")
    ax.scatter(*state.xy.T,s=50,color="#3d5ae1",label="EV")
    colors = {"serv":"#f0a53a","reb":"#6577ba","chg":"#13aa8b"}
    for i,a in enumerate(problem.actions):
        if a.kind=="idle" or result.decision[i]<0.5:
            continue
        destination = state.request_xy[a.target] if a.kind=="serv" else (state.zone_xy[a.target] if a.kind=="reb" else state.station_xy[a.target])
        origin = state.xy[a.vehicle]
        ax.annotate("",xy=destination,xytext=origin,arrowprops=dict(arrowstyle="->",color=colors[a.kind],lw=1.4))
    ax.set(xlabel="Synthetic coordinate x (km)",ylabel="Synthetic coordinate y (km)",title="A joint feasible fleet decision")
    ax.legend(loc="upper center",ncol=4,fontsize=8)
    ax.set_ylim(-0.8,1.2)
    x = np.arange(len(reports))
    axes[1].plot(x,[r["operating_cost"] for r in reports],"o-",color="#3d5ae1",label="Operating cost")
    axes[1].set(xlabel="Control step",ylabel="Dimensionless synthetic cost",title="Executed closed-loop smoke run")
    axes[1].text(.05,.05,"SCIP MIQP • modeled constraints audited\nSynthetic inputs • no benchmark claim",transform=axes[1].transAxes,fontsize=9,color="#41556d")
    fig.savefig(directory/"fleet_snapshot.png",dpi=170)
    plt.close(fig)
    contributions = reports[0]["intrinsic"]["concepts"]
    names,values = list(contributions),list(contributions.values())
    fig,ax = plt.subplots(figsize=(8,4),layout="constrained")
    ax.barh(names,values,color=["#3d5ae1" if v>=0 else "#f0a53a" for v in values])
    ax.axvline(0,color="#bbc5d3",lw=1)
    ax.set(xlabel="Fixed-candidate additive contribution",title="The score exposes seven operational concepts")
    fig.savefig(directory/"concept_contributions.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":
    main()
