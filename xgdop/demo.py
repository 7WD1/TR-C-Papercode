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
                            "intrinsic":intrinsic(output),"fixed_score_shap":fixed_score_shap(output)})
            state = environment.step(problem,result.decision)
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
                        charge_target=alternative_result.decision[list(alternative_problem.charge_index.values())])
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
