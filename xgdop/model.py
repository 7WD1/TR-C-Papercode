"""Typed attention, temporal memory, quantile forecasts and concept actor."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Bernoulli, Beta
from .environment import RELATIONS, TYPES, geometry_candidates, action_features


def mlp(inp, out, width):
    return nn.Sequential(nn.Linear(inp, width), nn.ELU(), nn.Linear(width, out))


class TypedAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_width = heads, width // heads
        self.transforms = nn.ModuleDict({r: nn.Linear(width, width, bias=False) for r in RELATIONS})
        self.attention = nn.ParameterDict({r: nn.Parameter(torch.empty(heads, 2 * self.head_width)) for r in RELATIONS})
        for parameter in self.attention.values():
            nn.init.xavier_uniform_(parameter)

    def forward(self, h, edges, masks=None):
        count = len(h)
        out = h.new_zeros(count, self.heads, self.head_width)
        attention = {}
        for relation, edge in edges.items():
            if edge.numel() == 0:
                attention[relation] = h.new_empty((0, self.heads))
                continue
            source, dest = edge
            transformed = self.transforms[relation](h).reshape(count, self.heads, self.head_width)
            pair = torch.cat((transformed[dest], transformed[source]), -1)
            logits = F.leaky_relu((pair * self.attention[relation]).sum(-1), negative_slope=0.2)
            maximum = h.new_full((count, self.heads), -torch.inf)
            maximum.scatter_reduce_(0, dest[:, None].expand(-1, self.heads), logits,
                                    reduce="amax", include_self=True)
            exps = torch.exp(logits - maximum[dest])
            denominator = h.new_zeros(count, self.heads).index_add(0, dest, exps)
            alpha = exps / denominator[dest].clamp_min(1e-12)
            attention[relation] = alpha
            multiplier = 1 if masks is None else masks[relation][:, None, None]
            messages = transformed[source] * alpha[..., None] * multiplier
            out = out.index_add(0, dest, messages)
        return F.elu(out.reshape(count, -1)), attention


class ActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        width = cfg["hidden_dim"]
        self.inputs = nn.ModuleList([nn.Linear(10, width) for _ in TYPES])
        self.layers = nn.ModuleList([TypedAttention(width, cfg["attention_heads"])
                                     for _ in range(cfg["graph_layers"])])
        self.gru = nn.GRUCell(width, width)
        self.action_encoder = mlp(4 * width + 10, width, width)
        self.zone_quantiles = mlp(width, 2 * (cfg["prediction_horizon"] + 1) * 5, width)
        self.action_quantiles = mlp(width, 10, width)
        self.concepts = mlp(2 * width, 7, width)
        self.concept_weights = nn.ModuleList([mlp(2 * width, 1, width) for _ in range(7)])
        self.residual = mlp(2 * width, cfg["residual_dim"], width)
        self.residual_weight = nn.Linear(cfg["residual_dim"], 1, bias=False)
        self.rejection = mlp(width + 10, 1, width)
        self.charging = mlp(width, 1, width)
        self.critic = mlp(width, 1, width)

    def encode(self, history, feature_noise=None, masks=None):
        memory = {}
        for t, frame in enumerate(history):
            features = frame["features"]
            if feature_noise is not None and t == len(history) - 1:
                features = features + feature_noise
            kinds = frame["kinds"]
            h = sum(layer(features) * (kinds == kind)[:, None] for kind, layer in enumerate(self.inputs))
            for layer in self.layers:
                h, attention = layer(h, frame["edges"], masks if t == len(history) - 1 else None)
            previous = torch.stack([memory.get(key, h.new_zeros(h.shape[1])) for key in frame["keys"]])
            h = self.gru(h, previous)
            memory = dict(zip(frame["keys"], h.unbind()))
        return h, h.mean(0), frame["index"], attention

    @staticmethod
    def quantiles(raw):
        return torch.cumsum(F.softplus(raw), dim=-1)

    def forward(self, state, quota=None, feature_noise=None, masks=None, support=None):
        cfg = self.cfg
        h, pooled, index, attention = self.encode(state.history, feature_noise, masks)
        geometry = geometry_candidates(state, cfg, quota)
        raw = action_features(geometry, state, cfg)
        width = cfg["hidden_dim"]
        inputs = []
        for i, a in enumerate(geometry):
            station = h[index[("station", a.target)]] if a.kind == "chg" else h.new_zeros(width)
            inputs.append(torch.cat((h[index[("vehicle", a.vehicle)]], h[index[("zone", a.origin_zone)]],
                                     h[index[("zone", a.destination_zone)]], station, raw[i])))
        if not inputs:
            raise RuntimeError("No dispatchable vehicle; advance/reset this synthetic episode before calling the policy.")
        embedding_all = self.action_encoder(torch.stack(inputs))
        te = self.quantiles(self.action_quantiles(embedding_all).reshape(-1, 2, 5))
        # Idling has exactly zero travel and energy; neural supervision still includes its head outputs.
        time_all, energy_all = te[:, 0, -1], te[:, 1, -1]
        eligible = []
        for i, a in enumerate(geometry):
            if a.kind == "idle":
                eligible.append(i)
                continue
            safe = state.soc[a.vehicle] - float(energy_all[i].detach()) / state.battery[a.vehicle] >= cfg["minimum_soc"]
            if a.kind == "serv":
                safe = safe and state.wait[a.target] + float(time_all[i].detach()) <= cfg["max_wait_min"]
            if a.kind == "chg":
                safe = safe and float(time_all[i].detach()) < cfg["interval_min"]
            if safe:
                eligible.append(i)
        if support is not None:
            # During PPO epochs, the collected action mask is part of the stored
            # observation. Reuse it exactly even if predictor parameters change.
            positions = {action:i for i,action in enumerate(geometry)}
            eligible = [positions[action] for action in support]
        actions = [geometry[i] for i in eligible]
        embedding = embedding_all[eligible]
        joint = torch.cat((pooled.expand(len(actions), -1), embedding), -1)
        concept = self.concepts(joint)
        applicable = concept.new_ones(concept.shape)
        for i,a in enumerate(actions):
            if a.kind!="chg":
                applicable[i,3] = applicable[i,6] = 0
            if a.kind!="serv":
                applicable[i,4] = 0
        concept = concept*applicable
        weight = torch.cat([head(joint) for head in self.concept_weights], -1)
        residual_vector = self.residual(joint)
        residual = self.residual_weight(residual_vector).squeeze(-1)
        score = (concept * weight).sum(-1) + residual
        zones = torch.stack([h[index[("zone", z)]] for z in range(len(state.zone_xy))])
        ds = self.quantiles(self.zone_quantiles(zones).reshape(-1, cfg["prediction_horizon"] + 1, 2, 5))
        probability = torch.zeros_like(score)
        groups = []
        for v in np.flatnonzero(state.busy <= 0):
            ids = [i for i, a in enumerate(actions) if a.vehicle == v]
            groups.append(ids)
            probability[ids] = F.softmax(score[ids] / cfg["decoder_temperature"], 0)
        request_rows = [state.history[-1]["features"][index[("request", int(r))]] for r in state.request_ids]
        reject_input = (torch.cat((pooled.expand(len(request_rows), -1), torch.stack(request_rows)), -1)
                        if request_rows else pooled.new_empty((0,width+10)))
        rejection_probability = torch.sigmoid(self.rejection(reject_input)).squeeze(-1)
        charge_mean = torch.sigmoid(self.charging(embedding)).squeeze(-1).clamp(0.01, 0.99)
        time = time_all[eligible]
        energy = energy_all[eligible]
        active = torch.tensor([a.kind != "idle" for a in actions], dtype=time.dtype)
        time, energy = time * active, energy * active
        labels = self.concept_labels(state, actions, ds.detach(), time.detach(), energy.detach())
        return dict(actions=actions, geometry=geometry, te=te, ds=ds, embeddings=embedding,
                    pooled=pooled, score=score, concept=concept, weights=weight, residual=residual,
                    residual_vector=residual_vector, probability=probability, groups=groups,
                    reject_probability=rejection_probability, charge_mean=charge_mean,
                    time=time, energy=energy, labels=labels,
                    value=self.critic(pooled).squeeze() if cfg["loss_weights"]["rl"] else pooled.new_zeros(()),
                    attention=attention)

    def concept_labels(self, state, actions, ds, times, energies):
        cfg, rows = self.cfg, []
        for i, a in enumerate(actions):
            z = a.affected_zone
            demand_now, demand_future = ds[z, 0, 0, 2], ds[z, -1, 0, 2]
            supply_future = ds[z, -1, 1, 2]
            soc_margin = max(0, cfg["minimum_soc"] + cfg["soc_margin"] - state.soc[a.vehicle]
                             + float(energies[i]) / state.battery[a.vehicle])
            rows.append([float((demand_future - demand_now) / (1 + 1e-6)),
                         float(torch.relu(demand_future - supply_future) / (demand_future + 1e-6)),
                         soc_margin,
                         min(1, (state.occupied[a.target] + state.queue[a.target]) / max(1, state.ports[a.target])) if a.kind == "chg" else 0,
                         float(times[i]) / (cfg["max_wait_min"] + 1e-6) if a.kind == "serv" else 0,
                         float(energies[i]) / (state.battery[a.vehicle] + 1e-6),
                         state.carbon[a.target] / (max(state.carbon) + 1e-6) if a.kind == "chg" else 0])
        return torch.tensor(rows, dtype=torch.float32)

    def distributions(self, output):
        cats = [Categorical(probs=output["probability"][ids]) for ids in output["groups"]]
        rejection = Bernoulli(probs=output["reject_probability"].clamp(1e-6, 1-1e-6))
        charge_ids = [i for i, a in enumerate(output["actions"]) if a.kind == "chg"]
        mean = output["charge_mean"][charge_ids]
        concentration = self.cfg["beta_concentration"]
        charge = Beta(mean * concentration, (1 - mean) * concentration)
        return cats, rejection, charge, charge_ids

    def sample(self, output):
        cats, rejection, charge, charge_ids = self.distributions(output)
        latent = {"choices": [d.sample() for d in cats], "rejection": rejection.sample(),
                  "fraction": charge.sample()}
        selected = torch.zeros_like(output["probability"])
        for ids, choice in zip(output["groups"], latent["choices"]):
            selected[ids[int(choice)]] = 1
        latent["selected"] = selected
        return latent, self.logprob(output, latent)[0]

    def logprob(self, output, latent):
        cats, rejection, charge, _ = self.distributions(output)
        logp = sum(d.log_prob(a) for d, a in zip(cats, latent["choices"]))
        logp = logp + rejection.log_prob(latent["rejection"]).sum() + charge.log_prob(latent["fraction"]).sum()
        entropy = sum(d.entropy() for d in cats) + rejection.entropy().sum() + charge.entropy().sum()
        return logp, entropy

    def decode(self, output, problem, latent=None, scores=None):
        probability = output["probability"]
        if scores is not None:
            probability = torch.zeros_like(scores)
            for ids in output["groups"]:
                probability[ids] = F.softmax(scores[ids] / self.cfg["decoder_temperature"], 0)
        selected = probability if latent is None else latent["selected"]
        binary = torch.stack([selected[i] if a.kind != "idle" else selected[i] * 0
                              for i, a in enumerate(problem.actions)])
        if latent is None:
            fractions = output["charge_mean"][list(problem.charge_index)]
            rejection = output["reject_probability"]
        else:
            fractions, rejection = latent["fraction"], latent["rejection"]
        quantities = torch.stack([binary[i] * fractions[k] * problem.qbar[i]
                                  for k, i in enumerate(problem.charge_index)]) if problem.charge_index else binary.new_empty(0)
        return torch.cat((binary, quantities, rejection))
