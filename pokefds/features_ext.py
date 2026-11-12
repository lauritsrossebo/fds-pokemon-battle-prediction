"""Extra features for the final model (added on top of pokefds.features).

* HP trajectory      - HP balance after turns 10 and 20, momentum over the last 10 turns
* weighted HP        - remaining HP weighted by each Pokémon's base-stat total
* per-stat boosts    - boosts of the active Pokémon at turn 30, stat by stat
* risk counts        - Pokémon alive with low HP, Pokémon alive and healthy
* survivors         - mean stats, bulk and type matchup of the Pokémon still alive, speed/counters vs the active foe
* sleep turns        - turns spent asleep by each side
* bag of species     - which species are on P1's team / revealed for P2
* bag of moves       - how often each side used the most common moves
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .features import STATS, create_features, is_fainted

BOOSTS = ("atk", "def", "spa", "spd", "spe")


def build_vocab(train_data, top_moves=60, min_species=30):
    """Moves and species frequent enough in the TRAINING inputs to get their own column."""
    moves = Counter(t[f"p{p}_move_details"]["name"] for b in train_data for t in b["battle_timeline"]
                    for p in (1, 2) if t[f"p{p}_move_details"])
    species = Counter(p["name"] for b in train_data for p in b["p1_team_details"])
    species.update(t["p2_pokemon_state"]["name"] for b in train_data
                   for t in {x["p2_pokemon_state"]["name"]: x for x in b["battle_timeline"]}.values())
    return {"moves": [m for m, _ in moves.most_common(top_moves)],
            "species": sorted(s for s, c in species.items() if c >= min_species)}


def hp_balance(timeline, upto, p1_names):
    """P1 minus P2 total HP after `upto` turns (unseen Pokémon at full HP)."""
    hp = {1: {n: 1.0 for n in p1_names}, 2: {}}
    for t in timeline[:upto]:
        for p in (1, 2):
            s = t[f"p{p}_pokemon_state"]
            hp[p][s["name"]] = 0.0 if is_fainted(s) else s["hp_pct"]
    return sum(hp[1].values()) - (sum(hp[2].values()) + (6 - len(hp[2])))


def extra_features(battle, species, chart, vocab, mean_bst):
    f = {}
    tl = battle["battle_timeline"]
    p1_names = [p["name"] for p in battle["p1_team_details"]]

    # HP trajectory
    b10, b20, b30 = (hp_balance(tl, k, p1_names) for k in (10, 20, 30))
    f["hp_balance_t10"], f["hp_balance_t20"] = b10, b20
    f["hp_momentum_last10"] = b30 - b20
    f["hp_momentum_mid10"] = b20 - b10

    # last known state of every Pokémon
    last = {1: {}, 2: {}}
    sleep_turns = {1: 0, 2: 0}
    for t in tl:
        for p in (1, 2):
            s = t[f"p{p}_pokemon_state"]
            last[p][s["name"]] = s
            if s["status"] == "slp":
                sleep_turns[p] += 1

    def bst(name):
        d = species.get(name)
        return sum(d[s] for s in STATS) if d else mean_bst

    for p in (1, 2):
        names = p1_names if p == 1 else list(last[2])
        w_hp, low, healthy = 0.0, 0, 0
        for n in names:
            s = last[p].get(n)
            h = 1.0 if s is None else (0.0 if is_fainted(s) else s["hp_pct"])
            w_hp += h * bst(n) / 600
            if s is not None and not is_fainted(s):
                low += h < 0.3
                healthy += s["status"] == "nostatus" and h > 0.7
            elif s is None:
                healthy += 1
        if p == 2:                                   # unseen P2 Pokémon: full HP, average strength
            w_hp += (6 - len(names)) * mean_bst / 600
            healthy += 6 - len(names)
        f[f"p{p}_weighted_hp"] = w_hp
        f[f"p{p}_low_hp_alive"] = low
        f[f"p{p}_healthy"] = healthy
        f[f"p{p}_sleep_turns"] = sleep_turns[p]
    f["diff_weighted_hp"] = f["p1_weighted_hp"] - f["p2_weighted_hp"]
    f["diff_healthy"] = f["p1_healthy"] - f["p2_healthy"]
    f["diff_sleep_turns"] = f["p1_sleep_turns"] - f["p2_sleep_turns"]

    # what is still alive at turn 30: strength and type matchup of the survivors
    def alive(p, names):
        out = []
        for n in names:
            s = last[p].get(n)
            if (s is None or not is_fainted(s)) and n in species:
                out.append({"name": n, **species[n]})
        return out
    a1 = alive(1, p1_names)
    a2 = alive(2, list(last[2]))
    for p, team in ((1, a1), (2, a2)):
        for s in ("atk", "spa", "spe"):
            f[f"p{p}_alive_mean_{s}"] = float(np.mean([m[s] for m in team])) if team else 0.0
        f[f"p{p}_alive_mean_bulk"] = float(np.mean([m["hp"] * (m["def"] + m["spd"]) / 2 for m in team])) / 1e4 if team else 0.0
    if a1 and a2:
        f["alive_offense_p1"] = float(np.mean([max(chart.best_stab(x["types"], d["types"]) for d in a2) for x in a1]))
        f["alive_offense_p2"] = float(np.mean([max(chart.best_stab(x["types"], d["types"]) for d in a1) for x in a2]))
    else:
        f["alive_offense_p1"] = f["alive_offense_p2"] = 0.0
    f["alive_offense_diff"] = f["alive_offense_p1"] - f["alive_offense_p2"]
    act2 = species.get(tl[-1]["p2_pokemon_state"]["name"])
    act1 = species.get(tl[-1]["p1_pokemon_state"]["name"])
    f["p1_alive_faster_than_p2_active"] = sum(m["spe"] > act2["spe"] for m in a1) if act2 else 0
    f["p2_alive_faster_than_p1_active"] = sum(m["spe"] > act1["spe"] for m in a2) if act1 else 0
    f["p1_counters_to_p2_active"] = sum(chart.best_stab(m["types"], act2["types"]) >= 2 for m in a1) if act2 else 0
    f["p2_counters_to_p1_active"] = sum(chart.best_stab(m["types"], act1["types"]) >= 2 for m in a2) if act1 else 0

    # per-stat boosts of the active Pokémon at turn 30
    s1, s2 = tl[-1]["p1_pokemon_state"], tl[-1]["p2_pokemon_state"]
    for k in BOOSTS:
        f[f"boost_{k}_diff"] = s1["boosts"].get(k, 0) - s2["boosts"].get(k, 0)

    # bag of species
    p1_set, p2_set = set(p1_names), set(last[2]) | {battle["p2_lead_details"]["name"]}
    for sp in vocab["species"]:
        f[f"sp_{sp}"] = int(sp in p1_set) - int(sp in p2_set)

    # bag of moves
    used = {1: Counter(), 2: Counter()}
    for t in tl:
        for p in (1, 2):
            m = t[f"p{p}_move_details"]
            if m:
                used[p][m["name"]] += 1
    for mv in vocab["moves"]:
        f[f"mv_{mv}_p1"] = used[1][mv]
        f[f"mv_{mv}_p2"] = used[2][mv]
    return f


def create_features_ext(data, species, chart, vocab):
    base = create_features(data, species, chart)
    mean_bst = float(np.mean([sum(d[s] for s in STATS) for d in species.values()]))
    extra = pd.DataFrame([extra_features(b, species, chart, vocab, mean_bst) for b in data])
    return pd.concat([base.reset_index(drop=True), extra], axis=1).fillna(0)
