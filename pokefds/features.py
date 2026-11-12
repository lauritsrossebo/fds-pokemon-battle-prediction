"""Feature engineering for the FDS 2026 Pokémon battle challenge (Gen 2 OU).

One row per battle. Three groups of features:

* static   - team composition known before the battle (P1 full team, P2 revealed team)
* dynamic  - what happened during the 30 observed turns
* snapshot - the state of the two active Pokémon at the last observed turn

Everything is derived from the competition files only: base stats and types come
from the battles themselves, type effectiveness from ``type_chart_gen2.csv``.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
STATUS_PENALTY = {"nostatus": 0, "psn": 1, "brn": 2, "par": 2, "tox": 2, "slp": 3, "frz": 3}
MAJOR_STATUSES = ("slp", "par", "tox", "psn", "frz", "brn")
TRACKED_EFFECTS = ("substitute", "confusion", "leechseed", "trapped", "perish", "protect")
TRACKED_ITEMS = ("leftovers", "thickclub", "miracleberry", "mintberry", "goldberry",
                 "lightball", "berryjuice", "charcoal")
DAMAGING = {"physical", "special"}


# --------------------------------------------------------------------------- #
# Type chart and species dictionary
# --------------------------------------------------------------------------- #
class TypeChart:
    """Gen 2 type effectiveness read from the supplied CSV (missing pairs = 1.0)."""

    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.table = {(a.lower(), d.lower()): float(m)
                      for a, d, m in df[["attacking_type", "defending_type", "multiplier"]]
                      .itertuples(index=False)}

    def multiplier(self, attack_type, defending_types):
        attack_type = (attack_type or "").lower()
        out = 1.0
        for t in defending_types:
            out *= self.table.get((attack_type, t), 1.0)
        return out

    def best_stab(self, attacker_types, defender_types):
        """Best multiplier the attacker gets with its own (STAB) types."""
        if not attacker_types:
            return 1.0
        return max(self.multiplier(t, defender_types) for t in attacker_types)


def clean_types(pokemon):
    return [t.lower() for t in pokemon.get("types", []) if t and t.lower() != "notype"]


def build_species_dictionary(*datasets):
    """species name -> {types, base stats}, collected from the teams listed in the given battles.
    We build it from train.jsonl only; at prediction time each battle adds its own listed Pokémon
    (see ``create_features``), so the test file is used only to predict its own battles."""
    species = {}
    for data in datasets:
        for battle in data:
            for p in battle.get("p1_team_details", []) + [battle.get("p2_lead_details") or {}]:
                if p.get("name"):
                    species[p["name"]] = {"types": clean_types(p),
                                          **{s: p.get(f"base_{s}", 0) for s in STATS}}
    return species


# --------------------------------------------------------------------------- #
# Static features
# --------------------------------------------------------------------------- #
def team_stats(f, prefix, team):
    for s in STATS:
        vals = [p[s] for p in team] or [0]
        f[f"{prefix}_mean_{s}"] = float(np.mean(vals))
        f[f"{prefix}_max_{s}"] = float(np.max(vals))
    f[f"{prefix}_mean_bst"] = float(np.mean([sum(p[s] for s in STATS) for p in team])) if team else 0.0


def role_counts(f, prefix, team):
    f[f"{prefix}_phys_sweepers"] = sum(p["atk"] >= 100 and p["spe"] >= 100 for p in team)
    f[f"{prefix}_spec_sweepers"] = sum(p["spa"] >= 100 and p["spe"] >= 100 for p in team)
    f[f"{prefix}_phys_walls"] = sum(p["hp"] * p["def"] >= 10000 for p in team)
    f[f"{prefix}_spec_walls"] = sum(p["hp"] * p["spd"] >= 10000 for p in team)


def matchup_scores(f, chart, team1, team2):
    """Average best-STAB multiplier and number of super-effective threats, both directions."""
    def offense(att, dfn):
        if not att or not dfn:
            return 0.0
        return float(np.mean([np.mean([chart.best_stab(a["types"], d["types"]) for d in dfn]) for a in att]))

    def threats(att, dfn):
        return sum(any(chart.best_stab(a["types"], d["types"]) >= 2 for d in dfn) for a in att)

    f["team_offense_p1_vs_p2"] = offense(team1, team2)
    f["team_offense_p2_vs_p1"] = offense(team2, team1)
    f["team_offense_diff"] = f["team_offense_p1_vs_p2"] - f["team_offense_p2_vs_p1"]
    f["se_threats_p1"] = threats(team1, team2)
    f["se_threats_p2"] = threats(team2, team1)
    f["se_threats_diff"] = f["se_threats_p1"] - f["se_threats_p2"]


def item_features(f, p1_details):
    items = Counter(p.get("item") or "noitem" for p in p1_details)
    for it in TRACKED_ITEMS:
        f[f"p1_item_{it}"] = items.get(it, 0)
    f["p1_distinct_items"] = len(items)


# --------------------------------------------------------------------------- #
# Dynamic features (the 30 observed turns)
# --------------------------------------------------------------------------- #
def is_fainted(state):
    return state.get("status") == "fnt" or state.get("hp_pct", 1.0) <= 0


def replay_timeline(timeline, p1_names, species, chart):
    """Walk the turns once and collect everything the dynamic features need."""
    hp = {1: {n: 1.0 for n in p1_names}, 2: {}}
    status = {1: {n: "nostatus" for n in p1_names}, 2: {}}
    acc = {p: defaultdict(float) for p in (1, 2)}
    dmg = {1: [], 2: []}
    eff_power = {1: [], 2: []}
    first_faint_turn = {1: 31, 2: 31}
    prev = None

    for turn in timeline:
        for p, o in ((1, 2), (2, 1)):
            st = turn[f"p{p}_pokemon_state"]
            opp = turn[f"p{o}_pokemon_state"]
            move = turn.get(f"p{p}_move_details")
            name = st["name"]
            a = acc[p]

            # switches (voluntary = previous active was still alive) and skipped turns
            switched = prev is not None and prev[f"p{p}_pokemon_state"]["name"] != name
            if switched and not is_fainted(prev[f"p{p}_pokemon_state"]):
                a["switches"] += 1
            if move is None and not switched and not is_fainted(st):
                a["skipped_turns"] += 1

            if move:
                cat = move.get("category")
                a[f"moves_{cat}"] += 1
                if move.get("priority", 0) > 0:
                    a["priority_moves"] += 1
                if cat in DAMAGING and move.get("base_power", 0) > 0:
                    att_types = species.get(name, {}).get("types", [])
                    stab = 1.5 if move["type"] in att_types else 1.0
                    mult = chart.multiplier(move["type"], species.get(opp["name"], {}).get("types", []))
                    eff_power[p].append(move["base_power"] * stab * mult)
                    if mult >= 2:
                        a["super_effective_hits"] += 1
                    elif mult == 0:
                        a["immune_hits"] += 1
                    # damage dealt: drop of the opponent's HP if it stayed in
                    if prev is not None and prev[f"p{o}_pokemon_state"]["name"] == opp["name"]:
                        drop = prev[f"p{o}_pokemon_state"]["hp_pct"] - opp["hp_pct"]
                        if drop > 0:
                            dmg[p].append(drop)

            # healing: HP going up while the same Pokémon stays in (recovery moves, leftovers...)
            if prev is not None and not switched:
                gain = st["hp_pct"] - prev[f"p{p}_pokemon_state"]["hp_pct"]
                if gain > 0:
                    a["hp_healed"] += gain
                    if gain >= 0.25:
                        a["big_heals"] += 1

            hp[p][name] = st["hp_pct"]
            status[p][name] = st["status"]
            if is_fainted(st) and first_faint_turn[p] == 31:
                first_faint_turn[p] = turn["turn"]
        prev = turn

    return hp, status, acc, dmg, eff_power, first_faint_turn


def dynamic_features(f, battle, species, chart):
    timeline = battle["battle_timeline"]
    p1_names = [p["name"] for p in battle["p1_team_details"]]
    hp, status, acc, dmg, eff_power, first_faint = replay_timeline(timeline, p1_names, species, chart)

    for p in (1, 2):
        a, pre = acc[p], f"p{p}"
        fainted = sum(1 for n in hp[p] if hp[p][n] <= 0 or status[p][n] == "fnt")
        revealed = len(hp[p])
        unseen = 6 - revealed
        f[f"{pre}_fainted"] = fainted
        f[f"{pre}_revealed"] = revealed
        f[f"{pre}_hp_sum"] = sum(hp[p].values()) + (unseen * 1.0 if p == 2 else 0.0)
        f[f"{pre}_alive"] = 6 - fainted
        f[f"{pre}_first_faint_turn"] = first_faint[p]
        for s in MAJOR_STATUSES:
            f[f"{pre}_status_{s}"] = sum(1 for n, v in status[p].items() if v == s)
        f[f"{pre}_status_penalty"] = sum(STATUS_PENALTY.get(v, 0) for v in status[p].values())
        for k in ("switches", "skipped_turns", "moves_physical", "moves_special", "moves_status",
                  "priority_moves", "super_effective_hits", "immune_hits", "hp_healed", "big_heals"):
            f[f"{pre}_{k}"] = a.get(k, 0.0)
        f[f"{pre}_avg_damage"] = float(np.mean(dmg[p])) if dmg[p] else 0.0
        f[f"{pre}_max_damage"] = float(np.max(dmg[p])) if dmg[p] else 0.0
        f[f"{pre}_total_damage"] = float(np.sum(dmg[p]))
        f[f"{pre}_avg_eff_power"] = float(np.mean(eff_power[p])) if eff_power[p] else 0.0

    for k in ("fainted", "hp_sum", "status_penalty", "switches", "skipped_turns", "hp_healed",
              "avg_damage", "total_damage", "avg_eff_power", "super_effective_hits", "status_slp"):
        f[f"diff_{k}"] = f[f"p1_{k}"] - f[f"p2_{k}"]
    total = f["p1_hp_sum"] + f["p2_hp_sum"]
    f["hp_share_p1"] = f["p1_hp_sum"] / total if total > 0 else 0.5


def snapshot_features(f, last_turn, species, chart):
    s1, s2 = last_turn["p1_pokemon_state"], last_turn["p2_pokemon_state"]
    d1, d2 = species.get(s1["name"], {}), species.get(s2["name"], {})
    f["active_hp_diff"] = s1["hp_pct"] - s2["hp_pct"]
    f["active_boost_diff"] = sum(s1["boosts"].values()) - sum(s2["boosts"].values())
    for s in STATS:
        f[f"active_{s}_diff"] = d1.get(s, 0) - d2.get(s, 0)
    t1, t2 = d1.get("types", []), d2.get("types", [])
    f["active_matchup_diff"] = chart.best_stab(t1, t2) - chart.best_stab(t2, t1)
    f["active_status_adv"] = STATUS_PENALTY.get(s2["status"], 0) - STATUS_PENALTY.get(s1["status"], 0)
    for e in TRACKED_EFFECTS:
        f[f"active_effect_{e}_diff"] = int(e in s1["effects"]) - int(e in s2["effects"])


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def create_features(data, species, chart):
    rows = []
    for battle in data:
        f = {}
        # the battle's own listed Pokémon complete the species dictionary for this battle only
        own = build_species_dictionary([battle])
        species_b = {**species, **own}
        p1_team = [{"name": p["name"], "types": clean_types(p), **{s: p[f"base_{s}"] for s in STATS}}
                   for p in battle["p1_team_details"]]
        p2_names = list(dict.fromkeys([battle["p2_lead_details"]["name"]] +
                                      [t["p2_pokemon_state"]["name"] for t in battle["battle_timeline"]]))
        p2_team = [{"name": n, **species_b[n]} for n in p2_names if n in species_b]

        team_stats(f, "p1", p1_team)
        team_stats(f, "p2", p2_team)
        for s in STATS:
            f[f"diff_mean_{s}"] = f[f"p1_mean_{s}"] - f[f"p2_mean_{s}"]
        role_counts(f, "p1", p1_team)
        role_counts(f, "p2", p2_team)
        matchup_scores(f, chart, p1_team, p2_team)
        item_features(f, battle["p1_team_details"])
        f["p1_type_diversity"] = len({t for p in p1_team for t in p["types"]})
        f["p2_type_diversity"] = len({t for p in p2_team for t in p["types"]})

        dynamic_features(f, battle, species_b, chart)
        snapshot_features(f, battle["battle_timeline"][-1], species_b, chart)

        f["battle_id"] = battle["battle_id"]
        if "player_won" in battle:
            f["player_won"] = int(battle["player_won"])
        rows.append(f)
    return pd.DataFrame(rows).fillna(0)
