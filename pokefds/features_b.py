"""Feature set B: battle-state features, first used in the Logistic Regression +
HistGradientBoosting blend (public score 0.699). The code comes from that notebook and is only
wrapped into a module: call ``init(train_data, type_chart_df)`` once, then ``create_features(battles)``.

Main ideas: replay the 30 turns keeping the last known state of every Pokémon, sleep from Rest vs.
inflicted sleep, "usable" Pokémon, HP-weighted remaining strength, key Gen 2 moves, HP-difference
curve (turns 10/20/30, mean/min/max, last-10-turn trend, slope), species-frequency "team quality".
All lookup tables are built from train.jsonl only.
"""
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
BAD_STATUS = ['slp', 'par', 'tox', 'psn', 'frz', 'brn']
TYPES = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground',
         'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel']
# Moves that define Gen 2 OU strategy; we count how often each side used them.
KEY_MOVES = ['rest', 'sleeptalk', 'spikes', 'rapidspin', 'explosion', 'selfdestruct', 'curse',
             'toxic', 'thunderwave', 'lovelykiss', 'sleeppowder', 'roar', 'whirlwind',
             'batonpass', 'substitute', 'protect', 'recover', 'softboiled', 'milkdrink',
             'growth', 'swordsdance', 'bellydrum', 'perishsong', 'meanlook', 'thief',
             'hiddenpower', 'thunder', 'earthquake', 'pursuit', 'reflect', 'lightscreen']
KEY_EFFECTS = ['substitute', 'confusion', 'leechseed', 'trapped', 'perish', 'encore']

# Type chart: missing pairs are neutral (1.0)
TYPE_CHART = defaultdict(lambda: 1.0)

def effectiveness(move_type, defender_types):
    mult = 1.0
    for t in defender_types:
        if t and t != 'notype':
            mult *= TYPE_CHART[(move_type, t)]
    return mult

def build_species_db(battles):
    db = {}
    for b in battles:
        for p in b['p1_team_details'] + [b['p2_lead_details']]:
            db[p['name']] = {'types': p['types'], **{s: p['base_' + s] for s in STATS}}
    return db

def build_species_freq(battles):
    freq = Counter()
    for b in battles:
        for p in b['p1_team_details']:
            freq[p['name']] += 1
        for n in {t['p2_pokemon_state']['name'] for t in b['battle_timeline']}:
            freq[n] += 1
    return freq


SPECIES_DB, SPECIES_FREQ, TOTAL_FREQ, AVG_BST = {}, Counter(), 1, 0.0


def init(train_data, type_chart_df):
    """Build the lookup tables from the training battles and the supplied type chart."""
    global SPECIES_DB, SPECIES_FREQ, TOTAL_FREQ, AVG_BST
    TYPE_CHART.clear()
    for atk_type, def_type, mult in type_chart_df.itertuples(index=False):
        TYPE_CHART[(atk_type, def_type)] = float(mult)
    SPECIES_DB = build_species_db(train_data)
    SPECIES_FREQ = build_species_freq(train_data)
    TOTAL_FREQ = sum(SPECIES_FREQ.values())
    AVG_BST = np.mean([sum(v[s] for s in STATS) for v in SPECIES_DB.values()])


def team_stat_features(prefix, team):
    f = {}
    for s in STATS:
        vals = [p[s] for p in team] if team else [0]
        f[f'{prefix}_mean_{s}'] = float(np.mean(vals))
        f[f'{prefix}_max_{s}'] = float(np.max(vals))
    f[f'{prefix}_mean_bst'] = float(np.mean([sum(p[s] for s in STATS) for p in team])) if team else 0.0
    return f

def species_types(name):
    return (SPECIES_DB.get(name) or {}).get('types', [])

def species_stat(name, stat, default=0):
    return (SPECIES_DB.get(name) or {}).get(stat, default)

def log_freq(name):
    return np.log((SPECIES_FREQ.get(name, 0) + 1) / TOTAL_FREQ)


def battle_features(b):
    f = {'battle_id': b['battle_id']}
    tl = b['battle_timeline']
    p1_team = [{'name': p['name'], 'types': p['types'], 'item': p.get('item'), 'level': p['level'],
                **{s: p['base_' + s] for s in STATS}} for p in b['p1_team_details']]

    # ---------- 1. static team features ----------
    f.update(team_stat_features('p1', p1_team))
    f['p1_n_leftovers'] = sum(p['item'] == 'leftovers' for p in p1_team)
    f['p1_min_level'] = min(p['level'] for p in p1_team)
    for t in TYPES:
        f[f'p1_type_{t}'] = sum(t in p['types'] for p in p1_team)
    lead = b['p2_lead_details']
    for s in STATS:
        f[f'p2lead_{s}'] = lead['base_' + s]

    # ---------- replay the timeline ----------
    state = {'p1': {p['name']: {'hp': 1.0, 'status': 'nostatus'} for p in p1_team}, 'p2': {}}
    rested = {'p1': set(), 'p2': set()}        # Pokémon whose sleep comes from their own Rest
    cnt = {'p1': Counter(), 'p2': Counter()}
    dmg_dealt = {'p1': 0.0, 'p2': 0.0}
    first_faint_turn = {'p1': 31, 'p2': 31}
    hp_diff_curve = []
    prev = None

    for turn in tl:
        for side, opp in (('p1', 'p2'), ('p2', 'p1')):
            ps = turn[f'{side}_pokemon_state']
            name = ps['name']
            mon = state[side].setdefault(name, {'hp': 1.0, 'status': 'nostatus'})
            mon['hp'], mon['status'] = ps['hp_pct'], ps['status']
            if ps['status'] == 'fnt' and first_faint_turn[side] == 31:
                first_faint_turn[side] = turn['turn']
            for e in ps['effects']:
                if e != 'noeffect':
                    cnt[side]['eff_' + e] += 1
            cnt[side]['boost_sum'] += sum(ps['boosts'].values())

            mv = turn[f'{side}_move_details']
            if mv is not None and mv['name'] == 'rest':
                rested[side].add(name)
            elif ps['status'] != 'slp':
                rested[side].discard(name)

            if mv is None:
                cnt[side]['no_move'] += 1
                prev_ps = prev[f'{side}_pokemon_state'] if prev else None
                if prev_ps and prev_ps['name'] != name and prev_ps['status'] != 'fnt':
                    cnt[side]['switches'] += 1          # voluntary switch (not after a faint)
                continue

            cnt[side]['moves'] += 1
            cnt[side]['cat_' + mv['category']] += 1
            cnt[side]['mv_' + mv['name']] += 1
            if mv['priority'] > 0:
                cnt[side]['priority'] += 1
            if mv['base_power'] > 0:
                opp_types = species_types(turn[f'{opp}_pokemon_state']['name'])
                eff = effectiveness(mv['type'], opp_types) if opp_types else 1.0
                stab = 1.5 if mv['type'] in species_types(name) else 1.0
                cnt[side]['power_sum'] += mv['base_power'] * eff * stab
                cnt[side]['n_damaging'] += 1
                cnt[side]['n_super_eff'] += eff > 1
                cnt[side]['n_not_eff'] += eff < 1

        # damage: HP drop of the opponent's active Pokémon (same Pokémon on both turns)
        if prev is not None:
            for side, opp in (('p1', 'p2'), ('p2', 'p1')):
                a, c = prev[f'{opp}_pokemon_state'], turn[f'{opp}_pokemon_state']
                if a['name'] == c['name'] and c['hp_pct'] < a['hp_pct']:
                    dmg_dealt[side] += a['hp_pct'] - c['hp_pct']

        p1_hp = sum(m['hp'] for m in state['p1'].values())
        p2_hp = sum(m['hp'] for m in state['p2'].values()) + (6 - len(state['p2']))  # unseen = full HP
        hp_diff_curve.append(p1_hp - p2_hp)
        prev = turn

    # ---------- 2 & 3. end-of-window state and battle flow ----------
    for side in ('p1', 'p2'):
        mons = state[side]
        unseen = 6 - len(mons) if side == 'p2' else 0
        alive = {n: m for n, m in mons.items() if m['hp'] > 0 and m['status'] != 'fnt'}
        f[f'{side}_fainted'] = len(mons) - len(alive)
        f[f'{side}_hp_total'] = sum(m['hp'] for m in mons.values()) + unseen
        f[f'{side}_n_revealed'] = len(mons)
        f[f'{side}_n_low_hp'] = sum(0 < m['hp'] < 0.3 for m in mons.values())
        for st in BAD_STATUS:
            f[f'{side}_status_{st}'] = sum(m['status'] == st for m in mons.values())
        f[f'{side}_n_statused'] = sum(m['status'] in BAD_STATUS for m in mons.values())
        f[f'{side}_slp_rest'] = sum(m['status'] == 'slp' and n in rested[side] for n, m in mons.items())
        f[f'{side}_slp_inflicted'] = f[f'{side}_status_slp'] - f[f'{side}_slp_rest']
        f[f'{side}_healthy'] = sum(m['status'] == 'nostatus' and m['hp'] > 0.5 for m in mons.values()) + unseen
        f[f'{side}_n_usable'] = sum(m['status'] not in ('frz', 'slp') for m in alive.values()) + unseen
        f[f'{side}_alive_bst'] = sum(sum(species_stat(n, s, AVG_BST / 6) for s in STATS) * m['hp']
                                     for n, m in mons.items()) + unseen * AVG_BST
        f[f'{side}_alive_atk'] = sum(max(species_stat(n, 'atk', 80), species_stat(n, 'spa', 80)) * m['hp']
                                     for n, m in alive.items())
        f[f'{side}_alive_spe_max'] = max([species_stat(n, 'spe') for n in alive] or [0])
        f[f'{side}_first_faint_turn'] = first_faint_turn[side]

        c = cnt[side]
        for k in ['switches', 'no_move', 'moves', 'priority', 'boost_sum', 'n_damaging',
                  'n_super_eff', 'n_not_eff', 'cat_status', 'cat_physical', 'cat_special']:
            f[f'{side}_{k}'] = c[k]
        f[f'{side}_avg_eff_power'] = c['power_sum'] / max(c['n_damaging'], 1)
        f[f'{side}_dmg_dealt'] = dmg_dealt[side]
        for m in KEY_MOVES:
            f[f'{side}_mv_{m}'] = c['mv_' + m]
        for e in KEY_EFFECTS:
            f[f'{side}_eff_{e}'] = c['eff_' + e]

    # stats of P2's revealed Pokémon
    p2_seen = [dict(name=n, **SPECIES_DB[n]) for n in state['p2'] if n in SPECIES_DB]
    f.update(team_stat_features('p2seen', p2_seen))

    # ---------- 5. final active matchup ----------
    a1, a2 = tl[-1]['p1_pokemon_state'], tl[-1]['p2_pokemon_state']
    t1, t2 = species_types(a1['name']), species_types(a2['name'])
    f['last_p1_hp'], f['last_p2_hp'] = a1['hp_pct'], a2['hp_pct']
    f['last_hp_diff'] = a1['hp_pct'] - a2['hp_pct']
    f['last_p1_bad_status'] = int(a1['status'] in BAD_STATUS)
    f['last_p2_bad_status'] = int(a2['status'] in BAD_STATUS)
    f['last_boost_diff'] = sum(a1['boosts'].values()) - sum(a2['boosts'].values())
    f['last_p1_best_eff'] = max([effectiveness(t, t2) for t in t1 if t != 'notype'] or [1.0])
    f['last_p2_best_eff'] = max([effectiveness(t, t1) for t in t2 if t != 'notype'] or [1.0])
    f['last_speed_diff'] = species_stat(a1['name'], 'spe') - species_stat(a2['name'], 'spe')

    # ---------- 6. team quality ----------
    p1_lf = [log_freq(p['name']) for p in p1_team]
    p2_lf = [log_freq(n) for n in state['p2']]
    f['p1_team_std'], f['p1_team_min_std'] = np.mean(p1_lf), min(p1_lf)
    f['p2_team_std'], f['p2_team_min_std'] = np.mean(p2_lf), min(p2_lf)

    # ---------- differences (P1 - P2) ----------
    for k in ['fainted', 'hp_total', 'n_statused', 'healthy', 'switches', 'n_low_hp', 'dmg_dealt',
              'status_slp', 'status_par', 'status_tox', 'boost_sum', 'avg_eff_power', 'n_super_eff',
              'slp_inflicted', 'alive_bst', 'alive_atk', 'alive_spe_max', 'n_usable', 'team_std']:
        f[f'diff_{k}'] = f[f'p1_{k}'] - f[f'p2_{k}']
    for k in ['spikes', 'rest', 'explosion', 'rapidspin']:
        f[f'diff_mv_{k}'] = f[f'p1_mv_{k}'] - f[f'p2_mv_{k}']

    # ---------- 4. momentum ----------
    hc = hp_diff_curve
    f['hpdiff_t10'], f['hpdiff_t20'], f['hpdiff_t30'] = hc[9], hc[19], hc[29]
    f['hpdiff_mean'], f['hpdiff_min'], f['hpdiff_max'] = float(np.mean(hc)), min(hc), max(hc)
    f['hpdiff_trend_last10'] = hc[29] - hc[19]
    f['hpdiff_slope'] = float(np.polyfit(np.arange(len(hc)), hc, 1)[0])

    if 'player_won' in b:
        f['player_won'] = int(b['player_won'])
    return f


def create_features(battles):
    return pd.DataFrame([battle_features(b) for b in battles]).fillna(0)
