"""v19 place map on the real log (Section VII-I rebuilt). Track graph from balise-group transitions: the
distance between consecutive groups b1 -> b2 is the median of D_LRBG at the last sample before the change;
graph distance by Dijkstra (undirected).  Map score of an epoch = number of event starts of the reference
trains within 2.5 km (graph distance) of the epoch's last balise group, divided by the number of passes of the
reference trains through that group.  Combination with TLR-ord / TLR-mix-ord: mean of the two rank scores
(equal weights, no tuning).  Evaluations: (a) 300 random half splits of the 70 test trains (map from the
calibration half, AUC on the other half); (b) days 1-2 -> days 3-8 (all trains)."""
import json, heapq
import numpy as np
from collections import defaultdict
import sncf_eval19 as S, sncf_run19 as R

# --- graph
pairs = defaultdict(list)
for x in S.TR:
    bg, dl = x['bg'], x['dl']
    for i in range(1, len(bg)):
        if np.isfinite(bg[i]) and np.isfinite(bg[i - 1]) and bg[i] != bg[i - 1] and np.isfinite(dl[i - 1]):
            a, b = int(bg[i - 1]), int(bg[i])
            pairs[(min(a, b), max(a, b))].append(dl[i - 1])
G = defaultdict(list)
for (a, b), v in pairs.items():
    if len(v) >= 2 and 0 < np.median(v) < 5000:
        G[a].append((b, float(np.median(v)))); G[b].append((a, float(np.median(v))))
NEAR = {}
for s in G:
    dist = {s: 0.0}; pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18) or d > 2500: continue
        for v, w in G[u]:
            nd = d + w
            if nd <= 2500 and nd < dist.get(v, 1e18):
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    NEAR[s] = set(dist)

# --- event starts and passes per train
ev_bg, passes = [], []
for x in S.TR:
    for (s, e) in x['events']:
        i = np.searchsorted(x['t'], s)
        if i < len(x['bg']) and np.isfinite(x['bg'][i]): ev_bg.append((x['train'], x['day'], int(x['bg'][i])))
    seen = [int(b) for b in x['bg'][np.isfinite(x['bg'])]]
    for b in set(seen): passes.append((x['train'], x['day'], b))


def epoch_bg(e):
    b = e['bg'][np.isfinite(e['bg'])]
    return int(b[-1]) if b.size else None


def score_fn(ref):
    """ref(train, day) -> bool : which events and passes form the map"""
    evc = defaultdict(int); pc = defaultdict(int)
    for tr, d, b in ev_bg:
        if ref(tr, d): evc[b] += 1
    for tr, d, b in passes:
        if ref(tr, d): pc[b] += 1
    def f(b):
        if b is None or pc.get(b, 0) == 0: return 0.0
        return sum(evc.get(v, 0) for v in NEAR.get(b, {b})) / pc[b]
    return f


def rank(x):
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


pbg = np.array([epoch_bg(e) for e in S.PRE], dtype=object); hbg = np.array([epoch_bg(e) for e in S.HL], dtype=object)
ptr, htr = R.ptr, R.htr; pday = np.array([e['day'] for e in S.PRE]); hday = R.hday
TEST = np.array(S.TEST)
out = {}
for d in ('TLR-ord', 'TLR-mix-ord'):
    res = dict(map=[], seq=[], comb=[])
    for s in range(300):
        rng = np.random.default_rng(77 + s); cal = set(rng.permutation(TEST)[:35])
        f = score_fn(lambda tr, dd: tr in cal)
        ep = np.isin(ptr, TEST) & ~np.isin(ptr, list(cal)); eh = np.isin(htr, TEST) & ~np.isin(htr, list(cal))
        mp = np.r_[[f(b) for b in pbg[ep]], [f(b) for b in hbg[eh]]]
        sq = np.r_[R.PMAX[d][ep], R.HMAX[d][eh]]
        y = np.r_[np.ones(ep.sum()), np.zeros(eh.sum())]
        cb = (rank(mp) + rank(sq)) / 2
        for k, v in (('map', mp), ('seq', sq), ('comb', cb)):
            res[k].append(S.auc(v[y == 1], v[y == 0]))
    days = sorted(set(hday)); early = set(days[:2])
    f = score_fn(lambda tr, dd: dd in early)
    ep = ~np.isin(pday, list(early)); eh = ~np.isin(hday, list(early))
    mp = np.r_[[f(b) for b in pbg[ep]], [f(b) for b in hbg[eh]]]; sq = np.r_[R.PMAX[d][ep], R.HMAX[d][eh]]
    y = np.r_[np.ones(ep.sum()), np.zeros(eh.sum())]
    out[d] = {k: [float(np.mean(v))] + np.percentile(v, [2.5, 97.5]).tolist() for k, v in res.items()}
    out[d]['later_days'] = dict(map=S.auc(mp[y == 1], mp[y == 0]), seq=S.auc(sq[y == 1], sq[y == 0]),
                                comb=S.auc(((rank(mp) + rank(sq)) / 2)[y == 1], ((rank(mp) + rank(sq)) / 2)[y == 0]))
    print(d, {k: [round(x, 3) for x in v] if isinstance(v, list) else {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out[d].items()})
# concentration of events
cnt = defaultdict(int)
for tr, d, b in ev_bg: cnt[b] += 1
nb = len(set(b for _, _, b in passes)); c = np.sort(list(cnt.values()))[::-1]
out['concentration'] = dict(n_groups=nb, n_events=int(c.sum()), share_in_top5pct=float(c[:int(round(0.05 * nb))].sum() / c.sum()),
                            share_in_top10pct=float(c[:int(round(0.10 * nb))].sum() / c.sum()), graph_nodes=len(G))
print(out['concentration'])
json.dump(out, open('out_v19/sncf/map.json', 'w'), indent=1)
