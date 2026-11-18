import json, glob
import numpy as np
R = [json.load(open(f)) for f in sorted(glob.glob('out_v19/theoryW/6*.json'))]
rng = np.random.default_rng(5); out = dict(n=len(R))
for a in ('0.001', '0.01'):
    o = {}
    for j, m in enumerate((30, 330)):
        pr = {w: np.array([r['rows'][j][f'W{w}|{a}']['prog'] for r in R]) for w in (150, 300, 600)}
        de = {w: np.array([r['rows'][j][f'W{w}|{a}']['det'] for r in R]) for w in (150, 300, 600)}
        idx = rng.integers(0, len(R), (20000, len(R)))
        d = {}
        for w in (150, 600):
            diff = pr[w] - pr[300]; b = diff[idx].mean(1)
            d[w] = [float(diff.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975)), int((diff < 0).sum())]
        o[m] = dict(prog={w: float(pr[w].mean()) for w in pr}, det={w: float(de[w].mean()) for w in de}, diff_vs_300=d)
    out[a] = o
# MFL-SR with W in {150, 600} against LQE at 330 s (LQE from the healthy-period check, same seeds and protocol)
T3 = {r['index']: r for r in [json.load(open(f)) for f in glob.glob('out_v19/theory3/6*.json')]}
j3 = [row['m'] for row in next(iter(T3.values()))['rows']].index(330)
for w in (150, 300, 600):
    lq = np.array([T3[r['index']]['rows'][j3]['LQE|0.001']['prog'] for r in R]); mw = np.array([r['rows'][1][f'W{w}|0.001']['prog'] for r in R])
    out.setdefault('vs_lqe_330', {})[w] = int((mw < lq).sum())
json.dump(out, open('out_v19/theoryW_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
