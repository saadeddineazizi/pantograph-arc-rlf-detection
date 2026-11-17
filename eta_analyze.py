"""Summarize the eta sensitivity runs -> out_v19/eta_summary.json"""
import json, glob
import numpy as np
E = ('0.95', '0.98', '0.99'); out = {}
main = lambda kind, tag, d, k='eprog': json.load(open(f'out_v19/{kind}/{tag}.json'))['results'][d]['0.001'][k]
for kind in ('held', 'phy', 'conf'):
    rows = []
    for f in sorted(glob.glob(f'out_v19/eta/{kind}_*.json')):
        r = json.load(open(f)); tag = f.split('/')[-1][len(kind) + 1:-5]
        rows.append(dict(tag=tag, **{e: r['results'][e] for e in E},
                         ref={d: main(kind, tag, d) for d in ('TLRmixSR', 'TLRsr', 'SRon', 'LQE')}))
    if not rows: continue
    o = dict(n=len(rows))
    for e in E:
        o[e] = dict(prog=float(np.mean([x[e]['eprog'] for x in rows])), det=float(np.mean([x[e]['det'] for x in rows])))
        for d in ('TLRsr', 'SRon', 'LQE'):
            diff = np.array([x[e]['eprog'] - x['ref'][d] for x in rows])
            o[e][f'vs_{d}'] = float(diff.mean()); o[e][f'win_{d}'] = int((diff < 0).sum())
    o['repro_099'] = float(max(abs(x['0.99']['eprog'] - x['ref']['TLRmixSR']) for x in rows))
    o['max_shift'] = float(max(abs(x[e]['eprog'] - x['0.99']['eprog']) for x in rows for e in E))
    if kind == 'phy':
        rail = [x for x in rows if 'rail' in x['tag']]
        o['rail_det'] = {e: float(np.mean([x[e]['det'] for x in rail])) for e in E}
    out[kind] = o
json.dump(out, open('out_v19/eta_summary.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
