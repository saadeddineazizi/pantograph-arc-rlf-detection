"""Freeze all v19 constants from the development files (argmin of mean eprog at alpha 1e-3 over the six
development regimes, all three grid passes pooled). Writes out_v19/frozen.json."""
import json, glob, numpy as np
acc = {}
for f in glob.glob('out_v19/dev/*.json') + glob.glob('out_mixdev/mixdev_*.json'):
    for k, v in json.load(open(f))['results'].items():
        acc.setdefault(k, []).append(v['0.001']['eprog'])
mean = {k: float(np.mean(v)) for k, v in acc.items() if len(v) == 6}
fam = lambda p: {k: m for k, m in mean.items() if k.split('_')[0] == p}
fz = {}
for p in ('R1', 'R4', 'L1', 'LQE', 'CC'):
    k = min(fam(p), key=fam(p).get); fz[p] = float(k.split('_')[1]); fz[p + '_dev'] = fam(p)[k]
k = min(fam('CR'), key=fam('CR').get); _, g, pf = k.split('_'); fz['CR'] = [float(g), float(pf)]; fz['CR_dev'] = fam('CR')[k]
for form in ('TLRmix', 'TLRmixSR'):
    d = {k: m for k, m in mean.items() if k.startswith(form) and k[len(form):][:1].isdigit()}
    k = min(d, key=d.get); fz['eta_' + form] = float(k[len(form):]); fz[form + '_dev'] = d[k]
fz['delta'] = 0.003; fz['sr_window'] = 300; fz['sre_lam'] = 0.002; fz['sre_q'] = 0.05; fz['lstm_rate'] = 0.05
fz['edge_note'] = ('R1, R4, L1, CR, LQE optima moved to the no-forgetting limit over three development passes; '
                   'the frozen values are the smallest tested; the eta curve is flat above 0.95.')
json.dump(fz, open('out_v19/frozen.json', 'w'), indent=1)
print(json.dumps(fz, indent=1))
