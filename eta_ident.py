import sys, json, numpy as np
sys.path.insert(0, '.')
import sim as S, core as C, engine as E
res = {}
for i in (600, 601, 602, 603):
    cfg = S.draw_regime(i); S.ONSET_S = 60.0
    FF = C.fit_frames(E.fit_features('A', cfg, 61_000_000 + 10_000 * i + 21, 3000))
    row = {}
    for eta in (0.0, 0.5, 0.9, 0.95, 0.99):
        f = C.tlr_fit(FF, eta=eta); row[str(eta)] = dict(nll=f['nll'], theta=f['theta'])
    res[i] = row
    print(i, {e: (round(v['nll'], 5), [round(t, 4) for t in v['theta']]) for e, v in row.items()}, flush=True)
json.dump(res, open('out_v19/eta_ident.json', 'w'), indent=1)
