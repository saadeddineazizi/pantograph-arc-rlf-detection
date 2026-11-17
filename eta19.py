"""v19 eta sensitivity (added after review): TLR-mix-SR with eta in {0.95, 0.98, 0.99} on the
15 held-out regimes, the 6 Simulator B regimes and the 100 pre-registered random regimes, with the
same seeds as eval19 (eta=0.99 reproduces the main results). -> out_v19/eta/<kind>_<tag>.json"""
import sys, json, time, os
import numpy as np
import core as C, engine as E
from sim import regime_cfg, seed_held, draw_regime, K_H
FZ = json.load(open('out_v19/frozen.json'))
ETAS = (0.95, 0.98, 0.99); A = 0.001

def run(sim, cfg, vb, sizes=(3000, 10000, 1200)):
    t0 = time.time(); nfit, ncal, ndeg = sizes
    FF = C.fit_frames(E.fit_features(sim, cfg, vb + 1, nfit))
    fits = {e: C.tlr_fit(FF, eta=e) for e in ETAS}
    def sc(F):
        out = {}
        for e, f in fits.items():
            lr, ll = C.llr(F, f, FZ['delta']); out[str(e)] = C.srw(lr + ll)
        return out
    cal = E.batches(sim, cfg, vb + 10, ncal, K_H, None, lambda D, F: {k: C.healthy_max(p) for k, p in sc(F).items()})
    thr = {k: C.threshold(cal[k], A) for k in cal}
    taus, Ts = {}, []
    for ti, T in enumerate(E.TS):
        nT = ndeg // len(E.TS)
        r = E.batches(sim, cfg, vb + 3 + 1000 * ti, nT, 600 + 10 * T, T, lambda D, F: {k: C.alarm_frame(p, thr[k]) for k, p in sc(F).items()})
        for k, v in r.items(): taus.setdefault(k, []).append(v)
        Ts.append(np.full(nT, T))
    Tarr = np.concatenate(Ts)
    res = {k: C.deg_metrics(np.concatenate(v), Tarr) for k, v in taus.items()}
    return dict(results=res, fits={str(e): fits[e]['theta'] for e in ETAS}, seconds=time.time() - t0)

if __name__ == '__main__':
    kind = sys.argv[1]
    if kind in ('held', 'phy'):
        reg, cell = sys.argv[2], sys.argv[3]; sim, cfg = regime_cfg(reg, cell); vb = seed_held(reg, cell); tag = f'{reg}_{cell}'
    else:
        i = int(sys.argv[2]); cfg = draw_regime(i); sim = 'A'; vb = 61_000_000 + 10_000 * i; tag = str(i)
    out = f'out_v19/eta/{kind}_{tag}.json'
    if os.path.exists(out): sys.exit()
    r = run(sim, cfg, vb); r['regime'] = [kind, tag]
    E.save(r, out)
    print(tag, {k: (round(v['eprog'], 3), round(v['det'], 3)) for k, v in r['results'].items()}, round(r['seconds']))
