"""Computing cost per frame and per train of each detector (offline, vectorized NumPy on one CPU core)."""
import time, json
import numpy as np
import sim as S, core as C, engine as E, baselines as Bs, eval19 as V, lstm19
cfg = S.draw_regime(600); vb = 61_000_000 + 10_000 * 600 + 11
S.ONSET_S = 60.0
ctx = V.prep(E.fit_features('A', cfg, vb + 1, 3000)); k = V.KZ; d = V.FZ['delta']
res = {}
def fn(D, F):
    n, K = F['x'].shape if 'x' in F else next(iter(F.values())).shape
    jobs = {'LQE': lambda: Bs.lqe(F, k['LQE'], ctx['lqe'][k['LQE']]),
            'OR (R4 and L1)': lambda: (Bs.r4(F, k['R4']), Bs.l1(F, ctx['p0'], k['L1'])),
            'SRon': lambda: C.srw(np.log(C.sre_evalues(F, ctx['pools'], onset=True))),
            'LSTM': lambda: lstm19.score(F),
            'TLR-SR': lambda: C.srw(sum(C.llr(F, ctx['fit'], d))),
            'MFL (CUSUM)': lambda: C.cusum(sum(C.llr(F, ctx['fit_mix'], d))),
            'MFL-SR': lambda: C.srw(sum(C.llr(F, ctx['fit_mix'], d)))}
    for name, f in jobs.items():
        t = time.perf_counter(); f(); dt = time.perf_counter() - t
        res.setdefault(name, []).append(dt / (n * (K - C.K0)) * 1e6)
    return {}
E.batches('A', cfg, vb + 99, 400, 600, None, fn)
out = {k2: float(np.median(v)) for k2, v in res.items()}
json.dump(out, open('out_v19/timing.json', 'w'), indent=1); print(out)
