"""Second theorem check (18 Nov 2026): adds
 - LQEw: the same moving average with forgetting rate 1/300 (memory equal to the SR window) -> falsification:
   Theorem 1 predicts that its threshold stays nearly flat and that it does NOT lose ground as m grows;
 - TLRmix: the CUSUM form of the mixture (reflected at 0, stationary under H0 -> threshold grows like log m);
 - the mean degrading path of LQE after onset, for the finite-ramp (semi-analytic) prediction of the delay.
usage: python3 theory_flip2.py <INDEX>  -> out_v19/theory2/<INDEX>.json"""
import sys, time
import numpy as np
import sim as S, core as C, engine as E, baselines as Bs, eval19 as V
from theory_flip import bl_threshold, MS, ALPHAS, T

RW = 1 / 300
KEYS = ('LQE', 'LQEw', 'TLRmixSR', 'TLRmix')


def main(i):
    t0 = time.time()
    cfg = S.draw_regime(i); vb = 61_000_000 + 10_000 * i + 11
    S.ONSET_S = 60.0
    Ffit = E.fit_features('A', cfg, vb + 1, 3000)
    ctx = V.prep(Ffit); k = V.KZ; d = V.FZ['delta']
    stw = Bs.lqe_stats(Ffit, RW)
    def paths(F):
        lr, ll = C.llr(F, ctx['fit_mix'], d)
        lam = lr + ll
        return {'LQE': Bs.lqe(F, k['LQE'], ctx['lqe'][k['LQE']]), 'LQEw': Bs.lqe(F, RW, stw),
                'TLRmixSR': C.srw(lam), 'TLRmix': C.cusum(lam)}
    out = dict(index=i, rows=[])
    for m in MS:
        S.ONSET_S = 30.0 + m
        onset = int(round(10 * S.ONSET_S)); N = onset - C.K0
        def hfn(D, F):
            P = paths(F); r = {k2: P[k2][:, :N].max(1) for k2 in KEYS}
            z = P['LQE'][:, :N]; r.update(z0=z[:, 0], zN=z[:, N - 1], zh=z[:, N // 2])
            return r
        cal = E.batches('A', cfg, vb + 10 + m, 3000, onset, None, hfn)
        thr = {a: {k2: float(C.threshold(cal[k2], a)) for k2 in KEYS} for a in ALPHAS}
        inc = cal['zN'] - cal['zh']; L = N - 1 - N // 2
        c0 = float(inc.mean() / L); s2 = float(inc.var() / L); z0 = float(cal['z0'].mean())
        pred = {a: z0 + bl_threshold(c0, s2, N, a) for a in ALPHAS}
        def dfn(D, F):
            P = paths(F); r = {}
            for a in ALPHAS:
                for k2 in KEYS: r[f'{k2}|{a}'] = C.alarm_frame(P[k2], thr[a][k2])
            for k2 in KEYS: r['m' + k2] = P[k2][:, N:N + 10 * T].mean(0, keepdims=True)
            return r
        dg = E.batches('A', cfg, vb + 3 + m, 400, onset + 10 * T, T, dfn)
        row = dict(m=m, N=N, c0=c0, s2=s2, z0=z0, zN=float(cal['zN'].mean()), thr=thr, pred_LQE=pred,
                   mean_path={k2: dg['m' + k2].mean(0).tolist() for k2 in KEYS})
        for a in ALPHAS:
            for k2 in KEYS:
                al = dg[f'{k2}|{a}']; tau = al - onset
                ok = (al >= 0) & (tau >= 0) & (tau < 10 * T)
                mp = np.array(row['mean_path'][k2]); cr = np.nonzero(mp > thr[a][k2])[0]
                row[f'{k2}|{a}'] = dict(det=float(ok.mean()), prog=float(np.where(ok, tau / (10 * T), 1.0).mean()),
                                        cross=int(cr[0]) if cr.size else None)
        out['rows'].append(row)
        print(i, m, {k2: (round(thr['0.01' if False else 0.01][k2], 2), round(row[f'{k2}|0.01']['prog'], 3)) for k2 in KEYS}, flush=True)
    out['seconds'] = time.time() - t0
    E.save(out, f'out_v19/theory2/{i}.json')


if __name__ == '__main__':
    main(int(sys.argv[1]))
