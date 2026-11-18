"""Monitoring-length check on all 100 pre-registered regimes with windowed baselines (18 Nov 2026, v21).
Healthy periods m in MS3 before a 40 s ramp; thresholds from 3000 healthy runs of the same length; 400 degrading runs.
Statistics: LQE (frozen, rate 1e-4) and LQEw (rate 1/300); R4, L1 and their windowed CUSUM forms (W = 300 frames);
OR / ORw (R4 or L1 at alpha/2 each); TLR-mix CUSUM and TLR-mix-SR (windowed SR, W = 300).
usage: python3 theory_flip3.py <INDEX> -> out_v19/theory3/<INDEX>.json"""
import sys, time
import numpy as np
import sim as S, core as C, engine as E, baselines as Bs, eval19 as V
from theory_flip import bl_threshold
from wmin import wcusum

MS3 = (30, 60, 150, 330, 480); ALPHAS = (0.01, 0.001); T = 40; W = 300; RW = 1 / W
SINGLE = ('LQE', 'LQEw', 'R4', 'R4w', 'L1', 'L1w', 'TLRmix', 'TLRmixSR')


def main(i):
    t0 = time.time()
    cfg = S.draw_regime(i); vb = 61_000_000 + 10_000 * i + 11
    S.ONSET_S = 60.0
    Ffit = E.fit_features('A', cfg, vb + 1, 3000)
    ctx = V.prep(Ffit); k = V.KZ; d = V.FZ['delta']
    stw = Bs.lqe_stats(Ffit, RW)
    dR = k['R4']; p0 = ctx['p0']; p1 = min(p0 + k['L1'], 0.5)

    def paths(F):
        lr, ll = C.llr(F, ctx['fit_mix'], d); lam = lr + ll
        u = Bs._resid(F); incR = dR * (u - dR / 2)
        incL = np.where(F['onset'], np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))
        return {'LQE': Bs.lqe(F, k['LQE'], ctx['lqe'][k['LQE']]), 'LQEw': Bs.lqe(F, RW, stw),
                'R4': C.cusum(incR), 'R4w': wcusum(incR, W), 'L1': C.cusum(incL), 'L1w': wcusum(incL, W),
                'TLRmix': C.cusum(lam), 'TLRmixSR': C.srw(lam)}
    out = dict(index=i, mech=cfg['mech'], rows=[])
    for m in MS3:
        S.ONSET_S = 30.0 + m
        onset = int(round(10 * S.ONSET_S)); N = onset - C.K0

        def hfn(D, F):
            P = paths(F); r = {k2: P[k2][:, :N].max(1) for k2 in SINGLE}
            z = P['LQE'][:, :N]; r.update(z0=z[:, 0], zN=z[:, N - 1], zh=z[:, N // 2])
            return r
        cal = E.batches('A', cfg, vb + 10 + m, 3000, onset, None, hfn)
        thr = {}
        for a in ALPHAS:
            thr[a] = {k2: float(C.threshold(cal[k2], a)) for k2 in SINGLE}
            for o, (x, y) in (('OR', ('R4', 'L1')), ('ORw', ('R4w', 'L1w'))):
                thr[a][o] = (float(C.threshold(cal[x], a / 2)), float(C.threshold(cal[y], a / 2)))
            thr[a]['ORfa'] = float(((cal['R4'] > thr[a]['OR'][0]) | (cal['L1'] > thr[a]['OR'][1])).mean())
        inc = cal['zN'] - cal['zh']; L = N - 1 - N // 2
        c0 = float(inc.mean() / L); s2 = float(inc.var() / L); z0 = float(cal['z0'].mean())
        pred = {a: z0 + bl_threshold(c0, s2, N, a) for a in ALPHAS}

        def dfn(D, F):
            P = paths(F); r = {}
            for a in ALPHAS:
                for k2 in SINGLE: r[f'{k2}|{a}'] = C.alarm_frame(P[k2], thr[a][k2])
                for o, (x, y) in (('OR', ('R4', 'L1')), ('ORw', ('R4w', 'L1w'))):
                    ax = C.alarm_frame(P[x], thr[a][o][0]); ay = C.alarm_frame(P[y], thr[a][o][1])
                    big = 10 ** 9
                    al = np.minimum(np.where(ax < 0, big, ax), np.where(ay < 0, big, ay))
                    r[f'{o}|{a}'] = np.where(al == big, -1, al)
            return r
        dg = E.batches('A', cfg, vb + 3 + m, 400, onset + 10 * T, T, dfn)
        row = dict(m=m, N=N, c0=c0, s2=s2, z0=z0, zN=float(cal['zN'].mean()), thr=thr, pred_LQE=pred)
        for a in ALPHAS:
            for k2 in SINGLE + ('OR', 'ORw'):
                al = dg[f'{k2}|{a}']; tau = al - onset
                ok = (al >= 0) & (tau >= 0) & (tau < 10 * T)
                row[f'{k2}|{a}'] = dict(det=float(ok.mean()), prog=float(np.where(ok, tau / (10 * T), 1.0).mean()),
                                        pre=float(((al >= 0) & (tau < 0)).mean()))
        out['rows'].append(row)
        print(i, m, {k2: round(row[f'{k2}|0.001']['prog'], 3) for k2 in SINGLE + ('OR', 'ORw')}, flush=True)
    out['seconds'] = time.time() - t0
    E.save(out, f'out_v19/theory3/{i}.json')


if __name__ == '__main__':
    main(int(sys.argv[1]))
