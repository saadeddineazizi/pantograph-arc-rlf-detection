"""Numerical check of the regime-flip theorem (18 Nov 2026).
Claim: with run-level conformal thresholds over m frames of healthy monitoring,
 (a) the no-forgetting moving average (LQE, EWMA with r = 1e-4 started at 0, i.e. a running sum)
     needs a threshold whose excess over its healthy drift grows like sqrt(m) (Bachelier-Levy law),
 (b) the windowed Shiryaev-Roberts mixture statistic (TLR-mix-SR, window W) is stationary after W frames
     with a near-Gaussian tail, so its threshold grows at most like sqrt(log m),
 (c) hence the alarm delay of LQE under a ramp grows with m while that of TLR-mix-SR is nearly flat,
     and the two cross at a monitoring length predicted from healthy statistics and the mean path.
Same frozen constants and simulator as the paper (Simulator A, registered regimes INDEX 600-619).
usage: python3 theory_flip.py <INDEX>   -> out_v19/theory/<INDEX>.json"""
import sys, json, time
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import sim as S, core as C, engine as E, baselines as Bs, eval19 as V

MS = (30, 60, 120, 240, 480)       # seconds of healthy monitoring before the ramp
ALPHAS = (0.01, 0.001)
T = 40                              # ramp length in s


def bl_threshold(c0, s2, N, alpha):
    """Bachelier-Levy: P(max_{t<=N} (c0 t + sigma W_t) > h) = alpha, solved for h."""
    sd = np.sqrt(s2 * N)
    def p(h):
        a = 1 - norm.cdf((h - c0 * N) / sd)
        b = np.exp(np.clip(2 * c0 * h / s2, -700, 700)) * norm.cdf((-h - c0 * N) / sd)
        return a + b - alpha
    return brentq(p, max(c0 * N, 0) + 1e-9, abs(c0) * N + 20 * sd)


def main(i):
    t0 = time.time()
    cfg = S.draw_regime(i); vb = 61_000_000 + 10_000 * i + 11
    S.ONSET_S = 60.0
    Ffit = E.fit_features('A', cfg, vb + 1, 3000)
    ctx = V.prep(Ffit); k = V.KZ; d = V.FZ['delta']
    def paths(F):
        lr, ll = C.llr(F, ctx['fit_mix'], d)
        return {'LQE': Bs.lqe(F, k['LQE'], ctx['lqe'][k['LQE']]), 'TLRmixSR': C.srw(lr + ll)}
    out = dict(index=i, regime=cfg, rows=[])
    for m in MS:
        S.ONSET_S = 30.0 + m                       # monitoring starts at K0 = 30 s
        onset = int(round(10 * S.ONSET_S)); N = onset - C.K0
        def hfn(D, F):
            P = paths(F)
            z = P['LQE'][:, :N]
            return {'LQE': z.max(1), 'TLRmixSR': P['TLRmixSR'][:, :N].max(1),
                    'z0': z[:, 0], 'zN': z[:, N - 1], 'zh': z[:, N // 2]}
        cal = E.batches('A', cfg, vb + 10 + m, 3000, onset, None, hfn)
        thr = {a: {k2: float(C.threshold(cal[k2], a)) for k2 in ('LQE', 'TLRmixSR')} for a in ALPHAS}
        # healthy drift and long-run variance of the LQE statistic, from the second half of the path
        inc = cal['zN'] - cal['zh']; L = N - 1 - N // 2
        c0 = float(inc.mean() / L); s2 = float(inc.var() / L); z0 = float(cal['z0'].mean())
        pred = {a: z0 + bl_threshold(c0, s2, N, a) for a in ALPHAS}
        def dfn(D, F):
            P = paths(F)
            r = {}
            for a in ALPHAS:
                for k2 in ('LQE', 'TLRmixSR'):
                    r[f'{k2}|{a}'] = C.alarm_frame(P[k2], thr[a][k2])
            r['mLQE'] = P['LQE'][:, N:].mean(0, keepdims=True); r['mSR'] = P['TLRmixSR'][:, N:].mean(0, keepdims=True)
            return r
        dg = E.batches('A', cfg, vb + 3 + m, 400, onset + 10 * T, T, dfn)
        row = dict(m=m, N=N, c0=c0, s2=s2, z0=z0, thr=thr, pred_LQE=pred)
        mp = {'LQE': dg['mLQE'].mean(0), 'TLRmixSR': dg['mSR'].mean(0)}
        for a in ALPHAS:
            for k2 in ('LQE', 'TLRmixSR'):
                al = dg[f'{k2}|{a}']                     # absolute frame of the alarm, -1 if none
                tau = al - onset                          # frames after onset
                ok = (al >= 0) & (tau >= 0) & (tau < 10 * T)
                cross = np.nonzero(mp[k2] > thr[a][k2])[0]
                row[f'{k2}|{a}'] = dict(det=float(ok.mean()), pre=float(((al >= 0) & (tau < 0)).mean()),
                                        delay=float(tau[ok].mean()) if ok.any() else None,
                                        prog=float(np.where(ok, tau / (10 * T), 1.0).mean()),
                                        cross=int(cross[0]) if cross.size else None)
        out['rows'].append(row)
        print(m, {a: (round(thr[a]['LQE'], 2), round(pred[a], 2), round(thr[a]['TLRmixSR'], 2)) for a in ALPHAS},
              {k2: (round(row[f'{k2}|0.01']['prog'], 3), row[f'{k2}|0.01']['cross']) for k2 in ('LQE', 'TLRmixSR')}, flush=True)
    out['seconds'] = time.time() - t0
    E.save(out, f'out_v19/theory/{i}.json')


if __name__ == '__main__':
    main(int(sys.argv[1]))
