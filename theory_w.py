"""Sensitivity of MFL-SR to the window W (v24): W in {150, 300, 600} frames, healthy periods 30 s and 330 s,
100 pre-registered regimes, thresholds from 3000 healthy runs of the same length, 400 degrading runs, 40 s ramp.
usage: python3 theory_w.py <INDEX> -> out_v19/theoryW/<INDEX>.json"""
import sys, time
import numpy as np
import sim as S, core as C, engine as E, eval19 as V

MSW = (30, 330); WS = (150, 300, 600); ALPHAS = (0.01, 0.001); T = 40


def main(i):
    t0 = time.time()
    cfg = S.draw_regime(i); vb = 61_000_000 + 10_000 * i + 11
    S.ONSET_S = 60.0
    Ffit = E.fit_features('A', cfg, vb + 1, 3000)
    ctx = V.prep(Ffit); d = V.FZ['delta']

    def paths(F):
        lr, ll = C.llr(F, ctx['fit_mix'], d); lam = lr + ll
        return {f'W{w}': C.srw(lam, w) for w in WS}
    out = dict(index=i, mech=cfg['mech'], rows=[])
    for m in MSW:
        S.ONSET_S = 30.0 + m
        onset = int(round(10 * S.ONSET_S)); N = onset - C.K0
        cal = E.batches('A', cfg, vb + 10 + m, 3000, onset, None, lambda D, F: {k: v[:, :N].max(1) for k, v in paths(F).items()})
        thr = {a: {k: float(C.threshold(cal[k], a)) for k in cal} for a in ALPHAS}
        def dfn(D, F):
            P = paths(F)
            return {f'{k}|{a}': C.alarm_frame(P[k], thr[a][k]) for a in ALPHAS for k in P}
        dg = E.batches('A', cfg, vb + 3 + m, 400, onset + 10 * T, T, dfn)
        row = dict(m=m, thr=thr)
        for key, al in dg.items():
            tau = al - onset; ok = (al >= 0) & (tau >= 0) & (tau < 10 * T)
            row[key] = dict(det=float(ok.mean()), prog=float(np.where(ok, tau / (10 * T), 1.0).mean()), n=int(len(al)), ndet=int(ok.sum()))
        out['rows'].append(row)
        print(i, m, {k: round(v['prog'], 3) for k, v in row.items() if '|0.001' in k}, flush=True)
    out['seconds'] = time.time() - t0
    E.save(out, f'out_v19/theoryW/{i}.json')


if __name__ == '__main__':
    main(int(sys.argv[1]))
