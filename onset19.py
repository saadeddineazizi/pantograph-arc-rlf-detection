"""v19 robustness check (declared 17 Nov 2026 after the confirmatory tests, before running it):
does the advantage of the near no-forgetting detectors (LQE, R4 with frozen constants) depend on the ramp
starting only 30 s after monitoring begins?  Same frozen constants; the onset is delayed by D seconds
(D in {0, 120, 300}); healthy calibration runs last until the onset, so the run-level threshold covers the
same monitoring length as the degrading runs.  Regimes: confirmatory INDEX 600-639 (Simulator A).
Sizes: fit 3000 (60 s runs, as always), calibration 3000, degrading 600 (200 per T), seeds as eval19 + 7.
Expectation written before running: the progress of LQE and R4 rises with D more than that of
TLR-mix-SR, whose windowed SR form forgets after 300 frames.  Reported whatever the outcome.
usage: python3 onset19.py <INDEX> <D>"""
import sys, json, time
import numpy as np
import sim as S, core as C, engine as E, baselines as Bs, eval19 as V

DETS = ('LQE', 'R4', 'TLRmixSR', 'TLRmix', 'TLRsr', 'SRon')
ALPHAS = (0.01, 0.001)


def main(i, D):
    t0 = time.time()
    S.ONSET_S = 60.0 + D
    onset = int(round(10 * S.ONSET_S))
    cfg = S.draw_regime(i); vb = 61_000_000 + 10_000 * i + 7
    Ffit = E.fit_features('A', cfg, vb + 1, 3000)          # 60 s healthy runs, no ramp
    ctx = V.prep(Ffit)
    def sc(F):
        k = V.KZ; d = V.FZ['delta']; out = {}
        out['LQE'] = Bs.lqe(F, k['LQE'], ctx['lqe'][k['LQE']]); out['R4'] = Bs.r4(F, k['R4'])
        out['SRon'] = C.srw(np.log(C.sre_evalues(F, ctx['pools'], onset=True)))
        lr, ll = C.llr(F, ctx['fit'], d); out['TLRsr'] = C.srw(lr + ll)
        lr, ll = C.llr(F, ctx['fit_mix'], d); out['TLRmix'] = C.cusum(lr + ll); out['TLRmixSR'] = C.srw(lr + ll)
        return out
    def hfn(Dd, F):
        return {k: p[:, :onset - C.K0].max(axis=1) for k, p in sc(F).items()}
    cal = E.batches('A', cfg, vb + 10, 3000, onset, None, hfn)
    thr = {k: {a: C.threshold(cal[k], a) for a in ALPHAS} for k in DETS}
    taus, Ts = {}, []
    for ti, T in enumerate(E.TS):
        def dfn(Dd, F):
            return {f'{k}|{a}': C.alarm_frame(p, thr[k][a]) for k, p in sc(F).items() for a in ALPHAS}
        r = E.batches('A', cfg, vb + 3 + 1000 * ti, 200, onset + 10 * T, T, dfn)
        for k, v in r.items(): taus.setdefault(k, []).append(v)
        Ts.append(np.full(200, T))
    Tarr = np.concatenate(Ts)
    res = {k: {str(a): C.deg_metrics(np.concatenate(taus[f'{k}|{a}']), Tarr, onset=onset) for a in ALPHAS} for k in DETS}
    out = dict(index=i, D=D, onset_frame=onset, regime=cfg, results=res, seconds=time.time() - t0)
    E.save(out, f'out_v19/onset/{i}_D{D}.json')
    print({k: round(v['0.001']['eprog'], 3) for k, v in res.items()}, round(out['seconds']))


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
