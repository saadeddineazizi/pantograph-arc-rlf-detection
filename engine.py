"""Evaluation of one regime under the frozen protocol (Section VI):
fit on healthy runs, run-level conformal thresholds on calibration runs, realized
false-alarm rate on independent test runs, detection metrics on degrading runs
(equal numbers for T = 30, 40, 50 s).  Seeds: fit vb+1, degrading vb+3,
calibration vb+10, test vb+20 (each split into chunks with a spawn key)."""
import json
import time
import numpy as np
from sim import simulate, K_H
import core as C

TS = (30, 40, 50)


def batches(sim, cfg, seed, n, K, T, fn, chunk=1000):
    out = {}
    for ci, a in enumerate(range(0, n, chunk)):
        m = min(chunk, n - a)
        rng = np.random.default_rng([seed, ci])
        D = sim_fn(sim, rng, m, K, cfg, T)
        F = C.feats(D)
        for k, v in fn(D, F).items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def sim_fn(sim, rng, n, K, cfg, T):
    return simulate(sim, rng, n, K, cfg, T)


def fit_features(sim, cfg, seed, n, chunk=1000):
    parts = []
    for ci, a in enumerate(range(0, n, chunk)):
        m = min(chunk, n - a)
        rng = np.random.default_rng([seed, ci])
        D = simulate(sim, rng, m, K_H, cfg, None)
        parts.append(C.feats(D))
    F = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return F


def evaluate(sim, cfg, vb, prep, scorer, sizes=(3000, 10000, 10000, 1200),
             alphas=C.ALPHAS, extra_healthy=None, extra_deg=None, log=None):
    t0 = time.time()
    nfit, ncal, nte, ndeg = sizes
    Ffit = fit_features(sim, cfg, vb + 1, nfit)
    ctx = prep(Ffit)
    hl = dict(loss=float(Ffit['lost'][:, 100:600].mean()),
              onset=float(Ffit['onset'][:, 100:600].mean()))
    if log: log(f'  fit done {time.time()-t0:.0f}s')

    def hfn(D, F):
        r = {k: C.healthy_max(p) for k, p in scorer(F, ctx).items()}
        if extra_healthy is not None:
            r.update(extra_healthy(D, F, ctx))
        return r
    cal = batches(sim, cfg, vb + 10, ncal, K_H, None, hfn)
    names = [k for k in cal if not k.startswith('_')]
    thr = {k: {a: C.threshold(cal[k], a) for a in alphas} for k in names}
    if log: log(f'  cal done {time.time()-t0:.0f}s')
    te = batches(sim, cfg, vb + 20, nte, K_H, None, hfn) if nte > 0 else None
    if log: log(f'  test done {time.time()-t0:.0f}s')

    taus = {}
    Tlist = []
    xd = {}
    for ti, T in enumerate(TS):
        def dfn(D, F):
            r = {}
            for k, p in scorer(F, ctx).items():
                for a in alphas:
                    r[f'{k}|{a}'] = C.alarm_frame(p, thr[k][a])
            if extra_deg is not None:
                r.update(extra_deg(D, F, ctx))
            return r
        nT = ndeg // len(TS)
        r = batches(sim, cfg, vb + 3 + 1000 * ti, nT, 600 + 10 * T, T, dfn)
        for k, v in r.items():
            (xd if k.startswith('_') else taus).setdefault(k, []).append(v)
        Tlist.append(np.full(nT, T))
    Tarr = np.concatenate(Tlist)
    res = {}
    for k in names:
        res[k] = {}
        for a in alphas:
            tau = np.concatenate(taus[f'{k}|{a}'])
            m = C.deg_metrics(tau, Tarr)
            if te is not None:
                m['fa_test'] = float((te[k] > thr[k][a]).mean())
            m['h'] = thr[k][a]
            res[k][str(a)] = m
    out = dict(results=res, healthy=hl, ctx_summary=ctx.get('summary', {}),
               sizes=list(sizes), seconds=time.time() - t0)
    if extra_healthy is not None:
        out['_cal_extra'] = {k: v for k, v in cal.items() if k.startswith('_')}
    if extra_deg is not None:
        out['_deg_extra'] = {k: np.concatenate(v) for k, v in xd.items()}
    return out


# ------------------------------------------------------------------ standard detectors
def prep_base(F, etas=()):
    fit = C.tlr_fit(C.fit_frames(F))
    ctx = dict(fit=fit, pools=C.make_pools(F), summary=dict(fit=fit['theta']))
    for eta in etas:
        f = C.tlr_fit(C.fit_frames(F), eta=eta)
        ctx[f'fit_eta{eta}'] = f
        ctx['summary'][f'fit_eta{eta}'] = f['theta']
    return ctx


def score_base(F, ctx, etas=(), sre=True):
    d = C.FROZEN['delta']
    lr, ll = C.llr(F, ctx['fit'], d)
    out = dict(TLR=C.cusum(lr + ll), TLRsr=C.srw(lr + ll), TLRloss=C.cusum(ll), TLRdips=C.cusum(lr))
    if sre:
        # windowed (W = 300) like TLR-SR: a plain SR grows like k in healthy data and would
        # alarm by the passage of time once a degrading run is longer than the calibrated window
        out['SRe'] = C.srw(np.log(C.sre_evalues(F, ctx['pools'], onset=False)))
        out['SRon'] = C.srw(np.log(C.sre_evalues(F, ctx['pools'], onset=True)))
    for eta in etas:
        lr, ll = C.llr(F, ctx[f'fit_eta{eta}'], d)
        out[f'TLRmix{eta}'] = C.cusum(lr + ll)
        out[f'TLRmixSR{eta}'] = C.srw(lr + ll)
    return out


def save(obj, path):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, default=conv)
