"""v19 evaluation of the whole frozen detector set on one regime.
usage: python3 eval19.py held <reg> <cell> | phy <reg> <cell> | conf <INDEX> | rand <INDEX> | devfresh <reg> <cell>
Detectors: R1 R4 L1 OR CR LQE CC SRe SRon LSTM TLR TLRsr TLRloss TLRdips TLRmix TLRmixSR.
OR alarms when R4 or L1 exceeds its own threshold at alpha/2."""
import sys, json, time
import numpy as np
import core as C, engine as E, baselines as Bs, lstm19
from sim import regime_cfg, seed_held, seed_dev, draw_regime, K_H

FZ = json.load(open('out_v19/frozen.json'))
KZ = dict(R1=FZ['R1'], R4=FZ['R4'], L1=FZ['L1'], CR=tuple(FZ['CR']), LQE=FZ['LQE'], CC=FZ['CC'])
ALPHAS = (0.05, 0.01, 0.001)
HALF = tuple(a / 2 for a in ALPHAS)


def prep(F):
    ctx = Bs.prep(F, consts=KZ)
    FF = C.fit_frames(F)
    ctx['fit'] = C.tlr_fit(FF)
    ctx['fit_mix'] = C.tlr_fit(FF, eta=FZ['eta_TLRmix'])
    ctx['fit_mixsr'] = ctx['fit_mix'] if FZ['eta_TLRmixSR'] == FZ['eta_TLRmix'] else C.tlr_fit(FF, eta=FZ['eta_TLRmixSR'])
    ctx['summary'] = dict(fit=ctx['fit']['theta'], fit_mix=ctx['fit_mix']['theta'], p0=ctx['p0'])
    return ctx


def scorer(F, ctx):
    k = KZ
    out = dict(R1=Bs.r1(F, k['R1']), R4=Bs.r4(F, k['R4']), L1=Bs.l1(F, ctx['p0'], k['L1']),
               CR=Bs.cr(F, *k['CR']), LQE=Bs.lqe(F, k['LQE'], ctx['lqe'][k['LQE']]), CC=Bs.cc(F, ctx['pools'], k['CC']))
    out['SRe'] = C.srw(np.log(C.sre_evalues(F, ctx['pools'], onset=False)))
    out['SRon'] = C.srw(np.log(C.sre_evalues(F, ctx['pools'], onset=True)))
    out['LSTM'] = lstm19.score(F)
    d = FZ['delta']
    lr, ll = C.llr(F, ctx['fit'], d)
    out.update(TLR=C.cusum(lr + ll), TLRsr=C.srw(lr + ll), TLRloss=C.cusum(ll), TLRdips=C.cusum(lr))
    lr, ll = C.llr(F, ctx['fit_mix'], d); out['TLRmix'] = C.cusum(lr + ll)
    lr, ll = C.llr(F, ctx['fit_mixsr'], d); out['TLRmixSR'] = C.srw(lr + ll)
    return out


def run(sim, cfg, vb, sizes=(3000, 10000, 10000, 1200)):
    t0 = time.time()
    nfit, ncal, nte, ndeg = sizes
    Ffit = E.fit_features(sim, cfg, vb + 1, nfit)
    ctx = prep(Ffit)
    hfn = lambda D, F: {k: C.healthy_max(p) for k, p in scorer(F, ctx).items()}
    cal = E.batches(sim, cfg, vb + 10, ncal, K_H, None, hfn)
    names = list(cal)
    alist = ALPHAS + HALF
    thr = {k: {a: C.threshold(cal[k], a) for a in alist} for k in names}
    te = E.batches(sim, cfg, vb + 20, nte, K_H, None, hfn) if nte else None
    taus, Ts = {}, []
    for ti, T in enumerate(E.TS):
        def dfn(D, F):
            r = {}
            for k, p in scorer(F, ctx).items():
                for a in (alist if k in ('R4', 'L1') else ALPHAS):
                    r[f'{k}|{a}'] = C.alarm_frame(p, thr[k][a])
            return r
        nT = ndeg // len(E.TS)
        r = E.batches(sim, cfg, vb + 3 + 1000 * ti, nT, 600 + 10 * T, T, dfn)
        for k, v in r.items(): taus.setdefault(k, []).append(v)
        Ts.append(np.full(nT, T))
    Tarr = np.concatenate(Ts)
    tau = {k: np.concatenate(v) for k, v in taus.items()}
    res = {}
    for k in names:
        res[k] = {}
        for a in ALPHAS:
            m = C.deg_metrics(tau[f'{k}|{a}'], Tarr)
            if te is not None: m['fa_test'] = float((te[k] > thr[k][a]).mean())
            m['h'] = thr[k][a]; res[k][str(a)] = m
    res['OR'] = {}
    for a, h2 in zip(ALPHAS, HALF):
        t1, t2 = tau[f'R4|{h2}'], tau[f'L1|{h2}']
        t = np.where(t1 < 0, t2, np.where(t2 < 0, t1, np.minimum(t1, t2)))
        m = C.deg_metrics(t, Tarr)
        if te is not None: m['fa_test'] = float(((te['R4'] > thr['R4'][h2]) | (te['L1'] > thr['L1'][h2])).mean())
        res['OR'][str(a)] = m
    hl = dict(loss=float(Ffit['lost'][:, 100:600].mean()), onset=float(Ffit['onset'][:, 100:600].mean()))
    return dict(results=res, healthy=hl, ctx_summary=ctx['summary'], sizes=list(sizes), seconds=time.time() - t0)


if __name__ == '__main__':
    kind = sys.argv[1]
    if kind in ('held', 'phy', 'devfresh'):
        reg, cell = sys.argv[2], sys.argv[3]
        sim, cfg = regime_cfg(reg, cell)
        vb = seed_held(reg, cell) if kind != 'devfresh' else seed_dev(reg, cell) + 500
        tag = f'{reg}_{cell}'; info = [reg, cell]
    else:
        i = int(sys.argv[2]); cfg = draw_regime(i); sim = 'A'
        vb = 61_000_000 + 10_000 * i; tag = str(i); info = cfg
    r = run(sim, cfg, vb)
    r['regime'] = info
    E.save(r, f'out_v19/{kind}/{tag}.json')
    for k, v in r['results'].items():
        print(k, round(v['0.001']['eprog'], 3), round(v['0.001']['det'], 3), round(v['0.001'].get('fa_test', -1), 4))
    print('seconds', round(r['seconds']))
