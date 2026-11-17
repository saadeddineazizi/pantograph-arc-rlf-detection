"""v19 RLF warning rate, with the v18 definition added (any alarm before the first RLF, pre-onset alarms included): share of degrading runs that reach an RLF (7 consecutive lost frames after the
ramp onset) and receive an alarm after the onset and before the first RLF. Regenerates the degrading runs
of eval19 with the same seeds and uses the thresholds stored in its result files (no refit of anything
but the deterministic fit on the same healthy runs).  Also records, for Simulator B diagnostics, the loss
probability of arc-affected frames by link-margin bin.
usage: python3 rlf19.py <kind> <tag>   (kind: held, phy, conf)"""
import sys, json
import numpy as np
import core as C, engine as E, eval19 as V
from sim import regime_cfg, seed_held, draw_regime

NR = 7
BINS = [-50, 5, 10, 15, 20, 25, 30, 100]


def first_rlf(lost, onset=600):
    run = np.zeros(lost.shape[0], int); out = np.full(lost.shape[0], -1)
    for k in range(onset, lost.shape[1]):
        run = np.where(lost[:, k], run + 1, 0)
        hit = (run >= NR) & (out < 0)
        out[hit] = k
    return out


def main(kind, tag):
    R = json.load(open(f'out_v19/{kind}/{tag}.json'))
    if kind in ('held', 'phy'):
        reg, cell = tag.split('_', 1) if not tag.startswith('phy_') else (tag.rsplit('_', 1)[0], tag.rsplit('_', 1)[1])
        sim, cfg = regime_cfg(reg, cell); vb = seed_held(reg, cell)
    else:
        i = int(tag); cfg = draw_regime(i); sim = 'A'; vb = 61_000_000 + 10_000 * i
    Ffit = E.fit_features(sim, cfg, vb + 1, 3000)
    ctx = V.prep(Ffit)
    dets = [d for d in R['results'] if d != 'OR']
    acc = {d: {a: [0, 0] for a in ('0.05', '0.01', '0.001')} for d in dets}
    acc18 = {d: {a: [0, 0, 0] for a in ('0.05', '0.01', '0.001')} for d in dets}
    nrlf = 0; ndeg = 0
    hist = np.zeros((2, len(BINS) - 1))
    for ti, T in enumerate(E.TS):
        def fn(D, F):
            rl = first_rlf(D['lost'])
            out = {'_rlf': rl}
            for d, p in V.scorer(F, ctx).items():
                if d in dets:
                    for a in ('0.05', '0.01', '0.001'):
                        out[f'{d}|{a}'] = C.alarm_frame(p, R['results'][d][a]['h'])
            arc = D['arc'][:, 600:]; lost = D['lost'][:, 600:]; m = np.nan_to_num(F['m0'][:, 600:], nan=60)
            b = np.digitize(m[arc], BINS) - 1
            out['_h0'] = np.bincount(b, minlength=len(BINS) + 1)[None, :len(BINS) - 1]
            out['_h1'] = np.bincount(b, weights=lost[arc], minlength=len(BINS) + 1)[None, :len(BINS) - 1]
            return out
        r = E.batches(sim, cfg, vb + 3 + 1000 * ti, 400, 600 + 10 * T, T, fn)
        rl = r['_rlf']; reach = rl >= 0; nrlf += int(reach.sum()); ndeg += rl.size
        hist[0] += r['_h0'].sum(0); hist[1] += r['_h1'].sum(0)
        for d in dets:
            for a in ('0.05', '0.01', '0.001'):
                t = r[f'{d}|{a}']
                ok = reach & (t >= 600) & (t < rl)
                acc[d][a][0] += int(ok.sum()); acc[d][a][1] += int(reach.sum())
                ok18 = reach & (t >= 0) & (t < rl); pre = reach & (t >= 0) & (t < 600)
                acc18[d][a][0] += int(ok18.sum()); acc18[d][a][1] += int(reach.sum()); acc18[d][a][2] += int(pre.sum())
    out = dict(kind=kind, tag=tag, n_deg=ndeg, n_rlf=nrlf,
               warn={d: {a: (v[0] / v[1] if v[1] else None) for a, v in x.items()} for d, x in acc.items()},
               warn18={d: {a: (v[0] / v[1] if v[1] else None) for a, v in x.items()} for d, x in acc18.items()},
               pre_onset={d: {a: (v[2] / v[1] if v[1] else None) for a, v in x.items()} for d, x in acc18.items()},
               arc_loss_by_margin=dict(bins=BINS, n=hist[0].tolist(), lost=hist[1].tolist()))
    E.save(out, f'out_v19/rlf18/{kind}_{tag}.json')
    print(tag, nrlf, ndeg, {d: round(out['warn'][d]['0.001'], 2) for d in ('LQE', 'TLRmixSR', 'TLRsr', 'SRon') if out['warn'][d]['0.001'] is not None})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
