"""Experiment 2, step 1: fix eta of TLR-mix / TLR-mix-SR on the six development regimes.
Criterion of the frozen protocol: mean over the six regimes of the expected alarm progress
at alpha = 1e-3.  Development seeds: 31 000 000 + 1000 (10 r + c); fit vb+1 (3000 runs),
calibration vb+10 (10 000 runs), degrading vb+3 (1200 runs).  No held-out regime is
touched by this script.
usage: python3 mix_dev.py <reg> <cell>"""
import sys
import numpy as np
import core as C
import engine as E
from sim import regime_cfg, seed_dev

ETA = (0.9, 0.95, 0.98, 0.99)


def prep(F):
    FF = C.fit_frames(F)
    fits = {eta: C.tlr_fit(FF, eta=eta) for eta in ETA}
    return dict(fits=fits, summary={str(e): f['theta'] for e, f in fits.items()})


def scorer(F, ctx):
    out = {}
    for eta, fit in ctx['fits'].items():
        lr, ll = C.llr(F, fit, C.FROZEN['delta'])
        out[f'TLRmix{eta}'] = C.cusum(lr + ll)
        out[f'TLRmixSR{eta}'] = C.srw(lr + ll)
    return out


if __name__ == '__main__':
    reg, cell = sys.argv[1], sys.argv[2]
    sim, cfg = regime_cfg(reg, cell)
    vb = seed_dev(reg, cell)
    r = E.evaluate(sim, cfg, vb, prep, scorer, sizes=(3000, 10000, 0, 1200), log=print)
    r['regime'] = [reg, cell]
    E.save(r, f'out_v19/dev/mixdev2_{reg}_{cell}.json')
    for k, v in r['results'].items():
        print(k, round(v['0.001']['eprog'], 4), round(v['0.001']['det'], 3))
