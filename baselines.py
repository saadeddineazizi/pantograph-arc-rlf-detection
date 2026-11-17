"""v19: baseline detectors re-implemented on the v17 code for the full rerun (option 3).
Definitions are written here once and copied into the paper; tuning grids are fixed here
before any development run (pre-registered, 17 Nov 2026).  All statistics start at K0 and
return paths of shape (n, K - K0); thresholds come from the run-level conformal rule.

R1   Gaussian mean-shift CUSUM on the standardized drop of the logged level:
     u_k = (med_ref(z) - z_k) / sigma, z = logged SNR + Friis trend (lost frames hold);
     increment d (u_k - d/2).                                  grid d in {0.1, 0.25, 0.5, 1, 2}
R4   the same on the median-of-nine residual x_k / sigma (lost frames: 0).
                                                                grid d in {0.1, 0.25, 0.5, 1, 2}
L1   Bernoulli CUSUM on loss onsets: p0 = healthy onset rate (fit runs), p1 = p0 + D.
                                                                grid D in {0.001, 0.002, 0.005, 0.01, 0.02}
OR   alarm when R4 or L1 exceeds its own threshold at alpha / 2 (frozen R4, L1 constants).
CR   bounded censored-residual CUSUM: y_k = min(x_k / sigma, B) for a received frame and,
     for a loss onset, y_k = max(min(c_k / sigma, B), z_f) with c_k = m0_k - g the margin
     at the censoring point and z_f = Phi^-1(1 - p_f); later frames of a burst: 0.
     B = 8, increment d (y - d/2), d = 0.5.                     grid g in {10, 15, 20} dB,
                                                                     p_f in {0.01, 0.02, 0.05}
LQE  standardized EWMA fusion: EWMA (rate r) of the loss indicator and of x_k / sigma,
     each standardized by its healthy mean and sd (fit runs, frames 300-599), summed.
                                                                grid r in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05}
CC   conformal CUSUM: same p-values as SRe, increment log(k) + (k - 1) log p.
                                                                grid k in {0.1, 0.2, 0.3, 0.5, 0.7}
"""
import numpy as np
from scipy.special import ndtri
import core as C

B_CR = 8.0
D_CR = 0.5
GRID = dict(R1=[0.1, 0.25, 0.5, 1.0, 2.0], R4=[0.1, 0.25, 0.5, 1.0, 2.0], L1=[0.001, 0.002, 0.005, 0.01, 0.02],
            CR=[(g, pf) for g in (10.0, 15.0, 20.0) for pf in (0.01, 0.02, 0.05)],
            LQE=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05], CC=[0.1, 0.2, 0.3, 0.5, 0.7])
# second development pass (17 Nov, before any held-out run): five optima sat on the lower edge
GRID2 = dict(R1=[0.025, 0.05], R4=[0.025, 0.05], L1=[0.0002, 0.0005],
             CR=[(g, 0.01) for g in (0.0, 5.0)], LQE=[0.0002, 0.0005], CC=[])
# third pass: the optimum keeps moving toward no forgetting; test the limit values
GRID3 = dict(R1=[0.005, 0.01], R4=[0.005, 0.01], L1=[0.0001], CR=[(-10.0, 0.01), (-5.0, 0.01)], LQE=[0.0001], CC=[])


def _level(F):
    z = F['z']
    ref = np.nanmedian(z[:, C.REF[0]:C.REF[1]], axis=1, keepdims=True)
    return (ref - z) / F['sig'][:, None]


def _resid(F):
    return np.nan_to_num(F['x'], nan=0.0) / F['sig'][:, None]


def gauss_cusum(u, d):
    return C.cusum(d * (u - d / 2))


def r1(F, d): return gauss_cusum(_level(F), d)
def r4(F, d): return gauss_cusum(_resid(F), d)


def l1(F, p0, D):
    p1 = min(p0 + D, 0.5)
    on = F['onset']
    lam = np.where(on, np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))
    return C.cusum(lam)


def cr(F, g, pf):
    y = np.minimum(_resid(F), B_CR)
    c = np.maximum(np.nan_to_num(F['m0'], nan=50.0) - g, 0.0) / F['sig'][:, None]
    yl = np.maximum(np.minimum(c, B_CR), ndtri(1 - pf))
    y = np.where(F['onset'], yl, np.where(F['lost'], 0.0, y))
    return C.cusum(D_CR * (y - D_CR / 2))


def ewma(a, r):
    out = np.empty_like(a, dtype=float); s = np.zeros(a.shape[0])
    for k in range(a.shape[1]):
        s = (1 - r) * s + r * a[:, k]; out[:, k] = s
    return out


def lqe_raw(F, r):
    lo = ewma(F['lost'][:, C.K0:].astype(float), r)
    x = ewma(_resid(F)[:, C.K0:], r)
    return lo, x


def lqe(F, r, st):
    lo, x = lqe_raw(F, r)
    return (lo - st[0]) / st[1] + (x - st[2]) / st[3]


def lqe_stats(F, r):
    lo, x = lqe_raw(F, r)
    return (lo.mean(), lo.std() + 1e-9, x.mean(), x.std() + 1e-9)


def cc(F, pools, k):
    p = C.pvals(C.ncscore(F), pools['all'])
    return C.cusum(np.log(k) + (k - 1) * np.log(p))


# ------------------------------------------------------------------ engine glue
def prep(F, consts=None):
    """consts: frozen constants; None = whole grid (development)."""
    p0 = float(F['onset'][:, C.FIT[0]:C.FIT[1]].mean())
    ctx = dict(p0=max(p0, 1e-4), pools=C.make_pools(F), consts=consts)
    rs = GRID['LQE'] if consts is None else [consts['LQE']]
    ctx['lqe'] = {r: lqe_stats(F, r) for r in rs}
    return ctx


def score(F, ctx):
    k = ctx['consts']; out = {}
    def grid(name): return GRID[name] if k is None else [k[name]]
    for d in grid('R1'): out[f'R1_{d}'] = r1(F, d)
    for d in grid('R4'): out[f'R4_{d}'] = r4(F, d)
    for D in grid('L1'): out[f'L1_{D}'] = l1(F, ctx['p0'], D)
    for g, pf in grid('CR'): out[f'CR_{g}_{pf}'] = cr(F, g, pf)
    for r in grid('LQE'): out[f'LQE_{r}'] = lqe(F, r, ctx['lqe'][r])
    for kk in grid('CC'): out[f'CC_{kk}'] = cc(F, ctx['pools'], kk)
    return out
