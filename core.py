"""Detector library, rebuilt (v17) from Section IV of the paper.

feats      drop x_k (eq. 2), margin level m0_k (eq. 3 before the sensitivity), onset flag,
           per-epoch scale sigma (1.4826 MAD over the reference segment, floor 0.5 dB)
tlr_fit    maximum likelihood of the censored mixture (eq. 4), six parameters
           (eps0, b, mu_a, sigma_a, s, gamma); optional fixed eta (margin-independent
           arc-loss probability, Section VIII-B)
llr        per-frame log-likelihood ratio (eq. 5), split into received and lost parts
cusum / srw  CUSUM (eq. 6) and windowed Shiryaev-Roberts (W = 300) from frame K0
sre        Shiryaev-Roberts e-detector on conformal p-values (SRe; SRon scores burst onsets)
threshold  run-level split-conformal threshold (eq. 7)
metrics    detection, early detection, expected alarm progress, pre-onset alarms, realized FA
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import ndtr, log_ndtr
from scipy.optimize import minimize

K0 = 300                 # first frame of the monitoring segment
K1 = 600                 # end of the healthy monitoring segment
REF = (100, 300)         # reference segment
FIT = (100, 600)         # frames of healthy runs used for the fit and the pools
LOG2PI = np.log(2 * np.pi)

FROZEN = dict(delta=0.003, delta_sr=0.003, onset=True, sr_window=300,
              sre_lam=0.002, sre_q=0.05)


# ------------------------------------------------------------------ features
def _trail_median(z, w):
    n, K = z.shape
    out = np.full((n, K), np.nan, np.float32)
    v = sliding_window_view(z, w, axis=1)[:, :K - w]   # window j..j+w-1 -> frame j+w
    out[:, w:] = np.median(v, axis=2)
    return out


def feats(D, chunk=1000):
    s, lost, d = D['s'], D['lost'], D['d']
    n, K = s.shape
    trend = 20 * np.log10(d / d[:, :1])
    z = (s + trend).astype(np.float32)
    m9 = np.empty((n, K), np.float32); m30 = np.empty((n, K), np.float32)
    for a in range(0, n, chunk):
        m9[a:a + chunk] = _trail_median(z[a:a + chunk], 9)
        m30[a:a + chunk] = _trail_median(z[a:a + chunk], 30)
    x = m9 - z
    x[lost] = np.nan
    m0 = m30 - trend
    F = dict(x=x, lost=lost.copy(), m0=m0, hand=D['hand'], arc=D['arc'], z=z)
    finish(F)
    return F


def finish(F):
    """Onset flag and per-epoch scale from the current x and lost."""
    lost = F['lost']
    prev = np.zeros_like(lost); prev[:, 1:] = lost[:, :-1]
    F['onset'] = lost & ~prev
    xr = F['x'][:, REF[0]:REF[1]]
    med = np.nanmedian(xr, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(xr - med), axis=1)
    sig = np.maximum(1.4826 * np.nan_to_num(mad, nan=1.0), 0.5)
    F['sig'] = sig.astype(np.float32)
    return F


def ho_mask(hand, W):
    """Frames within W frames of a serving-cell change (2W+1 frames per handover)."""
    m = hand.copy()
    for w in range(1, W + 1):
        m[:, w:] |= hand[:, :-w]
        m[:, :-w] |= hand[:, w:]
    return m


def quiet_gate(F, W):
    """Handover gating of Section IV-G: masked frames get x = 0 and are received."""
    G = dict(F)
    if W <= 0:
        return G
    m = ho_mask(F['hand'], W)
    x = F['x'].copy(); x[m] = 0.0
    lost = F['lost'] & ~m
    G.update(x=x, lost=lost, mask=m)
    finish(G)
    return G


# ------------------------------------------------------------------ model
def frame_loglik(th, x, sig, m0, lost, eta=0.0):
    eps, b, mua, siga, s, gam = th
    c = np.maximum(m0 - gam, 1.0)
    xx = np.where(lost, 0.0, x)
    lphs = -0.5 * (xx / sig) ** 2 - np.log(sig) - 0.5 * LOG2PI
    lpha = -0.5 * ((xx - mua) / siga) ** 2 - np.log(siga) - 0.5 * LOG2PI
    lmix = np.logaddexp(np.log1p(-eps) + lphs, np.log(eps) + np.log1p(-eta) + lpha)
    lrec = np.log1p(-b) + log_ndtr((c - xx) / s) + lmix
    Qs = ndtr(-c / np.sqrt(sig ** 2 + s ** 2))
    Qa = ndtr((mua - c) / np.sqrt(siga ** 2 + s ** 2))
    PL = b + (1 - b) * ((1 - eps) * Qs + eps * (Qa + (1 - Qa) * eta))
    return np.where(lost, np.log(np.maximum(PL, 1e-300)), lrec)


LO = np.array([1e-4, 1e-6, 6.0, 1.0, 0.3, -20.0])
HI = np.array([0.5, 0.3, 45.0, 20.0, 15.0, 60.0])


def _to_u(th):
    e, b, mu, sa, s, g = th
    return np.array([np.log(e / (1 - e)), np.log(b / (1 - b)), mu / 10, np.log(sa), np.log(s), g / 10])


def _from_u(u):
    return np.array([1 / (1 + np.exp(-u[0])), 1 / (1 + np.exp(-u[1])), 10 * u[2],
                     np.exp(u[3]), np.exp(u[4]), 10 * u[5]])


def fit_frames(F, mask=None, onset=True, nsub=250_000, seed=0):
    """Frames of healthy runs used by the fit: frames 100-599, non-onset lost frames dropped
    (onset variant), masked frames dropped (neutral gating)."""
    a, b = FIT
    x = F['x'][:, a:b]; lost = F['lost'][:, a:b]; m0 = F['m0'][:, a:b]
    sig = np.broadcast_to(F['sig'][:, None], x.shape)
    keep = np.ones_like(lost)
    if onset:
        keep &= ~(lost & ~F['onset'][:, a:b])
    if mask is not None:
        keep &= ~mask[:, a:b]
    keep &= np.isfinite(m0)
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    if idx.size > nsub:
        idx = rng.choice(idx, nsub, replace=False)
    return dict(x=x.ravel()[idx], lost=lost.ravel()[idx], m0=m0.ravel()[idx], sig=sig.ravel()[idx])


def tlr_fit(FF, eta=0.0):
    x, lost, m0, sig = FF['x'], FF['lost'], FF['m0'], FF['sig']
    blo, bhi = _to_u(LO), _to_u(HI)
    bounds = list(zip(np.minimum(blo, bhi), np.maximum(blo, bhi)))
    g0 = float(np.nanmedian(m0)) - 25.0
    starts = [np.array([0.03, 0.003, 20.0, 8.0, 2.0, np.clip(g0, -15, 55)]),
              np.array([0.08, 0.01, 8.0, 4.0, 4.0, np.clip(g0 - 10, -15, 55)])]
    best = None
    for th0 in starts:
        f = lambda u: -np.mean(frame_loglik(_from_u(u), x, sig, m0, lost, eta))
        r = minimize(f, _to_u(th0), method='L-BFGS-B', bounds=bounds,
                     options=dict(maxiter=400, eps=1e-6))
        if best is None or r.fun < best[1]:
            best = (_from_u(r.x), r.fun)
    th = best[0]
    return dict(theta=[float(v) for v in th], nll=float(best[1]), eta=float(eta))


def llr(F, fit, delta, mask=None, onset=True):
    """Returns (lam_rec, lam_loss): received-frame and lost-frame parts of eq. (5)."""
    eps0, b, mua, siga, s, gam = fit['theta']
    eta = fit.get('eta', 0.0)
    eps1 = min(eps0 + delta, 0.95)
    x, lost, m0 = F['x'], F['lost'], F['m0']
    sig = F['sig'][:, None]
    xx = np.where(lost, 0.0, np.nan_to_num(x, nan=0.0))
    lphs = -0.5 * (xx / sig) ** 2 - np.log(sig)
    lpha = -0.5 * ((xx - mua) / siga) ** 2 - np.log(siga)
    l1 = np.logaddexp(np.log1p(-eps1) + lphs, np.log(eps1) + np.log1p(-eta) + lpha)
    l0 = np.logaddexp(np.log1p(-eps0) + lphs, np.log(eps0) + np.log1p(-eta) + lpha)
    c = np.maximum(np.nan_to_num(m0, nan=50.0) - gam, 1.0)
    Qs = ndtr(-c / np.sqrt(sig ** 2 + s ** 2))
    Qa = ndtr((mua - c) / np.sqrt(siga ** 2 + s ** 2))
    P1 = b + (1 - b) * ((1 - eps1) * Qs + eps1 * (Qa + (1 - Qa) * eta))
    P0 = b + (1 - b) * ((1 - eps0) * Qs + eps0 * (Qa + (1 - Qa) * eta))
    lam_rec = np.where(lost, 0.0, l1 - l0)
    sc = F['onset'] if onset else lost
    lam_loss = np.where(sc, np.log(P1) - np.log(P0), 0.0)
    if mask is not None:
        lam_rec = np.where(mask, 0.0, lam_rec)
        lam_loss = np.where(mask, 0.0, lam_loss)
    return lam_rec, lam_loss


# ------------------------------------------------------------------ statistics
def cusum(lam):
    C = np.cumsum(lam[:, K0:], axis=1)
    return C - np.minimum.accumulate(np.minimum(C, 0.0), axis=1)


def srw(lam, W=300):
    """log of the windowed Shiryaev-Roberts statistic sum_{j>k-W} prod_{i=j..k} e^lam_i."""
    L = lam[:, K0:]
    n, m = L.shape
    C = np.cumsum(L, axis=1)
    out = np.empty((n, m))
    A = np.zeros(n)
    for i in range(m):
        A = np.exp(L[:, i]) * (A + 1.0)
        if i >= W:
            prev = C[:, i - W - 1] if i - W - 1 >= 0 else 0.0
            A = A - np.exp(C[:, i] - prev)
        A = np.maximum(A, 1e-12)
        out[:, i] = np.log(A)
    return out


def sr_e(e):
    """log Shiryaev-Roberts of e-values from K0: R_k = (1 + R_{k-1}) e_k."""
    E = e[:, K0:]
    n, m = E.shape
    out = np.empty((n, m)); R = np.zeros(n)
    for i in range(m):
        R = (1.0 + R) * E[:, i]
        out[:, i] = np.log(R)
    return out


def ncscore(F):
    s = np.abs(F['x']) / F['sig'][:, None]
    return np.where(F['lost'], np.inf, s)


def make_pools(F):
    a, b = FIT
    sc = ncscore(F)[:, a:b]
    lost = F['lost'][:, a:b]; on = F['onset'][:, a:b]
    pool_all = np.sort(sc.ravel())
    pool_on = np.sort(sc[~(lost & ~on)].ravel())
    return dict(all=pool_all, on=pool_on)


def pvals(sc, pool):
    N = pool.size
    ge = N - np.searchsorted(pool, sc, side='left')
    return (1.0 + ge) / (N + 1.0)


def sre_evalues(F, pools, onset=False, lam_b=0.002, q=0.05):
    sc = ncscore(F)
    p = pvals(sc, pools['on' if onset else 'all'])
    e = 1 - lam_b + lam_b * (p <= q) / q
    if onset:
        e = np.where(F['lost'] & ~F['onset'], 1.0, e)
    return e


# ------------------------------------------------------------------ calibration and metrics
ALPHAS = (0.05, 0.01, 0.001)


def threshold(M, alpha):
    M = np.sort(np.asarray(M))
    n = M.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(M[k - 1]) if k <= n else float('inf')


def healthy_max(path):
    return path[:, :K1 - K0].max(axis=1)


def alarm_frame(path, h):
    """First frame index (absolute) with statistic > h, or -1."""
    hit = path > h
    any_ = hit.any(axis=1)
    idx = np.argmax(hit, axis=1) + K0
    return np.where(any_, idx, -1)


def deg_metrics(tau, T, onset=600):
    tau = np.asarray(tau); T = np.asarray(T, float)
    K = onset + 10 * T
    pre = (tau >= 0) & (tau < onset)
    det = (tau >= onset) & (tau < K)
    prog = np.where(det, (tau - onset) / (10 * T), 1.0)
    return dict(det=float(det.mean()), det25=float((det & (prog <= 0.25)).mean()),
                det50=float((det & (prog <= 0.5)).mean()), eprog=float(prog.mean()),
                pre=float(pre.mean()))
