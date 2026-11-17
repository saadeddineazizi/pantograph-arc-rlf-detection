"""v19 matched-model check (Section VII-A): i.i.d. frames from the censored mixture with known
parameters. (a) mean time to false alarm vs e^h; (b) delay slope vs 1/KL for TLR, TLR-SR and the
baselines L1, R4; (c) KL decomposition (Prop. 1); (d) linear ramps vs the noise-free approximation (Prop. 2).
-> out_v19/matched.json"""
import json
import numpy as np
from scipy.special import ndtr, log_ndtr
from scipy.stats import norm

TH = dict(eps0=0.03, delta=0.1, c=22.0, sig=2.0, mua=20.0, siga=8.0, s=2.0, b=0.001)
rng = np.random.default_rng(19)


def sample(n, K, eps):
    eps = np.broadcast_to(np.asarray(eps, float), (n, K)) if np.ndim(eps) else np.full((n, K), eps)
    arc = rng.random((n, K)) < eps
    x = np.where(arc, rng.normal(TH['mua'], TH['siga'], (n, K)), rng.normal(0, TH['sig'], (n, K)))
    pl = TH['b'] + (1 - TH['b']) * (1 - ndtr((TH['c'] - x) / TH['s']))
    lost = rng.random((n, K)) < pl
    return np.where(lost, np.nan, x), lost


def loglik(x, lost, eps):
    c, s, sg, ma, sa, b = TH['c'], TH['s'], TH['sig'], TH['mua'], TH['siga'], TH['b']
    xx = np.nan_to_num(x)
    lq = norm.logpdf(xx, 0, sg); la = norm.logpdf(xx, ma, sa)
    lrec = np.log1p(-b) + log_ndtr((c - xx) / s) + np.logaddexp(np.log1p(-eps) + lq, np.log(eps) + la)
    Qs = ndtr(-c / np.hypot(sg, s)); Qa = ndtr((ma - c) / np.hypot(sa, s))
    PL = b + (1 - b) * ((1 - eps) * Qs + eps * Qa)
    return np.where(lost, np.log(PL), lrec), PL


def lam(x, lost):
    e0, e1 = TH['eps0'], TH['eps0'] + TH['delta']
    return loglik(x, lost, e1)[0] - loglik(x, lost, e0)[0]


def kl_parts(n=2_000_000):
    e0, e1 = TH['eps0'], TH['eps0'] + TH['delta']
    x, lost = sample(n // 1000, 1000, e1)
    L = lam(x, lost)
    p0 = float(np.asarray(loglik(np.zeros(1), np.ones(1, bool), e0)[1]).ravel()[0])
    p1 = float(np.asarray(loglik(np.zeros(1), np.ones(1, bool), e1)[1]).ravel()[0])
    loss_part = p1 * np.log(p1 / p0) + (1 - p1) * np.log((1 - p1) / (1 - p0))
    return float(L.mean()), float(loss_part), float(p0), float(p1)


def cusum(L):
    S = np.zeros(L.shape[0]); out = np.empty_like(L)
    for k in range(L.shape[1]):
        S = np.maximum(S + L[:, k], 0); out[:, k] = S
    return out


def sr(L):
    R = np.zeros(L.shape[0]); out = np.empty_like(L)
    for k in range(L.shape[1]):
        R = (1 + R) * np.exp(L[:, k]); out[:, k] = np.log(R)
    return out


def first(path, h):
    hit = path > h; a = hit.any(1)
    return np.where(a, hit.argmax(1) + 1, path.shape[1] + 1)


def main():
    KL, lossp, p0, p1 = kl_parts()
    out = dict(theta=TH, KL=KL, loss_part=lossp, loss_share=lossp / KL, p0=p0, p1=p1)
    # (a) false alarms: 4000 healthy runs of 20 000 frames
    x, lost = sample(2000, 20000, TH['eps0']); L0 = lam(x, lost)
    P0 = dict(TLR=cusum(L0), TLRsr=sr(L0))
    fa = {}
    for name, P in P0.items():
        rows = []
        for h in np.arange(2.0, 9.01, 1.0):
            t = first(P, h); frac = float((t <= P.shape[1]).mean())
            if frac >= 0.9:
                rows.append(dict(h=float(h), mean_T=float(t[t <= P.shape[1]].mean()), exp_h=float(np.exp(h)), frac=frac))
        fa[name] = rows
    out['false_alarm'] = fa
    # (b) delay after an abrupt change at frame 0
    x, lost = sample(2000, 400, TH['eps0'] + TH['delta']); L1 = lam(x, lost)
    dl = {}
    for name, P in dict(TLR=cusum(L1), TLRsr=sr(L1)).items():
        hs = np.arange(2.0, 9.01, 1.0); d = [float(first(P, h).mean()) for h in hs]
        slope = float(np.polyfit(hs, d, 1)[0]); dl[name] = dict(h=hs.tolist(), delay=d, slope=slope, slope_KL=slope * KL)
    out['delay'] = dl; out['inv_KL'] = 1 / KL
    # (d) linear ramps eps0 -> 0.35 over T frames, h = 6
    h = 6.0; e_end = 0.35
    eg = np.linspace(0.0, 0.4, 41); m = []
    for e in eg:
        xs, ls = sample(200, 1000, max(e, 1e-4)); m.append(float(lam(xs, ls).mean()))
    kap, icp = np.polyfit(eg, m, 1); estar = -icp / kap
    out['drift'] = dict(kappa=float(kap), eps_star=float(estar), max_resid=float(np.max(np.abs(np.polyval([kap, icp], eg) - m))))
    rr = []
    for T in (100, 200, 300, 600):
        eps = TH['eps0'] + (e_end - TH['eps0']) * np.minimum(np.arange(T + 400) / T, 1.0)
        xs, ls = sample(2000, T + 400, eps[None, :].repeat(2000, 0)); t = first(cusum(lam(xs, ls)), h)
        tc = T * max(estar - TH['eps0'], 0) / (e_end - TH['eps0'])
        env = tc + np.sqrt(2 * h * T / (kap * (e_end - TH['eps0'])))
        rr.append(dict(T=T, median=float(np.median(t)), approx=float(env)))
    out['ramp'] = rr
    json.dump(out, open('out_v19/matched.json', 'w'), indent=1)
    print(json.dumps(out, indent=1)[:3000])


if __name__ == '__main__':
    main()
