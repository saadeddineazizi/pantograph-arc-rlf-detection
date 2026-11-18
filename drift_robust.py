"""Robustness of the per-frame drift (v21, 18 Nov 2026). Matched parameters of the paper's matched-model check
(sigma 2, eps0 0.03, Delta 0.1, mu_a 20, sigma_a 8, s 2) with background loss b = 0.01, at margins c.
Truths: margin-coupled (eta_t = 0) and margin-free (eta_t = 1). Detectors: TLR (eta = 0) and MFL (eta = 0.99),
both with the true quiet and arc parameters. Reports the post-change drift E_Q1[lambda], the pre-change drift E_Q0[lambda],
and the efficiency E_Q1[lambda] / KL(Q1 || Q0)."""
import json
import numpy as np
from scipy.stats import norm

sig, e0, D, mua, siga, s, b = 2.0, 0.03, 0.1, 20.0, 8.0, 2.0, 0.01
e1 = e0 + D
x = np.linspace(-15, 70, 17001); dx = x[1] - x[0]
phs = norm.pdf(x, 0, sig); pha = norm.pdf(x, mua, siga)


def law(eps, c, eta):
    rec = (1 - b) * norm.cdf((c - x) / s) * ((1 - eps) * phs + eps * (1 - eta) * pha)
    Qs = norm.cdf(-c / np.hypot(sig, s)); Qa = norm.cdf((mua - c) / np.hypot(siga, s))
    lost = b + (1 - b) * ((1 - eps) * Qs + eps * (Qa + (1 - Qa) * eta))
    return rec, lost


def score(c, eta):
    r1, l1 = law(e1, c, eta); r0, l0 = law(e0, c, eta)
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.where(r0 > 0, np.log(np.maximum(r1, 1e-300) / np.maximum(r0, 1e-300)), 0.0)
    return lr, np.log(l1 / l0)


def drift(truth_eta, det_eta, c, eps):
    rq, lq = law(eps, c, truth_eta); lr, ll = score(c, det_eta)
    return float((rq * lr).sum() * dx + lq * ll)


def theta_star(truth_eta, det_eta, c):
    """positive root of E_Q0[exp(theta lambda)] = 1 (exists when E_Q0[lambda] < 0); ARL ~ exp(theta h)."""
    from scipy.optimize import brentq
    rq, lq = law(e0, c, truth_eta); lr, ll = score(c, det_eta)
    f = lambda t: float((rq * np.exp(t * lr)).sum() * dx + lq * np.exp(t * ll)) - 1.0
    if drift(truth_eta, det_eta, c, e0) >= 0: return 0.0
    hi = 1.0
    while f(hi) < 0: hi *= 2
    return brentq(f, 1e-6, hi)


def kl(truth_eta, c):
    return drift(truth_eta, truth_eta, c, e1)


out = []
for c in (5, 10, 15, 20, 25, 30, 35, 40):
    row = dict(c=c)
    for tn, te in (('coupled', 0.0), ('free', 1.0)):
        K = kl(te, c); row[f'KL_{tn}'] = K
        for dn, de in (('TLR', 0.0), ('MFL', 0.99)):
            row[f'{dn}_{tn}_d1'] = drift(te, de, c, e1); row[f'{dn}_{tn}_d0'] = drift(te, de, c, e0)
            th = theta_star(te, de, c); row[f'{dn}_{tn}_theta'] = th
            row[f'{dn}_{tn}_eff'] = max(th * row[f'{dn}_{tn}_d1'], 0.0) / K
    out.append(row)
    print(c, {k: round(v, 3) for k, v in row.items() if k.endswith('eff') or k.startswith('KL')})
# limit values of Proposition 5
Q1 = b + (1 - b) * e1; Q0 = b + (1 - b) * e0
klb = Q1 * np.log(Q1 / Q0) + (1 - Q1) * np.log((1 - Q1) / (1 - Q0))
xs = np.linspace(-30, 30, 60001); g = lambda e: np.log(1 - e + e * norm.pdf(xs, mua, siga) / norm.pdf(xs, 0, sig))
tlr_lim = (1 - Q1) * float((norm.pdf(xs, 0, sig) * (g(e1) - g(e0))).sum() * (xs[1] - xs[0]))
print('limit: KL bernoulli', klb, 'TLR drift limit', tlr_lim)
json.dump(dict(rows=out, kl_bern=float(klb), tlr_limit=tlr_lim, params=dict(sig=sig, e0=e0, D=D, mua=mua, siga=siga, s=s, b=b)),
          open('out_v19/drift_robust.json', 'w'), indent=1)
