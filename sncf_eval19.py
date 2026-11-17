"""v19 real-log evaluation (rebuilt Section VII-I/J). Pre-registered in this header (17 Nov 2026, before running):
Split: 100 trains -> 30 development / 70 test trains, seed 11.  Constants chosen on development trains only,
criterion = AUC (max statistic over the epoch) between pre-event and healthy epochs.
Detectors on the ordinal sequence c_k = DL Rxqual (healthy class pmf q0 from development healthy epochs):
  TLR-ord   exponential tilt: lam = th*c - log sum q0(j) e^{th j}; CUSUM.      grid th in {0.25, 0.5, 1, 1.5, 2, 3}
  TLR-ord-SR the same, Shiryaev-Roberts form (log R).
  TLR-mix-ord margin-free mixture: post-change law (1-d) q0 + d a, a uniform on classes >= ka;
            lam = log(1 - d + d a(c)/q0(c)); SR form.                         grid ka in {3, 4, 5}, d in {0.01, 0.02, 0.05, 0.1};
            d extended to {0.001, 0.002, 0.005} after the first development pass (optimum on the edge)
  L1        Bernoulli CUSUM on c >= k, p1 = p0 + D.                           grid k in {3, 4}, D in {0.05, 0.1, 0.2}
  R1        Gaussian CUSUM on the standardized class, d = 0.5 (no tuning).
  SRe-ord / SRon-ord  e-detectors, constants of the simulations (q = 0.05, lambda = 0.002); p = P0(C >= c);
            SRon bets only on the first sample of a run of samples with p <= q.
Calibration: 300 random half splits of the 70 test trains (seed 77 + s): thresholds from calibration-train
healthy epochs (run-level conformal), realized false-alarm rate on the other half, share of its pre-event
epochs flagged.  AUC on the 70 test trains with a train-level bootstrap.
Injection: Section VII-J protocol, arcs into the last 30 s of healthy epochs of the evaluation half,
base arcs (median 30 ms, sigma_ln 0.6) and typical arcs (median 3.6 ms, sigma_ln 0.79, rate x 7.3)."""
import pickle, json, sys
import numpy as np

B = pickle.load(open('out_v19/sncf/base.pkl', 'rb'))
TR = B['traces']
TRAINS = sorted(set(x['train'] for x in TR))
rng0 = np.random.default_rng(11)
perm = rng0.permutation(TRAINS)
DEV, TEST = set(perm[:30]), list(perm[30:])
BANDS = np.array([0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6])     # BER (%) bands of Rxqual 0..7


def epoch_list(kind):
    out = []
    for x in TR:
        for w in (x['pre'] if kind == 'pre' else x['healthy']):
            a, b = w[0], w[1]
            m = (x['t'] >= a) & (x['t'] < b)
            if m.sum() < 60:          # epochs with a recording gap (6 healthy epochs) are dropped
                continue
            out.append(dict(train=x['train'], day=x['day'], c=x['q'][m], t=x['t'][m] - a, file=x['file'],
                            start=a, bg=x['bg'][m], v=x['v'][m], dl=x['dl'][m]))
    return out


PRE, HL = epoch_list('pre'), epoch_list('healthy')
q0 = np.bincount(np.concatenate([e['c'] for e in HL if e['train'] in DEV]), minlength=8).astype(float)
q0 = (q0 + 0.5) / (q0 + 0.5).sum()
tail = np.cumsum(q0[::-1])[::-1]                    # P0(C >= c)
cmean = float((np.arange(8) * q0).sum()); csd = float(np.sqrt(((np.arange(8) - cmean) ** 2 * q0).sum()))


# ---------------------------------------------------------------- statistics on one sequence
def cusum(l):
    s = 0.0; out = np.empty(len(l))
    for i, v in enumerate(l):
        s = max(0.0, s + v); out[i] = s
    return out


def srlog(l):
    r = 0.0; out = np.empty(len(l))
    for i, v in enumerate(l):
        r = (1 + r) * np.exp(v); out[i] = np.log(r)
    return out


def lam_tilt(c, th):
    return th * c - np.log((q0 * np.exp(th * np.arange(8))).sum())


def lam_mix(c, ka, d):
    a = np.where(np.arange(8) >= ka, 1.0 / (8 - ka), 0.0)
    return np.log(1 - d + d * a[c] / q0[c])


def lam_l1(c, k, D):
    p0 = tail[k]; p1 = min(p0 + D, 0.99)
    return np.where(c >= k, np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))


def e_ord(c, onset):
    p = tail[c]; hit = p <= 0.05
    if onset:
        prev = np.r_[False, hit[:-1]]; hit = hit & ~prev
    return np.log(1 - 0.002 + 0.002 * hit / 0.05)


def paths(c, K):
    c = np.asarray(c, int)
    P = {}
    P['TLR-ord'] = cusum(lam_tilt(c, K['th']))
    P['TLR-ord-SR'] = srlog(lam_tilt(c, K['th']))
    P['TLR-mix-ord'] = srlog(lam_mix(c, K['ka'], K['d']))
    P['L1'] = cusum(lam_l1(c, K['k'], K['D']))
    P['R1'] = cusum(0.5 * ((c - cmean) / csd - 0.25))
    P['SRe-ord'] = srlog(e_ord(c, False))
    P['SRon-ord'] = srlog(e_ord(c, True))
    return P


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    r = np.argsort(np.argsort(np.r_[pos, neg], kind='mergesort'), kind='mergesort') + 1.0
    # average ranks for ties
    allv = np.r_[pos, neg]; order = np.argsort(allv, kind='mergesort'); rk = np.empty(allv.size)
    sv = allv[order]; i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]: j += 1
        rk[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((rk[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def tune():
    best = {}
    dpre = [e for e in PRE if e['train'] in DEV]; dhl = [e for e in HL if e['train'] in DEV]
    def score(fn):
        return auc([fn(e['c']).max() for e in dpre], [fn(e['c']).max() for e in dhl])
    g = {th: score(lambda c: cusum(lam_tilt(np.asarray(c), th))) for th in (0.25, 0.5, 1, 1.5, 2, 3)}
    best['th'] = max(g, key=g.get); best['grid_th'] = g
    g = {(ka, d): score(lambda c: srlog(lam_mix(np.asarray(c), ka, d))) for ka in (3, 4, 5) for d in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)}
    k = max(g, key=g.get); best['ka'], best['d'] = k; best['grid_mix'] = {f'{a}_{b}': v for (a, b), v in g.items()}
    g = {(k_, D): score(lambda c: cusum(lam_l1(np.asarray(c), k_, D))) for k_ in (3, 4) for D in (0.05, 0.1, 0.2)}
    k = max(g, key=g.get); best['k'], best['D'] = k; best['grid_l1'] = {f'{a}_{b}': v for (a, b), v in g.items()}
    return best


def threshold(M, a):
    M = np.sort(M); n = M.size; k = int(np.ceil((n + 1) * (1 - a)))
    return M[k - 1] if k <= n else np.inf


if __name__ == '__main__':
    K = tune()
    json.dump(dict(K=K, n_pre=len(PRE), n_hl=len(HL), dev=sorted(DEV), q0=q0.tolist()),
              open('out_v19/sncf/frozen_ord.json', 'w'), indent=1, default=float)
    print('frozen', {k: v for k, v in K.items() if not k.startswith('grid')})
    print('grids', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in K.items() if k.startswith('grid')})
