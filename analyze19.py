"""v19 analysis: tables and the pre-registered tests (out_v19/HYPOTHESES.txt). -> out_v19/summary19.json"""
import json, glob, os
import numpy as np
from scipy.stats import beta

DET = ['R1', 'R4', 'L1', 'OR', 'CR', 'LQE', 'CC', 'SRe', 'SRon', 'LSTM', 'TLR', 'TLRsr', 'TLRloss', 'TLRdips', 'TLRmix', 'TLRmixSR']


def load(kind):
    out = {}
    for f in sorted(glob.glob(f'out_v19/{kind}/*.json')):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out


def table(runs, a='0.001'):
    t = {}
    for d in DET:
        p = [r['results'][d][a]['eprog'] for r in runs.values() if d in r['results']]
        q = [r['results'][d][a]['det'] for r in runs.values() if d in r['results']]
        q50 = [r['results'][d][a]['det50'] for r in runs.values() if d in r['results']]
        if p: t[d] = dict(prog=float(np.mean(p)), det=float(np.mean(q)), det50=float(np.mean(q50)), n=len(p))
    return t


def boot_less(x, y, B=10000, seed=19):
    """one-sided p for mean(x) < mean(y), paired over regimes"""
    d = np.asarray(x) - np.asarray(y); rng = np.random.default_rng(seed)
    bs = rng.choice(d, (B, d.size)).mean(1)
    ci = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float((bs >= 0).mean()), [float(ci[0]), float(ci[1])], int((d < 0).sum()), int(d.size)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(order):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj.tolist()


S = {}
for kind in ('held', 'phy', 'devfresh', 'conf', 'rand'):
    R = load(kind)
    S[kind] = dict(n=len(R), table={a: table(R, a) for a in ('0.05', '0.01', '0.001')},
                   per=dict((k, {d: r['results'][d]['0.001']['eprog'] for d in DET if d in r['results']}) for k, r in R.items()))
    if kind in ('conf', 'rand'):
        S[kind]['mech'] = {k: r['regime']['mech'] for k, r in R.items()}
        for m in ('coupled', 'arcnoise', 'none'):
            sub = {k: r for k, r in R.items() if r['regime']['mech'] == m}
            S[kind][f'table_{m}'] = table(sub) if sub else {}

# calibration audit: realized/nominal on test runs, with the 95% band of the conformal rule (n = 1e4 cal, 1e4 test)
cal = []
for kind in ('held', 'phy', 'devfresh', 'conf', 'rand'):
    for k, r in load(kind).items():
        for d in DET:
            for a in ('0.05', '0.01', '0.001'):
                m = r['results'].get(d, {}).get(a, {})
                if 'fa_test' in m: cal.append((kind, k, d, float(a), m['fa_test']))
if cal:
    n = 10000
    rows = []
    for kind, k, d, a, fa in cal:
        kk = int(np.ceil((n + 1) * (1 - a))); lo, hi = beta.ppf([0.025, 0.975], n + 1 - kk, kk)
        lo2, hi2 = lo - 1.96 * np.sqrt(lo * (1 - lo) / n), hi + 1.96 * np.sqrt(hi * (1 - hi) / n)
        rows.append((a, fa / a, lo2 <= fa <= hi2))
    S['calibration'] = {str(a): dict(mean_ratio=float(np.mean([r[1] for r in rows if r[0] == a])),
                                     inside=float(np.mean([r[2] for r in rows if r[0] == a])),
                                     n=sum(1 for r in rows if r[0] == a)) for a in (0.05, 0.01, 0.001)}

# pre-registered tests
C = S['conf']
if C['n'] == 100:
    per, mech = C['per'], C['mech']
    def col(d, keys): return [per[k][d] for k in keys]
    allk = sorted(per); arc = [k for k in allk if mech[k] == 'arcnoise']; cpl = [k for k in allk if mech[k] == 'coupled']
    H = dict(H1=boot_less(col('TLRmixSR', allk), col('TLRsr', allk)),
             H2=boot_less(col('TLRmixSR', arc), col('SRon', arc)),
             H3=boot_less(col('TLRmixSR', cpl), col('TLRsr', cpl)),
             H4=boot_less(col('TLRmixSR', allk), col('SRon', allk)),
             H5=boot_less(col('TLRmixSR', allk), col('LQE', allk)))
    adj = holm([H[h][1] for h in sorted(H)])
    S['tests'] = {h: dict(diff=H[h][0], p=H[h][1], p_holm=adj[i], ci=H[h][2], wins=H[h][3], n=H[h][4])
                  for i, h in enumerate(sorted(H))}

json.dump(S, open('out_v19/summary19.json', 'w'), indent=1)
for kind in ('devfresh', 'held', 'phy', 'conf', 'rand'):
    t = S[kind]['table']['0.001']
    if t:
        print(f'== {kind} (n={S[kind]["n"]}) prog/det at 1e-3')
        print('  ' + '  '.join(f'{d} {t[d]["prog"]:.3f}/{t[d]["det"]:.2f}' for d in DET if d in t))
if 'calibration' in S: print('calibration', S['calibration'])
if 'tests' in S:
    for h, v in S['tests'].items(): print(h, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
