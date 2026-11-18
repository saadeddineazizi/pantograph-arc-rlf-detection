"""Summarise the regime-flip check -> out_v19/theory_summary.json and figure."""
import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = [json.load(open(f)) for f in sorted(glob.glob('out_v19/theory/6*.json'))]
MS = [r['m'] for r in R[0]['rows']]
import sys
a = sys.argv[1] if len(sys.argv) > 1 else '0.01'
out = dict(n_regimes=len(R), m=MS, alpha=a)
# (a) Bachelier-Levy prediction of the moving-average threshold
ratio = np.array([[row['pred_LQE'][a] / row['thr'][a]['LQE'] for row in r['rows']] for r in R])
out['bl_ratio_mean'] = ratio.mean(0).tolist()
# (b) excess of the threshold over the healthy mean path, against m (log-log slope, expected 0.5)
exc = np.array([[row['thr'][a]['LQE'] - (row['z0'] + row['c0'] * row['N']) for row in r['rows']] for r in R])
N = np.array([row['N'] for row in R[0]['rows']], float)
sl = [np.polyfit(np.log(N), np.log(np.maximum(e, 1e-9)), 1)[0] for e in exc]
out['lqe_excess_slope'] = dict(mean=float(np.mean(sl)), min=float(np.min(sl)), max=float(np.max(sl)))
# (c) windowed SR threshold against sqrt(log m)
hs = np.array([[row['thr'][a]['TLRmixSR'] for row in r['rows']] for r in R])
out['sr_thr_mean'] = hs.mean(0).tolist()
out['sr_thr_increase'] = float((hs[:, -1] - hs[:, 0]).mean())
out['lqe_thr_increase'] = float((np.array([[row['thr'][a]['LQE'] for row in r['rows']] for r in R])[:, -1] - np.array([[row['thr'][a]['LQE'] for row in r['rows']] for r in R])[:, 0]).mean())
# (d) delays: noise-free crossing of the mean path and measured mean delay / progress
def col(k, f):
    miss = 400.0 if f == 'cross' else np.nan          # mean path never crosses within the 40 s ramp -> censored at 400 frames
    return np.array([[(row[f'{k}|{a}'][f] if row[f'{k}|{a}'][f] is not None else miss) for row in r['rows']] for r in R], float)
for k in ('LQE', 'TLRmixSR'):
    out[k] = dict(cross=np.nanmean(col(k, 'cross'), 0).tolist(), delay=np.nanmean(col(k, 'delay'), 0).tolist(),
                  prog=np.nanmean(col(k, 'prog'), 0).tolist(), det=np.nanmean(col(k, 'det'), 0).tolist())
# m^{1/4} law for the moving-average crossing delay (large m)
cl = col('LQE', 'cross'); cs = col('TLRmixSR', 'cross')
out['lqe_cross_slope'] = float(np.mean([np.polyfit(np.log(N[2:]), np.log(c[2:]), 1)[0] for c in cl]))
out['sr_cross_slope'] = float(np.mean([np.polyfit(np.log(N[2:]), np.log(c[2:]), 1)[0] for c in cs]))
# (e) crossover: per regime, first m at which the windowed mixture's mean-path delay is shorter
def flip(c1, c2):
    d = c1 - c2                       # positive -> mixture earlier
    for j in range(1, len(d)):
        if d[j - 1] <= 0 < d[j]:      # linear interpolation in log m
            x0, x1 = np.log(MS[j - 1]), np.log(MS[j])
            return float(np.exp(x0 + (x1 - x0) * (-d[j - 1]) / (d[j] - d[j - 1])))
    return None if d[-1] <= 0 else float(MS[0])
fl = [flip(cl[i], cs[i]) for i in range(len(R))]
out['flip_s'] = fl
v = [x for x in fl if x is not None]
out['flip_median'] = float(np.median(v)) if v else None
out['flip_range'] = [float(min(v)), float(max(v))] if v else None
mp = np.array([[row[f'LQE|{a}']['prog'] for row in r['rows']] for r in R]); sp = np.array([[row[f'TLRmixSR|{a}']['prog'] for row in r['rows']] for r in R])
out['measured_prog_order'] = dict(lqe=mp.mean(0).tolist(), mix=sp.mean(0).tolist())
# Theorem 1 prediction from the shortest monitoring length only: m* = m0 (D_SR(m0)/D_MA(m0))^4
pred = [float(MS[0] * (cs[i, 0] / cl[i, 0]) ** 4) if cl[i, 0] < cs[i, 0] else None for i in range(len(R))]
out['flip_pred_s'] = pred
fp = [flip(mp[i], sp[i]) for i in range(len(R))]
out['flip_prog_s'] = fp
vp = [x for x in fp if x is not None]
out['flip_prog_median'] = float(np.median(vp)) if vp else None
out['n_mix_better'] = [int((mp[:, j] > sp[:, j]).sum()) for j in range(len(MS))]
json.dump(out, open(f'out_v19/theory_summary_{a}.json', 'w'), indent=1)
print(json.dumps(out, indent=1))

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 8.5, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
f, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
x = np.array(MS)
ax[0].plot(x, exc.mean(0), 'o-', color='#1b5e9b', ms=3.5, lw=1, label='moving average: threshold excess')
c = exc.mean(0)[0] / np.sqrt(N[0]); ax[0].plot(x, c * np.sqrt(N), ':', color='#1b5e9b', lw=0.9, label=r'$\propto\sqrt{m}$')
ax[0].plot(x, hs.mean(0) - hs.mean(0)[0], 's-', color='#c4581b', ms=3.5, lw=1, label='windowed mixture: threshold increase')
ax[0].set_xscale('log'); ax[0].set_xlabel('healthy monitoring before the ramp, m (s)'); ax[0].set_ylabel('threshold (statistic units)'); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title('(a) Calibrated thresholds')
ax[1].plot(x, np.nanmean(cl, 0) / 10, 'o-', color='#1b5e9b', ms=3.5, lw=1, label='moving average')
ax[1].plot(x, np.nanmean(cs, 0) / 10, 's-', color='#c4581b', ms=3.5, lw=1, label='margin-free mixture (SR)')
ax[1].set_xscale('log'); ax[1].set_xlabel('healthy monitoring before the ramp, m (s)'); ax[1].set_ylabel('mean-path alarm delay (s)'); ax[1].legend(frameon=False, fontsize=7); ax[1].set_title('(b) Alarm delay, 40 s ramp')
from matplotlib.ticker import NullLocator
for a_ in ax:
    for s_ in ('top', 'right'): a_.spines[s_].set_visible(False)
    a_.set_xticks(MS); a_.set_xticklabels([str(v) for v in MS]); a_.xaxis.set_minor_locator(NullLocator())
f.tight_layout(); f.savefig(f'out_v19/theory_flip_{a}.png', dpi=250)
