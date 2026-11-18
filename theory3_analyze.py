"""Summary of the 100-regime monitoring-length check with windowed baselines -> out_v19/theory3_summary.json, figures21/fig_law.pdf"""
import json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter, FixedLocator

R = [json.load(open(f)) for f in sorted(glob.glob('out_v19/theory3/6*.json'))]
MS = [r['m'] for r in R[0]['rows']]; N = np.array([r['N'] for r in R[0]['rows']], float)
STATS = ('LQE', 'LQEw', 'OR', 'ORw', 'TLRmix', 'TLRmixSR')
out = dict(n=len(R), m=MS)
rng = np.random.default_rng(7)


def boot(v, B=20000):
    v = np.asarray(v); idx = rng.integers(0, len(v), (B, len(v))); mm = v[idx].mean(1)
    return [float(v.mean()), float(np.quantile(mm, 0.025)), float(np.quantile(mm, 0.975))]


def flip(c1, c2):
    d = np.asarray(c1) - np.asarray(c2)
    for j in range(1, len(d)):
        if d[j - 1] <= 0 < d[j]:
            x0, x1 = np.log(MS[j - 1]), np.log(MS[j])
            return float(np.exp(x0 + (x1 - x0) * (-d[j - 1]) / (d[j] - d[j - 1])))
    return None if d[-1] <= 0 else float(MS[0])


for a in ('0.001', '0.01'):
    o = {}
    pr = {k: np.array([[row[f'{k}|{a}']['prog'] for row in r['rows']] for r in R]) for k in STATS}
    de = {k: np.array([[row[f'{k}|{a}']['det'] for row in r['rows']] for r in R]) for k in STATS}
    th = {k: np.array([[row['thr'][a][k] for row in r['rows']] for r in R]) for k in ('LQE', 'LQEw', 'TLRmix', 'TLRmixSR')}
    o['prog'] = {k: pr[k].mean(0).tolist() for k in STATS}; o['det'] = {k: de[k].mean(0).tolist() for k in STATS}
    o['thr'] = {k: th[k].mean(0).tolist() for k in th}
    o['mix_earlier'] = {k: [int(v) for v in (pr[k] > pr['TLRmixSR']).sum(0)] for k in STATS if k != 'TLRmixSR'}
    best_other = np.min(np.stack([pr[k] for k in STATS if k != 'TLRmixSR']), axis=0)
    o['mix_earlier_than_all'] = [int(v) for v in (best_other > pr['TLRmixSR']).sum(0)]
    o['diff_vs_LQE'] = [boot(pr['TLRmixSR'][:, j] - pr['LQE'][:, j]) for j in range(len(MS))]
    bw = np.minimum(pr['LQEw'], pr['ORw'])
    o['diff_vs_best_windowed'] = [boot(pr['TLRmixSR'][:, j] - bw[:, j]) for j in range(len(MS))]
    al = float(a)
    o['cusum_below_bound'] = float((th['TLRmix'] <= np.log(N / al)).mean())
    o['sr_below_bound'] = float((th['TLRmixSR'] <= np.log(N * 300 / al)).mean())
    o['sr_increase'] = float((th['TLRmixSR'][:, -1] - th['TLRmixSR'][:, 0]).mean())
    o['cusum_logslope'] = float(np.mean([np.polyfit(np.log(N), t, 1)[0] for t in th['TLRmix']]))
    exc = np.array([[row['thr'][a]['LQE'] - row['zN'] for row in r['rows']] for r in R])
    sl = np.array([np.polyfit(np.log(N), np.log(np.maximum(e, 1e-9)), 1)[0] for e in exc])
    o['lqe_excess_slope'] = [float(np.median(sl)), float(np.quantile(sl, 0.1)), float(np.quantile(sl, 0.9))]
    bl = np.array([[row['pred_LQE'][a] / row['thr'][a]['LQE'] for row in r['rows']] for r in R])
    o['bl_ratio'] = bl.mean(0).tolist()
    o['lqew_late'] = float((th['LQEw'][:, -1] - th['LQEw'][:, -2]).mean()); o['lqe_late'] = float((th['LQE'][:, -1] - th['LQE'][:, -2]).mean())
    fl = [flip(pr['LQE'][i], pr['TLRmixSR'][i]) for i in range(len(R))]
    v = [x for x in fl if x is not None]
    o['flip_lqe_median'] = float(np.median(v)); o['flip_lqe_q'] = [float(np.quantile(v, 0.1)), float(np.quantile(v, 0.9))]
    o['n_never_flip'] = int(sum(x is None for x in fl)); o['n_flip_at_start'] = int(sum(x == MS[0] for x in fl))
    o['or_fa'] = float(np.mean([row['thr'][a]['ORfa'] for r in R for row in r['rows']]))
    mech = np.array([r['mech'] for r in R])
    o['by_mech'] = {mm: {k: pr[k][mech == mm].mean(0).tolist() for k in STATS} | {'n': int((mech == mm).sum())} for mm in np.unique(mech)}
    out[a] = o
r_ = 1e-4; out['ewma_exponent'] = float(np.polyfit(np.log(N), np.log(np.sqrt(1 - (1 - r_) ** (2 * N))), 1)[0])
json.dump(out, open('out_v19/theory3_summary.json', 'w'), indent=1)
o = out['0.001']
print(json.dumps({k: o[k] for k in ('prog', 'mix_earlier', 'mix_earlier_than_all', 'diff_vs_LQE', 'diff_vs_best_windowed', 'flip_lqe_median', 'flip_lqe_q', 'n_never_flip', 'lqe_excess_slope', 'bl_ratio', 'sr_increase', 'cusum_logslope', 'cusum_below_bound', 'sr_below_bound', 'or_fa')}, indent=0)[:3000])

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 8.5, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
x = np.array(MS)
col = dict(LQE='#1b5e9b', LQEw='#6aa3d5', OR='#2e7d32', ORw='#8bc48a', TLRmix='#e3a36b', TLRmixSR='#c4581b')
lab = dict(LQE='LQE (no forgetting)', LQEw='LQE, 30 s memory', OR='OR', ORw='OR, windowed', TLRmix='MFL (CUSUM)', TLRmixSR='MFL-SR')
ls = dict(LQE='o-', LQEw='o--', OR='^-', ORw='^--', TLRmix='s--', TLRmixSR='s-')
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for k in ('LQE', 'LQEw', 'TLRmix', 'TLRmixSR'):
    t = np.array(o['thr'][k]); ax[0].plot(x, t / t[0], ls[k], ms=3, lw=1, color=col[k])
ax[0].plot(x, np.sqrt(x / x[0]), ':', color='0.45', lw=0.9)
ax[0].text(330, np.sqrt(330 / 30) * 1.08, r'$\sqrt{m/30}$', fontsize=7, color='0.35', ha='center')
ax[0].set_yscale('log'); ax[0].yaxis.set_major_locator(FixedLocator([1, 2, 4, 8])); ax[0].yaxis.set_major_formatter(ScalarFormatter()); ax[0].yaxis.set_minor_locator(NullLocator())
ax[0].set_ylabel('threshold / value at 30 s'); ax[0].set_title('(a) Threshold growth')
for k in STATS:
    ax[1].plot(x, o['prog'][k], ls[k], ms=3, lw=1, color=col[k], label=lab[k])
ax[1].set_ylabel('mean alarm progress'); ax[1].set_title('(b) Alarm progress, 40 s ramp')
blv = np.array([[row['pred_LQE']['0.001'] for row in r['rows']] for r in R]).mean(0)
ax[2].plot(x, o['thr']['LQE'], 'o-', ms=3, lw=1, color=col['LQE'], label='conformal')
ax[2].plot(x, blv, 'o--', ms=3, lw=1, color='0.45', label='Bachelier-Levy prediction')
ax[2].set_ylabel('LQE threshold'); ax[2].set_title('(c) LQE threshold'); ax[2].legend(frameon=False, fontsize=6.5, loc='upper left')
for a_ in ax:
    a_.set_xscale('log'); a_.set_xticks(MS); a_.set_xticklabels([str(v) for v in MS]); a_.xaxis.set_minor_locator(NullLocator())
    a_.set_xlabel('healthy period m (s)')
    for s_ in ('top', 'right'): a_.spines[s_].set_visible(False)
h, l = ax[1].get_legend_handles_labels()
f.legend(h, l, loc='lower center', ncol=6, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
f.tight_layout(rect=(0, 0.08, 1, 1)); f.savefig('/home/claude/p1r/src/paper/figures21/fig_law.pdf'); f.savefig('out_v19/fig_law3.png', dpi=200)
