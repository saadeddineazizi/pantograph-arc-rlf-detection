"""Summary of the extended monitoring-length check -> out_v19/theory2_summary.json and figures19/fig_law.pdf"""
import json, glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

R = [json.load(open(f)) for f in sorted(glob.glob('out_v19/theory2/6*.json'))]
MS = [r['m'] for r in R[0]['rows']]; N = np.array([r['N'] for r in R[0]['rows']], float)
K = ('LQE', 'LQEw', 'TLRmixSR', 'TLRmix')
out = dict(n=len(R), m=MS)


def flip(c1, c2):
    d = np.asarray(c1) - np.asarray(c2)
    for j in range(1, len(d)):
        if d[j - 1] <= 0 < d[j]:
            x0, x1 = np.log(MS[j - 1]), np.log(MS[j])
            return float(np.exp(x0 + (x1 - x0) * (-d[j - 1]) / (d[j] - d[j - 1])))
    return None if d[-1] <= 0 else float(MS[0])


for a in ('0.01', '0.001'):
    o = {}
    th = {k: np.array([[row['thr'][a][k] for row in r['rows']] for r in R]) for k in K}
    pr = {k: np.array([[row[f'{k}|{a}']['prog'] for row in r['rows']] for r in R]) for k in K}
    de = {k: np.array([[row[f'{k}|{a}']['det'] for row in r['rows']] for r in R]) for k in K}
    o['thr_mean'] = {k: th[k].mean(0).tolist() for k in K}
    o['prog_mean'] = {k: pr[k].mean(0).tolist() for k in K}
    o['det_mean'] = {k: de[k].mean(0).tolist() for k in K}
    # growth of the threshold with log N (slope per unit of ln N) for the stationary statistics; bound: slope <= 1
    for k in ('TLRmix', 'TLRmixSR'):
        s = [np.polyfit(np.log(N), t, 1)[0] for t in th[k]]
        o[f'{k}_logslope'] = dict(mean=float(np.mean(s)), min=float(np.min(s)), max=float(np.max(s)))
    # bounds log(N/alpha), log(N W/alpha): share of (regime, m) pairs below the bound
    al = float(a)
    o['cusum_below_bound'] = float((th['TLRmix'] <= np.log(N / al)).mean())
    o['sr_below_bound'] = float((th['TLRmixSR'] <= np.log(N * 300 / al)).mean())
    # MA excess over the healthy mean path: log-log slope (Theorem: 1/2)
    exc = np.array([[row['thr'][a]['LQE'] - row['zN'] for row in r['rows']] for r in R])
    sl = [np.polyfit(np.log(N), np.log(np.maximum(e, 1e-9)), 1)[0] for e in exc]
    o['lqe_excess_slope'] = dict(mean=float(np.mean(sl)), min=float(np.min(sl)), max=float(np.max(sl)))
    o['bl_ratio'] = np.array([[row['pred_LQE'][a] / row['thr'][a]['LQE'] for row in r['rows']] for r in R]).mean(0).tolist()
    # windowed MA: threshold growth after the start-up transient (m >= 240 s) vs the no-forgetting MA
    o['lqew_thr_growth_late'] = float((th['LQEw'][:, -1] - th['LQEw'][:, -2]).mean())
    o['lqe_thr_growth_late'] = float((th['LQE'][:, -1] - th['LQE'][:, -2]).mean())
    o['lqew_prog_growth_late'] = float((pr['LQEw'][:, -1] - pr['LQEw'][:, -2]).mean())
    o['lqe_prog_growth_late'] = float((pr['LQE'][:, -1] - pr['LQE'][:, -2]).mean())
    # regimes in which the mixture (SR) is earlier than each comparator, per m
    o['mix_earlier'] = {k: [int(v) for v in (pr[k] > pr['TLRmixSR']).sum(0)] for k in ('LQE', 'LQEw', 'TLRmix')}
    o['flip_lqe'] = [flip(pr['LQE'][i], pr['TLRmixSR'][i]) for i in range(len(R))]
    v = [x for x in o['flip_lqe'] if x is not None]; o['flip_lqe_median'] = float(np.median(v))
    o['flip_lqe_range'] = [float(min(v)), float(max(v))]
    # semi-analytic finite-ramp prediction of the MA mean-path delay:
    # ramp response r(s) measured once at m0 = 30 s, threshold predicted from healthy data (Bachelier-Levy)
    err, pc, mc = [], [], []
    for r in R:
        r0 = r['rows'][0]; rise = np.array(r0['mean_path']['LQE']) - r0['zN']
        pcr, mcr = [], []
        for row in r['rows']:
            idx = np.nonzero(row['zN'] + rise > row['pred_LQE'][a])[0]; p_ = float(idx[0]) if idx.size else 400.0
            mp = np.array(row['mean_path']['LQE']); j = np.nonzero(mp > row['thr'][a]['LQE'])[0]; m_ = float(j[0]) if j.size else 400.0
            pcr.append(p_); mcr.append(m_)
        pc.append(pcr); mc.append(mcr)
    pc, mc = np.array(pc) / 10, np.array(mc) / 10
    o['semi_pred_delay_s'] = pc.mean(0).tolist(); o['meas_delay_s'] = mc.mean(0).tolist()
    o['semi_mae_s'] = float(np.abs(pc - mc)[:, 1:].mean())
    # predicted reversal: semi-analytic MA delay vs measured SR mean-path delay
    sc = np.array([[ (lambda j: float(j[0]) if j.size else 400.0)(np.nonzero(np.array(row['mean_path']['TLRmixSR']) > row['thr'][a]['TLRmixSR'])[0]) for row in r['rows']] for r in R]) / 10
    o['sr_delay_s'] = sc.mean(0).tolist()
    fp = [flip(pc[i], sc[i]) for i in range(len(R))]; fm = [flip(mc[i], sc[i]) for i in range(len(R))]
    o['flip_pred'] = fp; o['flip_meas_meanpath'] = fm
    vv = [(x, y) for x, y in zip(fp, fm) if x is not None and y is not None]
    o['flip_pred_median'] = float(np.median([x for x, _ in vv])); o['flip_meas_median'] = float(np.median([y for _, y in vv]))
    o['flip_ratio_median'] = float(np.median([x / y for x, y in vv]))
    out[a] = o

r_ = 1e-4
out['ewma_exponent'] = float(np.polyfit(np.log(N), np.log(np.sqrt(1 - (1 - r_) ** (2 * N))), 1)[0])
json.dump(out, open('out_v19/theory2_summary.json', 'w'), indent=1)
print('ewma exponent', out['ewma_exponent'])

# figure
a = '0.001'; o = out[a]; x = np.array(MS)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'font.size': 8.5, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
f, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
col = dict(LQE='#1b5e9b', LQEw='#6aa3d5', TLRmixSR='#c4581b', TLRmix='#e3a36b')
lab = dict(LQE='MA, no forgetting (LQE)', LQEw='MA, memory 30 s', TLRmixSR='mixture, windowed SR', TLRmix='mixture, CUSUM')
mk = dict(LQE='o', LQEw='o', TLRmixSR='s', TLRmix='s')
for k in K:
    ax[0].plot(x, o['thr_mean'][k], marker=mk[k], ms=3, lw=1, color=col[k], label=lab[k])
from matplotlib.ticker import ScalarFormatter, FixedLocator
ax[0].set_yscale('log'); ax[0].yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50])); ax[0].yaxis.set_major_formatter(ScalarFormatter()); ax[0].yaxis.set_minor_locator(NullLocator()); ax[0].set_ylabel('threshold'); ax[0].set_title('(a) Calibrated thresholds')
for k in K:
    ax[1].plot(x, o['prog_mean'][k], marker=mk[k], ms=3, lw=1, color=col[k])
ax[1].set_ylabel('mean alarm progress'); ax[1].set_title('(b) Alarm progress, 40 s ramp')
bl = np.array([[row['pred_LQE'][a] for row in r['rows']] for r in R]).mean(0)
ax[2].plot(x, o['thr_mean']['LQE'], 'o-', ms=3, lw=1, color=col['LQE'], label='measured (conformal)')
ax[2].plot(x, bl, 'o--', ms=3, lw=1, color='0.45', label='Bachelier-Levy, healthy increments')
ax[2].set_ylabel('MA threshold'); ax[2].set_title('(c) MA threshold: law vs measured'); ax[2].legend(frameon=False, fontsize=6.5, loc='upper left')
for a_ in ax:
    a_.set_xscale('log'); a_.set_xticks(MS); a_.set_xticklabels([str(v) for v in MS]); a_.xaxis.set_minor_locator(NullLocator())
    a_.set_xlabel('healthy period m (s)')
    for s_ in ('top', 'right'): a_.spines[s_].set_visible(False)
h, l = ax[0].get_legend_handles_labels()
f.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.02))
f.tight_layout(rect=(0, 0.08, 1, 1)); f.savefig('/home/claude/p1r/src/paper/figures19/fig_law.pdf'); f.savefig('out_v19/fig_law.png', dpi=200)
