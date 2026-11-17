"""v19 figures for the paper -> figures/*.pdf"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'figures'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8.5, 'axes.linewidth': 0.7, 'legend.fontsize': 7.5, 'savefig.bbox': 'tight', 'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42})
K, G = '#111111', '#777777'
COL = {'TLRmixSR': '#1b5e9b', 'TLRsr': '#c4581b', 'LQE': '#2e7d32', 'R4': '#8e24aa', 'SRon': '#555555'}
LAB = {'TLRmixSR': 'TLR-mix-SR', 'TLRsr': 'TLR-SR', 'LQE': 'LQE', 'R4': 'R4', 'SRon': 'SRon'}


def theory():
    m = json.load(open('out_v19/matched.json'))
    f, ax = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for k, mk in (('TLR', 'o'), ('TLRsr', 's')):
        d = m['delay'][k]; ax[0].plot(d['h'], d['delay'], mk + '-', color=K if k == 'TLR' else G, ms=3.5, lw=1,
                                      label={'TLR': 'CUSUM', 'TLRsr': 'Shiryaev--Roberts'}[k] + f" (slope {d['slope']:.1f})")
    h = np.array(m['delay']['TLR']['h']); ax[0].plot(h, h / m['KL'] + (m['delay']['TLR']['delay'][0] - h[0] / m['KL']), ':', color=K, lw=0.9, label=f"slope 1/KL = {1/m['KL']:.1f}")
    ax[0].set_xlabel('threshold $h$'); ax[0].set_ylabel('mean delay (frames)'); ax[0].legend(frameon=False); ax[0].set_title('(a) Abrupt change')
    T = [r['T'] for r in m['ramp']]
    ax[1].plot(T, [r['median'] for r in m['ramp']], 'o-', color=K, ms=3.5, lw=1, label='median alarm time')
    ax[1].plot(T, [r['approx'] for r in m['ramp']], 's--', color=G, ms=3.5, lw=1, label='noise-free approximation (9)')
    ax[1].set_xlabel('ramp length $T$ (frames)'); ax[1].set_ylabel('alarm time (frames)'); ax[1].legend(frameon=False); ax[1].set_title('(b) Linear ramp')
    for a in ax:
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(f'{OUT}/fig_theory.pdf'); plt.close(f)


def calib():
    from scipy.stats import beta
    xs = {0.05: [], 0.01: [], 0.001: []}
    for kind in ('held', 'phy', 'devfresh', 'conf', 'rand'):
        for fpath in glob.glob(f'out_v19/{kind}/*.json'):
            r = json.load(open(fpath))['results']
            for d, v in r.items():
                for a in xs:
                    if 'fa_test' in v[str(a)]: xs[a].append(v[str(a)]['fa_test'] / a)
    f, ax = plt.subplots(figsize=(4.2, 2.5)); n = 10000
    for i, a in enumerate((0.05, 0.01, 0.001)):
        y = np.array(xs[a]); x = i + (np.random.default_rng(1).random(y.size) - 0.5) * 0.5
        ax.plot(x, y, '.', color=K, ms=1.2, alpha=0.35)
        k = int(np.ceil((n + 1) * (1 - a))); lo, hi = beta.ppf([0.025, 0.975], n + 1 - k, k)
        lo2 = (lo - 1.96 * np.sqrt(lo * (1 - lo) / n)) / a; hi2 = (hi + 1.96 * np.sqrt(hi * (1 - hi) / n)) / a
        ax.fill_between([i - 0.3, i + 0.3], lo2, hi2, color=G, alpha=0.25, lw=0)
    ax.axhline(1, color=K, lw=0.6, ls=':')
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['$\\alpha=0.05$', '$0.01$', '$0.001$']); ax.set_ylabel('realized / nominal rate')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(f'{OUT}/fig_calib.pdf'); plt.close(f)


def onset():
    S = json.load(open('out_v19/onset_summary.json'))
    D = [0, 120, 300]
    f, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
    for d in ('TLRmixSR', 'TLRsr', 'LQE', 'R4', 'SRon'):
        ax[0].plot([30 + x for x in D], [S[str(x)][d]['prog'] for x in D], 'o-', color=COL[d], ms=3.5, lw=1.1, label=LAB[d])
        ax[1].plot([30 + x for x in D], [S[str(x)][d]['det'] for x in D], 'o-', color=COL[d], ms=3.5, lw=1.1)
    ax[0].set_ylabel('mean alarm progress'); ax[1].set_ylabel('detection rate')
    for a in ax:
        a.set_xlabel('healthy monitoring before the ramp (s)'); a.set_xticks([30, 150, 330])
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    ax[0].legend(frameon=False, ncol=1); ax[0].set_title('(a) Earliness ($\\alpha=10^{-3}$)'); ax[1].set_title('(b) Detection')
    f.tight_layout(); f.savefig(f'{OUT}/fig_onset.pdf'); plt.close(f)


def inject():
    R = json.load(open('out_v19/sncf/inject.json'))
    f, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
    st = {'TLR-ord': ('o-', '#c4581b'), 'TLR-mix-ord': ('o-', '#1b5e9b'), 'SRe-ord': ('s--', '#555555'), 'SRon-ord': ('s-', '#999999'), 'L1': ('^:', '#2e7d32')}
    for i, cfg in enumerate(('base', 'typical')):
        for d, (mk, c) in st.items():
            ax[i].plot([1, 2, 4, 8], [R[f'{cfg}_{l}'][d]['det05'] for l in (1, 2, 4, 8)], mk, color=c, ms=3.2, lw=1, label=d)
        ax[i].set_xscale('log', base=2); ax[i].set_xticks([1, 2, 4, 8]); ax[i].set_xticklabels(['1', '2', '4', '8'])
        ax[i].set_xlabel('terminal arc rate (arcs/s)'); ax[i].set_ylim(0, 1.03)
        ax[i].set_title(['(a) Long arcs (median 30 ms)', '(b) Typical arcs (median 3.6 ms, rate x 7.3)'][i])
        for s in ('top', 'right'): ax[i].spines[s].set_visible(False)
    ax[0].set_ylabel('detection probability ($\\alpha=0.05$)'); ax[0].legend(frameon=False, loc='lower right')
    f.tight_layout(); f.savefig(f'{OUT}/fig_inject.pdf'); plt.close(f)


def sncf():
    import sncf_eval19 as S
    # (a) precursor profile: share of samples with Rxqual >= 3 by time before the event start (pre-event epochs)
    bins = np.arange(-60, 1, 5); prof = []
    for a, b in zip(bins[:-1], bins[1:]):
        v = np.concatenate([e['c'][(e['t'] - 60 >= a) & (e['t'] - 60 < b)] for e in S.PRE]); prof.append((v >= 3).mean())
    base = np.mean(np.concatenate([e['c'] for e in S.HL]) >= 3)
    # (b) around handover commands
    win = np.arange(-10, 10.01, 1.0); hv = []
    for lo in win[:-1]:
        vals = []
        for x in S.TR:
            for h in x['ho']:
                m = (x['t'] - h >= lo) & (x['t'] - h < lo + 1); vals.append(x['q'][m])
        v = np.concatenate(vals); hv.append((v >= 3).mean())
    f, ax = plt.subplots(1, 2, figsize=(6.8, 2.3))
    ax[0].step(bins[:-1], prof, where='post', color=K, lw=1.1, label='pre-event epochs'); ax[0].axhline(base, color=G, ls=':', lw=0.9, label='healthy epochs')
    ax[0].set_xlabel('time to event start (s)'); ax[0].set_ylabel('share of samples with Rxqual $\\geq 3$'); ax[0].legend(frameon=False); ax[0].set_title('(a) Before poor-quality events')
    ax[1].step(win[:-1], hv, where='post', color=K, lw=1.1); ax[1].axvline(0, color=G, lw=0.7, ls=':')
    ax[1].set_xlabel('time to handover command (s)'); ax[1].set_title('(b) Around handovers')
    for a in ax:
        for s in ('top', 'right'): a.spines[s].set_visible(False)
    f.tight_layout(); f.savefig(f'{OUT}/fig_sncf.pdf'); plt.close(f)
    json.dump(dict(profile=dict(bins=bins.tolist(), share=[float(x) for x in prof], healthy=float(base)), handover=dict(bins=win.tolist(), share=[float(x) for x in hv])),
              open('out_v19/sncf/profile.json', 'w'), indent=1)
    print('profile', [round(x, 3) for x in prof], 'healthy', round(base, 3)); print('handover', [round(x, 3) for x in hv])


if __name__ == '__main__':
    theory(); calib(); onset(); inject(); sncf()
