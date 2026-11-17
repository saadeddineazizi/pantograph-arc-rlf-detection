"""v19 real GSM-R log (SNCF ETCS raw dataset, Raw_BPL): loading, events, epochs.
Rebuilt from Section VII-I of the paper (the v16 code was lost).
-> out_v19/sncf/base.pkl : traces (per file: t, q, ci, v, bg, dlrbg, ho, train, day), events, epochs"""
import glob, os, pickle
import numpy as np, pandas as pd

ROOT = 'data/Raw_BPL'
OUT = 'out_v19/sncf'
EV_MIN = 5.0        # s, minimum event length
GAP = 1.0           # s, gaps merged inside an event
EP = 60.0           # s, epoch length


def load():
    tr, bad = [], []
    for f in sorted(glob.glob(f'{ROOT}/**/*.csv', recursive=True)):
        d = pd.read_csv(f)
        d['t'] = pd.to_datetime(d['Timestamp'], errors='coerce')
        q = d[d['DL Rxqual'].notna()]
        if len(q) < 20:
            bad.append((f, 'empty')); continue
        t = (q['t'] - q['t'].iloc[0]).dt.total_seconds().to_numpy()
        if np.any(np.diff(t) < 0):
            bad.append((f, 'non-monotonic')); continue
        # carry position, speed, cell and handover state from the full row stream to the quality samples
        full = d.sort_values('t')
        aux = full[['t', 'V_TRAIN (Km/h)', 'NID_BG', 'D_LRBG', 'CI']].ffill()
        m = pd.merge_asof(q[['t']].reset_index(drop=True), aux, on='t', direction='backward')
        h = full['HO_state'].fillna(0).to_numpy()
        ho = full['t'][(h == 1) & (np.r_[0, h[:-1]] != 1)]     # handover commands
        hot = ((ho - q['t'].iloc[0]).dt.total_seconds().to_numpy()) if len(ho) else np.zeros(0)
        train = f.split('/')[-3]
        tr.append(dict(file=os.path.relpath(f, ROOT), train=train, day=str(q['t'].iloc[0])[:10],
                       t0=q['t'].iloc[0], t=t, q=q['DL Rxqual'].to_numpy().astype(int),
                       v=m['V_TRAIN (Km/h)'].to_numpy(float), bg=m['NID_BG'].to_numpy(float),
                       dl=m['D_LRBG'].to_numpy(float), ci=m['CI'].astype(str).to_numpy(), ho=hot))
    return tr, bad


def events(t, q):
    """runs of q >= 3 lasting >= EV_MIN s, gaps <= GAP merged; returns list of (start, end) times"""
    bad = q >= 3
    idx = np.flatnonzero(bad)
    if idx.size == 0:
        return []
    runs = []; s = e = idx[0]
    for i in idx[1:]:
        if t[i] - t[e] <= GAP:
            e = i
        else:
            runs.append((s, e)); s = e = i
    runs.append((s, e))
    return [(t[a], t[b]) for a, b in runs if b - a + 1 >= 11]     # 11 samples at 0.48 s (5 s)


def epochs(t, ev):
    """pre-event: [s - 60, s) for an event starting at s with no event in [s - 60, s);
    healthy: tiled 60 s windows with no event overlapping [w, min(w + 120, end of trace))"""
    pre, hl = [], []
    for k, (s, e) in enumerate(ev):
        if s - EP < t[0]:
            continue
        if any(e2 > s - EP and s2 < s for s2, e2 in ev[:k]):
            continue
        pre.append((s - EP, s, k))
    w = t[0]
    while w + EP <= t[-1]:
        post = min(w + 2 * EP, t[-1])
        if not any(s2 < post and e2 > w for s2, e2 in ev):
            hl.append((w, w + EP))
        w += EP
    return pre, hl


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    tr, bad = load()
    nev = npre = nh = 0
    for x in tr:
        x['events'] = events(x['t'], x['q'])
        x['pre'], x['healthy'] = epochs(x['t'], x['events'])
        nev += len(x['events']); npre += len(x['pre']); nh += len(x['healthy'])
    hours = sum(x['t'][-1] - x['t'][0] for x in tr) / 3600
    ns = sum(len(x['t']) for x in tr)
    allq = np.concatenate([x['q'] for x in tr]); dt = np.concatenate([np.diff(x['t']) for x in tr])
    print(f'traces {len(tr)} excluded {len(bad)} {[b[1] for b in bad]}; trains {len(set(x["train"] for x in tr))}; '
          f'hours {hours:.1f}; samples {ns}; median spacing {np.median(dt):.3f} s; median speed {np.nanmedian(np.concatenate([x["v"] for x in tr])):.0f}')
    print(f'events {nev}; pre-event epochs {npre}; healthy epochs {nh}; handovers {sum(len(x["ho"]) for x in tr)}')
    print('class pmf', np.round(np.bincount(allq, minlength=8) / allq.size, 4))
    pickle.dump(dict(traces=tr, bad=bad), open(f'{OUT}/base.pkl', 'wb'))
