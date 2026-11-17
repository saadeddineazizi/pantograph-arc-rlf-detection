"""v19 real-log results (Section VII-I/J rebuilt). Uses the frozen constants of sncf_eval19 (development trains).
usage: python3 sncf_run19.py calib | inject | stream | map     -> out_v19/sncf/*.json"""
import sys, json
import numpy as np
import sncf_eval19 as S

K = json.load(open('out_v19/sncf/frozen_ord.json'))['K']
DETS = ['TLR-ord', 'TLR-ord-SR', 'TLR-mix-ord', 'L1', 'R1', 'SRe-ord', 'SRon-ord']
L = 140


def pack(seqs):
    n = len(seqs); C = np.zeros((n, L), int); M = np.zeros((n, L), bool); T = np.full((n, L), np.inf)
    for i, s in enumerate(seqs):
        c = s['c'][:L]; C[i, :len(c)] = c; M[i, :len(c)] = True; T[i, :len(c)] = s['t'][:L]
    return C, M, T


def lam_all(C):
    a = np.where(np.arange(8) >= K['ka'], 1.0 / (8 - K['ka']), 0.0)
    tilt = K['th'] * C - np.log((S.q0 * np.exp(K['th'] * np.arange(8))).sum())
    mix = np.log(1 - K['d'] + K['d'] * a[C] / S.q0[C])
    p0 = S.tail[K['k']]; p1 = min(p0 + K['D'], 0.99)
    l1 = np.where(C >= K['k'], np.log(p1 / p0), np.log((1 - p1) / (1 - p0)))
    r1 = 0.5 * ((C - S.cmean) / S.csd - 0.25)
    hit = S.tail[C] <= 0.05
    e = np.log(1 - 0.002 + 0.002 * hit / 0.05)
    on = hit & ~np.c_[np.zeros((C.shape[0], 1), bool), hit[:, :-1]]
    eon = np.log(1 - 0.002 + 0.002 * on / 0.05)
    return {'TLR-ord': ('c', tilt), 'TLR-ord-SR': ('s', tilt), 'TLR-mix-ord': ('s', mix), 'L1': ('c', l1),
            'R1': ('c', r1), 'SRe-ord': ('s', e), 'SRon-ord': ('s', eon)}


def run_paths(C, M):
    out = {}
    for d, (kind, lam) in lam_all(C).items():
        P = np.full(C.shape, -np.inf); s = np.zeros(C.shape[0])
        for j in range(L):
            if kind == 'c':
                s = np.maximum(0.0, s + lam[:, j])
            else:
                s = np.logaddexp(0.0, s) + lam[:, j]            # log R_k = log(1 + R_{k-1}) + lam_k
            P[:, j] = np.where(M[:, j], s, -np.inf)
        out[d] = P
    return out


def thr(M, a):
    M = np.sort(M); n = M.size; k = int(np.ceil((n + 1) * (1 - a)))
    return M[k - 1] if k <= n else np.inf


HLc, HLm, HLt = pack(S.HL); PRc, PRm, PRt = pack(S.PRE)
HP, PP = run_paths(HLc, HLm), run_paths(PRc, PRm)
HMAX = {d: HP[d].max(1) for d in DETS}; PMAX = {d: PP[d].max(1) for d in DETS}
htr = np.array([e['train'] for e in S.HL]); ptr = np.array([e['train'] for e in S.PRE])
hday = np.array([e['day'] for e in S.HL])
TEST = np.array(S.TEST)


def calib():
    res = {}
    te_h = np.isin(htr, TEST); te_p = np.isin(ptr, TEST)
    for d in DETS:
        # AUC on test trains, train-level bootstrap
        a0 = S.auc(PMAX[d][te_p], HMAX[d][te_h])
        rng = np.random.default_rng(5); bs = []
        for _ in range(1000):
            tb = rng.choice(TEST, TEST.size)
            hi = np.concatenate([np.flatnonzero(htr == t) for t in tb]); pi = np.concatenate([np.flatnonzero(ptr == t) for t in tb])
            if pi.size and hi.size: bs.append(S.auc(PMAX[d][pi], HMAX[d][hi]))
        r = dict(auc=a0, auc_ci=np.percentile(bs, [2.5, 97.5]).tolist())
        for a in (0.05, 0.01):
            fa, fl, lead = [], [], []
            for s in range(300):
                rng = np.random.default_rng(77 + s); cal = set(rng.permutation(TEST)[:35])
                ch = te_h & np.isin(htr, list(cal)); eh = te_h & ~np.isin(htr, list(cal)); ep = te_p & ~np.isin(ptr, list(cal))
                h = thr(HMAX[d][ch], a); fa.append(float((HMAX[d][eh] > h).mean())); fl.append(float((PMAX[d][ep] > h).mean()))
                if s == 0:
                    for i in np.flatnonzero(ep):
                        hit = np.flatnonzero(PP[d][i] > h)
                        if hit.size: lead.append(60.0 - PRt[i, hit[0]])
            r[f'fa_{a}'] = [float(np.mean(fa))] + np.percentile(fa, [2.5, 97.5]).tolist()
            r[f'flag_{a}'] = [float(np.mean(fl))] + np.percentile(fl, [2.5, 97.5]).tolist()
            if a == 0.05 and lead: r['lead_median_iqr_split0'] = [float(np.median(lead))] + np.percentile(lead, [25, 75]).tolist()
            # later days: calibrate on days 1-2 of test trains, evaluate on days 3-8
            days = sorted(set(hday)); early = np.isin(hday, days[:2])
            h = thr(HMAX[d][te_h & early], a); r[f'later_days_fa_{a}'] = float((HMAX[d][te_h & ~early] > h).mean())
        res[d] = r
    json.dump(res, open('out_v19/sncf/calib.json', 'w'), indent=1)
    for d in DETS:
        x = res[d]; print(d, round(x['auc'], 3), [round(v, 2) for v in x['auc_ci']], 'fa', [round(v, 4) for v in x['fa_0.05']], [round(v, 4) for v in x['fa_0.01']],
              'flag', round(x['flag_0.05'][0], 3), round(x['flag_0.01'][0], 3), 'later', round(x['later_days_fa_0.05'], 3), round(x['later_days_fa_0.01'], 3), 'lead', x.get('lead_median_iqr_split0'))


# ---------------------------------------------------------------- injection (Section VII-J)
BANDS = S.BANDS


def inject(C, T, lam_max, med, sig, rate_mult, rng, ramp=30.0):
    """arcs in the last `ramp` s of each epoch; returns new class matrix"""
    C2 = C.copy()
    end = np.where(np.isfinite(T), T, -np.inf).max(1)
    on = end - ramp
    u = np.clip((T - on[:, None]) / ramp, 0, 1); u[~np.isfinite(T)] = 0
    lam = 0.25 * lam_max * rate_mult * u * 0.48
    n = rng.poisson(lam)
    tot = np.zeros(C.size)
    flat = np.repeat(np.arange(C.size), n.ravel())
    np.add.at(tot, flat, np.exp(np.log(med) + sig * rng.standard_normal(flat.size)))
    tot = tot.reshape(C.shape)
    ber0 = BANDS[C] + rng.random(C.shape) * (BANDS[C + 1] - BANDS[C])
    ber = ber0 + 100 * 0.5 * tot / 0.48
    C2 = np.clip(np.searchsorted(BANDS, ber, side='right') - 1, 0, 7)
    C2[~np.isfinite(T)] = 0
    return C2, on


def inj():
    res = {}
    te_h = np.isin(htr, TEST)
    configs = [('base', 0.030, 0.6, 1.0), ('typical', 0.0036, 0.79, 7.3)]
    out = {}
    for name, med, sig, mult in configs:
        for lam_max in (1, 2, 4, 8):
            acc = {d: {a: [] for a in (0.05, 0.01)} for d in DETS}; ch = {d: [] for d in DETS}
            for s in range(100):
                rng = np.random.default_rng(2026 + s); cal = set(rng.permutation(TEST)[:35])
                ci = te_h & np.isin(htr, list(cal)); ei = np.flatnonzero(te_h & ~np.isin(htr, list(cal)))
                C2, on = inject(HLc[ei], HLt[ei], lam_max, med, sig, mult, rng)
                P2 = run_paths(C2, HLm[ei])
                post = HLt[ei] >= on[:, None]
                for d in DETS:
                    for a in (0.05, 0.01):
                        h = thr(HMAX[d][ci], a)
                        acc[d][a].append(float(((P2[d] > h) & post).any(1).mean()))
                    h = thr(HMAX[d][ci], 0.05)
                    ch[d].append(float(((HP[d][ei] > h) & post).any(1).mean()))
            out[f'{name}_{lam_max}'] = {d: dict(det05=float(np.mean(acc[d][0.05])), det01=float(np.mean(acc[d][0.01])),
                                                chance05=float(np.mean(ch[d]))) for d in DETS}
            print(name, lam_max, {d: round(out[f'{name}_{lam_max}'][d]['det05'], 2) for d in DETS}, flush=True)
    json.dump(out, open('out_v19/sncf/inject.json', 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] in ('calib', 'inject'): {'calib': calib, 'inject': inj}[sys.argv[1]]()


# ---------------------------------------------------------------- stream-level (block) calibration
def stream():
    SD = ['TLR-ord', 'TLR-mix-ord', 'SRon-ord', 'L1']
    wins, owner = [], []
    for ti, x in enumerate(S.TR):
        t = x['t']; w = t[0]
        while w + 60 <= t[-1]:
            m = (t >= w) & (t < w + 60)
            if m.sum() >= 60:
                wins.append(dict(c=x['q'][m], t=t[m] - w)); owner.append((ti, w))
            w += 5.8
    C, M, T = pack(wins)
    P = run_paths(C, M)
    WM = {d: P[d].max(1) for d in SD}
    owner = np.array(owner)
    # healthy 2 min blocks and pre-event episodes, per trace
    blocks, episodes = [], []
    for ti, x in enumerate(S.TR):
        t = x['t']; ev = x['events']; sel = np.flatnonzero(owner[:, 0] == ti); ws = owner[sel, 1]
        w = t[0]
        while w + 120 <= t[-1]:
            post = min(w + 180, t[-1])
            if not any(s2 < post and e2 > w for s2, e2 in ev):
                inb = sel[(ws >= w) & (ws + 60 <= w + 120)]
                if inb.size: blocks.append((x['train'], x['day'], inb))
            w += 120
        for (a, s, k) in x['pre']:
            inb = sel[(ws + 60 > s - 60) & (ws + 60 <= s)]
            if inb.size: episodes.append((x['train'], x['day'], inb))
    btr = np.array([b[0] for b in blocks]); etr = np.array([e[0] for e in episodes])
    bday = np.array([b[1] for b in blocks]); eday = np.array([e[1] for e in episodes])
    trains = np.array(S.TRAINS)
    out = dict(n_windows=len(wins), n_blocks=len(blocks), n_episodes=len(episodes))
    hrs_per_block = 2 / 60
    for d in SD:
        BM = np.array([WM[d][b[2]].max() for b in blocks]); EM = np.array([WM[d][e[2]].max() for e in episodes])
        r = {}
        for a in (0.05, 0.01, 0.0025):
            fa, det = [], []
            for s in range(300):
                rng = np.random.default_rng(77 + s); cal = set(rng.permutation(trains)[:50])
                cb = np.isin(btr, list(cal))
                h = thr(BM[cb], a); fa.append(float((BM[~cb] > h).mean())); det.append(float((EM[~np.isin(etr, list(cal))] > h).mean()))
            # rolling daily recalibration: threshold from the two previous days (all trains), applied to the next day
            days = sorted(set(bday)); rd, rf = [], []
            for i in range(2, len(days)):
                cb = np.isin(bday, days[i - 2:i]); h = thr(BM[cb], a)
                rf.append(float((BM[bday == days[i]] > h).mean())); rd.append(float((EM[eday == days[i]] > h).mean()))
            r[str(a)] = dict(block_fa=float(np.mean(fa)), detect=float(np.mean(det)), det_range=np.percentile(det, [2.5, 97.5]).tolist(),
                             clusters_per_h=float(np.mean(fa)) / hrs_per_block, rolling_fa=float(np.nanmean(rf)), rolling_detect=float(np.nanmean(rd)))
        # per-window alarm rate at the per-epoch alpha = 0.05 threshold (healthy epochs of the paper)
        h = thr(HMAX[d], 0.05)
        r['windows_per_h_at_epoch_0.05'] = float((WM[d] > h).sum() / (sum(x['t'][-1] - x['t'][0] for x in S.TR) / 3600))
        out[d] = r
        print(d, {a: (round(v['block_fa'], 4), round(v['detect'], 3), round(v['rolling_detect'], 3)) for a, v in r.items() if isinstance(v, dict)},
              round(r['windows_per_h_at_epoch_0.05'], 1), flush=True)
    json.dump(out, open('out_v19/sncf/stream.json', 'w'), indent=1)
    print({k: v for k, v in out.items() if k.startswith('n_')})


if __name__ == '__main__' and sys.argv[1] == 'stream':
    stream()
