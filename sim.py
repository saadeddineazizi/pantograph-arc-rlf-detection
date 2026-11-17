"""Simulators A and B, rebuilt (v17) from the description in the paper (Section III-C)
and in Table 6 of the supplement.  The original code (twin3.py, phy.py) was lost;
this is a re-implementation, so its numbers are not the numbers of the paper.

One frame every 100 ms.  A run is a dict of arrays of shape (n, K):
  s     logged SNR in dB (a lost frame repeats the last received value)
  lost  frame lost
  d     distance to the serving access point in m
  hand  serving-cell change at this frame (rail geometry only)
  arc   frame affected by an arc (diagnostics only, never seen by a detector)
Healthy runs have K = 600 (60 s); degrading runs K = 600 + 10 T, ramp from 60 s.
"""
import numpy as np
from scipy.signal import lfilter

FR = 0.1          # frame period, s
K_H = 600         # frames of a healthy epoch
ONSET = 600       # frame index of the ramp onset (60 s)
ONSET_S = 60.0    # ramp onset in s; changed only by onset19.py (delayed-onset robustness check)


# ---------------------------------------------------------------- helpers
def ramp_u(t, T, shape='linear'):
    """Degradation level in [0, 1] at times t (s); T=None is healthy."""
    t = np.asarray(t, float)
    if T is None:
        return np.zeros_like(t)
    if shape == 'step':
        return (t >= ONSET_S).astype(float)
    u = np.clip((t - ONSET_S) / T, 0.0, 1.0)
    if shape == 'convex':
        u = u ** 2
    return u


def ou(rng, n, K, sd, tau):
    """Ornstein-Uhlenbeck shadowing in dB on the frame grid, stationary start."""
    if sd <= 0:
        return np.zeros((n, K))
    rho = np.exp(-FR / tau)
    e = rng.standard_normal((n, K)) * sd * np.sqrt(1 - rho ** 2)
    x0 = rng.standard_normal(n) * sd
    e[:, 0] = x0
    return lfilter([1.0], [1.0, -rho], e, axis=1)


def ge_loss(rng, n, K, p, burst=5.0):
    """Gilbert-Elliott loss: stationary loss p, mean burst length `burst` frames."""
    if p <= 0:
        return np.zeros((n, K), bool)
    q = 1.0 / burst                 # bad -> good
    r = q * p / (1 - p)             # good -> bad
    bad = rng.random(n) < p
    out = np.empty((n, K), bool)
    u = rng.random((n, K))
    for k in range(K):
        bad = np.where(bad, u[:, k] >= q, u[:, k] < r)
        out[:, k] = bad
    return out


def hold(s, lost):
    """Sample-and-hold of the last received value."""
    n, K = s.shape
    idx = np.where(~lost, np.arange(K)[None, :], 0)
    idx = np.maximum.accumulate(idx, axis=1)
    return np.take_along_axis(s, idx, axis=1)


def shuttle(rng, n, K, v, lo=10.0, hi=40.0):
    span = hi - lo
    ph = rng.random(n) * 2 * span
    y = (ph[:, None] + v * FR * np.arange(K)[None, :]) % (2 * span)
    y = np.where(y > span, 2 * span - y, y)
    return lo + y


def nakagami_db(rng, shape, m=2.0):
    return 10 * np.log10(rng.gamma(m, 1.0 / m, size=shape))


# ------------------------------------------------------------ Simulator A
# Three-state arc chain at 100 Hz: contact, offline, arc (0, 6, 30 dB below nominal).
A_PA0, A_PA1 = 0.005, 0.02      # entry rate into arc, healthy and at full degradation
A_PP0, A_PP1 = 0.70, 0.95       # arc persistence
A_PO, A_POFF = 0.003, 0.70      # entry rate into offline, offline persistence


def simA(rng, n, K, cfg, T=None):
    shape = cfg.get('shape', 'linear')
    # chain, sampled at the start of each frame
    t_step = np.arange(K * 10) * 0.01
    u = ramp_u(t_step, T, shape)
    pa = A_PA0 + (A_PA1 - A_PA0) * u
    pp = A_PP0 + (A_PP1 - A_PP0) * u
    st = np.zeros(n, np.int8)
    for _ in range(300):                      # burn-in at healthy rates
        r = rng.random(n)
        st = np.where(st == 0, np.where(r < A_PA0, 2, np.where(r < A_PA0 + A_PO, 1, 0)),
                      np.where(st == 1, np.where(r < A_POFF, 1, 0), np.where(r < A_PP0, 2, 0))).astype(np.int8)
    state = np.empty((n, K), np.int8)
    for k in range(K):
        state[:, k] = st
        R = rng.random((10, n))
        for j in range(10):
            i = 10 * k + j
            r = R[j]
            st = np.where(st == 0, np.where(r < pa[i], 2, np.where(r < pa[i] + A_PO, 1, 0)),
                          np.where(st == 1, np.where(r < A_POFF, 1, 0), np.where(r < pp[i], 2, 0))).astype(np.int8)
    arc = state == 2
    off = state == 1
    # geometry and link
    d = shuttle(rng, n, K, cfg.get('speed', 10.0))
    nexp = cfg.get('n_exp', 2.0)
    sd, tau = cfg.get('sh', (0.0, 20.0))
    L = 50.0 - 10 * nexp * np.log10(d / 10.0) + ou(rng, n, K, sd, tau)
    dep_off = np.full((n, K), 6.0)
    dep_arc = np.full((n, K), 6.0 if cfg.get('nodip') else 30.0)
    if cfg.get('depthvar'):
        dep_off = dep_off + 2.0 * rng.standard_normal((n, K))
        dep_arc = dep_arc + 6.0 * rng.standard_normal((n, K))
    D = np.where(arc, dep_arc, np.where(off, dep_off, 0.0))
    snr = L + nakagami_db(rng, (n, K)) - D + rng.standard_normal((n, K))
    # loss of arc frames
    mech = cfg.get('mech', 'coupled')
    if mech == 'coupled':
        marg = L - 30.0
        p = 0.7 * np.clip((14.0 - marg) / 2.0, 0.0, 1.0)
    elif mech == 'arcnoise':
        p = np.full((n, K), 0.7)
    else:
        p = np.zeros((n, K))
    lost = arc & (rng.random((n, K)) < p)
    # background loss
    bg, bp = cfg.get('bg', ('none', 0.0))
    if bg == 'iid':
        lost |= rng.random((n, K)) < bp
    elif bg == 'ge':
        lost |= ge_loss(rng, n, K, bp)
    elif bg == 'far':
        lost |= rng.random((n, K)) < bp * (d - 10.0) / 30.0
    s = hold(snr, lost)
    return dict(s=s.astype(np.float32), lost=lost, d=d.astype(np.float32),
                hand=np.zeros((n, K), bool), arc=arc)


# ------------------------------------------------------------ Simulator B
B_TH_DET, B_W_DET = 3.0, 1.0      # preamble detection: logistic threshold and width (dB)
B_TH_DEC, B_W_DEC = 10.0, 1.0     # payload decoding
T_PRE, T_PAY = 80.0, 420.0        # microseconds


def _arc_cover(rng, n, K, T, shape):
    """Frames covered by an arc event: Poisson events, rate 0.5 -> 2 /s,
    log-normal duration with median 30 -> 150 ms, sigma_ln 0.8."""
    dur = K * FR
    lam_max = 2.0
    cnt = rng.poisson(lam_max * dur, n)
    rid = np.repeat(np.arange(n), cnt)
    a = rng.random(rid.size) * dur
    u = ramp_u(a, T, shape)
    keep = rng.random(rid.size) < (0.5 + 1.5 * u) / lam_max
    rid, a, u = rid[keep], a[keep], u[keep]
    med = 0.030 + 0.120 * u
    ln = np.exp(np.log(med) + 0.8 * rng.standard_normal(a.size))
    kf = np.ceil((a - 0.0005) / FR).astype(int)
    kl = np.floor((a + ln) / FR).astype(int)
    ok = (kl >= kf) & (kf < K)
    kf, kl, rid = np.clip(kf[ok], 0, K), np.clip(kl[ok], 0, K - 1), rid[ok]
    diff = np.zeros((n, K + 1), np.int32)
    np.add.at(diff, (rid, kf), 1)
    np.add.at(diff, (rid, kl + 1), -1)
    return np.cumsum(diff[:, :K], axis=1) > 0


def simB(rng, n, K, cfg, T=None):
    shape = cfg.get('shape', 'linear')
    geom = cfg.get('geom', 'short')
    nexp = cfg.get('n_exp', 2.0)
    p1m = cfg.get('s1_db', 70.0)
    sd, tau = cfg.get('sh', (0.0, 20.0))
    hand = np.zeros((n, K), bool)
    if geom == 'rail':
        Lc = cfg.get('cell_len', 1000.0)
        v = cfg.get('speed', 25.0)
        x = rng.random(n)[:, None] * Lc + v * FR * np.arange(K)[None, :]
        j = np.floor(x / Lc).astype(int)
        d = np.sqrt((x - (j + 0.5) * Lc) ** 2 + 20.0 ** 2)
        nc = j.max() + 2
        offs = rng.standard_normal((n, nc)) * cfg.get('cell_off_sd', 3.0)
        off = np.take_along_axis(offs, j, axis=1)
        hand[:, 1:] = j[:, 1:] != j[:, :-1]
    else:
        d = shuttle(rng, n, K, cfg.get('speed', 10.0))
        off = 0.0
    L = p1m - 10 * nexp * np.log10(d) + off + ou(rng, n, K, sd, tau)
    snr = L + nakagami_db(rng, (n, K))
    cov = _arc_cover(rng, n, K, T, shape)
    # impulses in covered frames
    ii = np.nonzero(cov)
    nci = ii[0].size
    MAXI = 8
    nimp = np.minimum(rng.poisson(1.5, nci), MAXI)
    tau_us = rng.uniform(10.0, 40.0, (nci, MAXI))
    inr = 10 ** ((44.0 + 6.0 * rng.standard_normal((nci, MAXI))) / 10)
    pos = rng.uniform(0.0, T_PRE + T_PAY, (nci, MAXI))
    on = np.arange(MAXI)[None, :] < nimp[:, None]
    e = inr * tau_us * on
    i_pre = np.zeros((n, K)); i_pay = np.zeros((n, K))
    i_pre[ii] = (e * (pos < T_PRE)).sum(1) / T_PRE
    i_pay[ii] = (e * (pos >= T_PRE)).sum(1) / T_PAY
    sinr_pre = snr - 10 * np.log10(1 + i_pre)
    sinr_pay = snr - 10 * np.log10(1 + i_pay)
    pdet = 1 / (1 + np.exp(-(sinr_pre - B_TH_DET) / B_W_DET))
    pdec = 1 / (1 + np.exp(-(sinr_pay - B_TH_DEC) / B_W_DEC))
    rec = (rng.random((n, K)) < pdet) & (rng.random((n, K)) < pdec)
    lost = ~rec
    lost |= hand & (rng.random((n, K)) < cfg.get('p_ho', 0.7))
    bg, bp = cfg.get('bg', ('none', 0.0))
    if bg == 'ge':
        lost |= ge_loss(rng, n, K, bp)
    elif bg == 'iid':
        lost |= rng.random((n, K)) < bp
    est = sinr_pre + rng.standard_normal((n, K))
    s = hold(est, lost)
    arc = cov & (i_pre + i_pay > 0)
    return dict(s=s.astype(np.float32), lost=lost, d=d.astype(np.float32), hand=hand, arc=arc)


# ------------------------------------------------------------ regimes
CELLS = {'clean': (0.0, 20.0), 'sd8': (8.0, 20.0), 'sd4': (4.0, 20.0),
         'sd4t5': (4.0, 5.0), 'sd12t60': (12.0, 60.0)}
CELL_ORDER = ['clean', 'sd8', 'sd4', 'sd4t5', 'sd12t60']

REGS = {
    # Simulator A
    'main': ('A', dict(mech='coupled')),
    'iid1': ('A', dict(mech='coupled', bg=('iid', 0.01))),
    'ge3': ('A', dict(mech='coupled', bg=('ge', 0.03))),
    'lossonly': ('A', dict(mech='arcnoise')),
    'nodip': ('A', dict(mech='arcnoise', nodip=True)),
    'dipsonly': ('A', dict(mech='none')),
    'dbg': ('A', dict(mech='coupled', bg=('far', 0.05))),
    'ge1': ('A', dict(mech='coupled', bg=('ge', 0.01))),
    'depthvar': ('A', dict(mech='coupled', depthvar=True)),
    'convex': ('A', dict(mech='coupled', shape='convex')),
    'step': ('A', dict(mech='coupled', shape='step')),
    'fast': ('A', dict(mech='coupled', speed=25.0)),
    'plexp25': ('A', dict(mech='coupled', n_exp=2.5)),
    'shadowing': ('A', dict(mech='coupled')),
    # Simulator B
    'phy_short': ('B', dict(geom='short', s1_db=70.0)),
    'phy_ge1': ('B', dict(geom='short', s1_db=70.0, bg=('ge', 0.01))),
    'phy_nexp3': ('B', dict(geom='short', s1_db=85.0, n_exp=3.0)),
    'phy_rail': ('B', dict(geom='rail', s1_db=76.0, cell_off_sd=3.0, cell_len=1000.0)),
    'phy_rail_ge1': ('B', dict(geom='rail', s1_db=76.0, cell_off_sd=3.0, cell_len=1000.0, bg=('ge', 0.01))),
}
REG_ORDER = list(REGS)

DEV = [('main', 'clean'), ('main', 'sd8'), ('iid1', 'clean'), ('iid1', 'sd8'),
       ('ge3', 'clean'), ('ge3', 'sd8')]
HELD_A = [('lossonly', 'clean'), ('lossonly', 'sd8'), ('nodip', 'clean'),
          ('dipsonly', 'clean'), ('dipsonly', 'sd8'), ('dbg', 'clean'),
          ('ge1', 'clean'), ('ge1', 'sd8'), ('depthvar', 'sd8'), ('convex', 'sd8'),
          ('step', 'sd8'), ('fast', 'sd8'), ('plexp25', 'clean'),
          ('shadowing', 'sd4t5'), ('shadowing', 'sd12t60')]
PHY = [('phy_short', 'clean'), ('phy_short', 'sd8'), ('phy_ge1', 'clean'),
       ('phy_nexp3', 'clean'), ('phy_rail', 'sd4'), ('phy_rail_ge1', 'sd4')]


def regime_cfg(reg, cell):
    sim, cfg = REGS[reg]
    cfg = dict(cfg)
    cfg['sh'] = CELLS[cell]
    return sim, cfg


def seed_held(reg, cell):
    return 51_000_000 + 10_000 * REG_ORDER.index(reg) + 1000 * CELL_ORDER.index(cell)


def seed_dev(reg, cell):
    return 31_000_000 + 1000 * (REG_ORDER.index(reg) * 10 + CELL_ORDER.index(cell))


def draw_regime(i):
    """Random regime of Simulator A (sampler of Section VII-K), seeded 7000 + i."""
    rng = np.random.default_rng(7000 + i)
    mech = rng.choice(['coupled', 'arcnoise', 'none'], p=[0.55, 0.20, 0.25])
    shape = rng.choice(['linear', 'convex', 'step'], p=[0.60, 0.25, 0.15])
    speed = float(np.exp(rng.uniform(np.log(5.0), np.log(30.0))))
    nexp = float(rng.uniform(2.0, 3.0))
    if rng.random() < 0.7:
        sh = (float(rng.uniform(2.0, 12.0)), float(np.exp(rng.uniform(np.log(5.0), np.log(60.0)))))
    else:
        sh = (0.0, 20.0)
    depthvar = bool(rng.random() < 0.3)
    b = rng.choice(['none', 'iid', 'ge', 'far'], p=[0.40, 0.20, 0.25, 0.15])
    if b == 'iid':
        bg = ('iid', float(rng.uniform(0.002, 0.02)))
    elif b == 'ge':
        bg = ('ge', float(rng.uniform(0.005, 0.03)))
    elif b == 'far':
        bg = ('far', float(rng.uniform(0.01, 0.05)))
    else:
        bg = ('none', 0.0)
    return dict(mech=str(mech), shape=str(shape), speed=speed, n_exp=nexp, sh=sh,
                depthvar=depthvar, bg=bg)


def simulate(sim, rng, n, K, cfg, T=None):
    return (simA if sim == 'A' else simB)(rng, n, K, cfg, T)
