"""v19 LSTM baseline: 32-unit one-layer LSTM on four per-frame inputs (x/sigma, lost, onset,
standardized level drop), trained on development trajectories only with frame labels
(1 from the ramp onset of a degrading run, 0 otherwise), output = EWMA (rate 0.05) of the logit.
Training set: per development regime 400 degrading runs (T = 30, 40, 50 s) and 400 healthy
runs of the same length, seeds vb + 60 (degrading) and vb + 61 (healthy), vb = seed_dev.
usage: python3 lstm19.py train   -> out_v19/lstm19.pt"""
import sys
import numpy as np
import torch
import core as C, baselines as Bs
from sim import regime_cfg, seed_dev, DEV, simulate

torch.set_num_threads(2)
RATE = 0.05


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(4, 32, batch_first=True)
        self.out = torch.nn.Linear(32, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h).squeeze(-1)


def inputs(F):
    u = Bs._level(F); r = Bs._resid(F)
    X = np.stack([np.clip(r, -8, 8), F['lost'].astype(np.float32), F['onset'].astype(np.float32),
                  np.clip(np.nan_to_num(u), -8, 8)], axis=-1).astype(np.float32)
    return X


def train(epochs=12, seed=0):
    torch.manual_seed(seed)
    Xs, Ys = [], []
    for reg, cell in DEV:
        sim, cfg = regime_cfg(reg, cell); vb = seed_dev(reg, cell)
        for ti, T in enumerate((30, 40, 50)):
            K = 600 + 10 * T
            for kind, sd in (('deg', vb + 60 + 1000 * ti), ('h', vb + 61 + 1000 * ti)):
                rng = np.random.default_rng([sd, 7])
                D = simulate(sim, rng, 134, K, cfg, T if kind == 'deg' else None)
                F = C.feats(D)
                X = inputs(F)[:, C.K0:]
                y = np.zeros(X.shape[:2], np.float32)
                if kind == 'deg':
                    y[:, 600 - C.K0:] = 1.0
                Xs.append(X); Ys.append(y)
    # pad to the longest length with masked frames
    L = max(x.shape[1] for x in Xs)
    X = np.concatenate([np.pad(x, ((0, 0), (0, L - x.shape[1]), (0, 0))) for x in Xs])
    Y = np.concatenate([np.pad(y, ((0, 0), (0, L - y.shape[1])), constant_values=-1) for y in Ys])
    X, Y = torch.tensor(X), torch.tensor(Y)
    net = Net(); opt = torch.optim.Adam(net.parameters(), 3e-3)
    lossf = torch.nn.BCEWithLogitsLoss(reduction='none')
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n); tot = 0.0
        for a in range(0, n, 64):
            i = perm[a:a + 64]
            out = net(X[i]); m = (Y[i] >= 0).float()
            l = (lossf(out, Y[i].clamp(min=0)) * m).sum() / m.sum()
            opt.zero_grad(); l.backward(); opt.step(); tot += float(l) * len(i)
        print('epoch', ep, round(tot / n, 4), flush=True)
    torch.save(net.state_dict(), 'out_v19/lstm19.pt')
    print('trained on', n, 'trajectories')


_NET = None


def score(F):
    global _NET
    if _NET is None:
        _NET = Net(); _NET.load_state_dict(torch.load('out_v19/lstm19.pt')); _NET.eval()
    X = torch.tensor(inputs(F)[:, C.K0:])
    with torch.no_grad():
        out = torch.cat([_NET(X[a:a + 500]) for a in range(0, X.shape[0], 500)]).numpy()
    return Bs.ewma(out, RATE)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
