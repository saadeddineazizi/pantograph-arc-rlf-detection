import numpy as np
import core as C


def trailing_min(P, s):
    """min over the trailing window of size s (P[:, i-s+1..i]); van Herk / Gil-Werman, vectorized over rows.
    P must already be padded on the left with s-1 columns of +inf."""
    n, L = P.shape
    nb = -(-L // s); pad = nb * s - L
    Q = np.concatenate([P, np.full((n, pad), np.inf)], axis=1).reshape(n, nb, s)
    g = np.minimum.accumulate(Q, axis=2).reshape(n, -1)[:, :L]
    h = np.minimum.accumulate(Q[:, :, ::-1], axis=2)[:, :, ::-1].reshape(n, -1)[:, :L]
    out = np.full((n, L), np.inf)
    i = np.arange(s - 1, L)
    out[:, i] = np.minimum(h[:, i - s + 1], g[:, i])
    return out


def wcusum(lam, W=300):
    """windowed CUSUM: max over start points in the last W frames of the partial sum (>= 0), from K0."""
    Cc = np.cumsum(lam[:, C.K0:], axis=1)
    n, m = Cc.shape
    Cp = np.concatenate([np.zeros((n, 1)), Cc], axis=1)            # Cp[k+1] = C_k, Cp[0] = 0
    P = np.concatenate([np.full((n, W), np.inf), Cp], axis=1)       # window size W + 1 on Cp
    mn = trailing_min(P, W + 1)[:, W:]                               # aligned with Cp
    return (Cp - mn)[:, 1:]


if __name__ == '__main__':
    rng = np.random.default_rng(0); K0 = C.K0
    lam = rng.normal(-0.05, 1, (5, K0 + 900)); W = 37
    got = wcusum(lam, W)
    Cc = np.cumsum(lam[:, K0:], axis=1); Cp = np.concatenate([np.zeros((5, 1)), Cc], axis=1)
    ref = np.array([[Cp[r, k + 1] - Cp[r, max(0, k + 1 - W):k + 2].min() for k in range(Cc.shape[1])] for r in range(5)])
    print('max abs err', np.abs(got - ref).max())
    big = wcusum(lam, 100000); print('equals cusum when W large', np.abs(big - C.cusum(lam)).max())
