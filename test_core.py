"""Unit checks of the statistics against direct definitions."""
import numpy as np
import core as C
rng = np.random.default_rng(3)
lam = np.zeros((5, 1000)); lam[:, C.K0:] = rng.normal(0.01, 0.3, (5, 700))
# CUSUM
S = np.zeros(5); ref = []
for k in range(C.K0, 1000):
    S = np.maximum(0, S + lam[:, k]); ref.append(S.copy())
assert np.allclose(C.cusum(lam), np.array(ref).T)
# windowed SR, W = 300: log sum_{j=k-W+1..k} exp(sum_{i=j..k} lam_i), j >= K0
L = lam[:, C.K0:]; out = C.srw(lam)
for k in (0, 10, 299, 300, 301, 650):
    j0 = max(0, k - 300 + 1)
    v = np.log(sum(np.exp(L[:, j:k + 1].sum(1)) for j in range(j0, k + 1)))
    assert np.allclose(out[:, k], v, rtol=1e-8, atol=1e-8), k
# threshold rule
M = rng.random(9999); h = C.threshold(M, 0.001)
assert h == np.sort(M)[int(np.ceil(10000 * 0.999)) - 1]
# eta = 0 reproduces the TLR log-likelihood ratio
F = dict(x=rng.normal(0, 3, (3, 50)).astype(np.float32), lost=rng.random((3, 50)) < 0.05,
         m0=np.full((3, 50), 30.0, np.float32), sig=np.full(3, 3.0, np.float32))
prev = np.zeros_like(F['lost']); prev[:, 1:] = F['lost'][:, :-1]; F['onset'] = F['lost'] & ~prev
th = dict(theta=[0.03, 0.01, 20.0, 8.0, 2.0, 10.0])
a = C.llr(F, dict(th, eta=0.0), 0.003); b = C.llr(F, dict(th, eta=1e-12), 0.003)
assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
print('core checks passed')
