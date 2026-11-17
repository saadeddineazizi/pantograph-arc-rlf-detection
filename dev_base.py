"""v19 step 1: development grid of the baseline detectors (baselines.GRID) on the six
development regimes, development seeds as in mix_dev.py.  usage: python3 dev_base.py reg cell [small]"""
import sys
import core as C, engine as E, baselines as Bs
from sim import regime_cfg, seed_dev
reg, cell = sys.argv[1], sys.argv[2]
small = len(sys.argv) > 3 and sys.argv[3] == 'small'
sim, cfg = regime_cfg(reg, cell)
vb = seed_dev(reg, cell)
sizes = (600, 1000, 0, 300) if small else (3000, 10000, 0, 1200)
if len(sys.argv) > 3 and sys.argv[3] in ('g2', 'g3'):
    Bs.GRID = Bs.GRID2 if sys.argv[3] == 'g2' else Bs.GRID3; small = False
r = E.evaluate(sim, cfg, vb, lambda F: Bs.prep(F), Bs.score, sizes=sizes, log=print)
r['regime'] = [reg, cell]
if not small: E.save(r, f'out_v19/dev/base{sys.argv[3][1] if len(sys.argv) > 3 else ""}_{reg}_{cell}.json')
for k, v in r['results'].items():
    print(k, round(v['0.001']['eprog'], 4), round(v['0.001']['det'], 3), round(v['0.05']['eprog'], 3))
