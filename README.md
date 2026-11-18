# Calibrated detection of pantograph-arcing radio link degradation

Code and result files for the article "Margin-Free Detection of Pantograph-Arc Radio Link Degradation with Calibrated False Alarms: Simulation and Arc Injection into a Real GSM-R Log" (S. E. Azizi, Z. Yu, S. Azizi). In the code, the margin-free likelihood (MFL in the article) is called TLRmix, and MFL-SR is TLRmixSR.

Every table and figure in the article is computed from the JSON files in `out_v19/`, which these scripts write. Run all commands from the repository root.

The per-run result folders (`conf`, `dev`, `devfresh`, `eta`, `held`, `onset`, `phy`, `rand`, `rlf`, `rlf18`) are packed in `out_v19_runs.zip` to keep the repository small. Unpack it into `out_v19/` before running the analysis scripts:

```
unzip out_v19_runs.zip -d out_v19
```

## Requirements

Python 3.10 or later, and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

## Real GSM-R log

The real-log analysis uses the public SNCF ETCS raw dataset (Raw_BPL, RailNet). Download it and place the `Raw_BPL` folder at `data/Raw_BPL`. Then build the event and epoch file:

```
python3 sncf19.py          # -> out_v19/sncf/base.pkl
```

## Pipeline

| Step | Command | Output |
|---|---|---|
| Unit checks | `python3 test_core.py` | console |
| Development passes (baselines, mixture eta grid) | `bash q19_dev.sh`, `bash q19_dev2.sh` | `out_v19/dev/` |
| LSTM baseline | `python3 lstm19.py train` | `out_v19/lstm19.pt` |
| Freeze constants | `python3 freeze19.py` | `out_v19/frozen.json` |
| Pre-registered hypotheses | written before evaluation | `out_v19/HYPOTHESES.txt` |
| Main evaluation (held-out, Simulator B, fresh development, 100 confirmatory and 80 exploratory random regimes) | `bash worker19.sh 0` and `bash worker19.sh 1` (jobs in `jobs19.txt`) | `out_v19/{held,phy,devfresh,conf,rand}/` |
| Tables, calibration audit, tests H1 to H5 | `python3 analyze19.py` | `out_v19/summary19.json` |
| Delayed-onset check | `bash worker_onset.sh 0/1`, then `python3 onset_analyze.py` | `out_v19/onset/`, `out_v19/onset_summary.json` |
| RLF warning rate and arc loss by margin | `bash worker_rlf.sh 0/1` | `out_v19/rlf/` |
| RLF warning rate under the earlier definition (any alarm before the first RLF) | `bash rlf18_worker.sh 0/1` | `out_v19/rlf18/` |
| Matched-model check | `python3 matched19.py` | `out_v19/matched.json` |
| Sensitivity to eta | `bash eta_worker.sh 0/1`, then `python3 eta_analyze.py` | `out_v19/eta/`, `out_v19/eta_summary.json` |
| Real log: frozen ordinal detectors, AUC | `python3 sncf_eval19.py` | `out_v19/sncf/frozen_ord.json` |
| Real log: calibration, injection, stream | `python3 sncf_run19.py` | `out_v19/sncf/{calib,inject,stream}.json` |
| Real log: place map | `python3 sncf_map19.py` | `out_v19/sncf/map.json` |
| Figures | `python3 figs19.py` | `figures/` |

The worker scripts take every second job of their job list and skip jobs whose output already exists, so two workers can run in parallel and an interrupted run can be resumed.

## Files

- `sim.py`: Simulator A and Simulator B, regime definitions and seeds.
- `engine.py`: batched simulation and feature extraction.
- `core.py`: censored (Tobit) likelihood, margin-free mixture (TLR-mix), CUSUM and Shiryaev-Roberts statistics, e-detectors, conformal thresholds and metrics.
- `baselines.py`, `lstm19.py`: R1, R4, L1, OR, CR, LQE, CC and LSTM baselines.
- `eval19.py`: evaluation of all frozen detectors on one regime.
- `sncf19.py`, `sncf_eval19.py`, `sncf_run19.py`, `sncf_map19.py`: real-log pipeline.

## Seeds

Held-out regimes use base 51 000 000 + 10 000 r + 1000 c, random regimes 61 000 000 + 10 000 i, the real-log train split seed 11, calibration splits 77 + s, and injections 2026 + s.

## Analysis checks (added for the revised article)

| Step | Command | Output |
|---|---|---|
| Efficiency of the Tobit and margin-free scores under both loss mechanisms (Proposition 5, Table 2) | `python3 drift_robust.py` | `out_v19/drift_robust.json` |
| Healthy likelihood against eta | `python3 eta_ident.py` | `out_v19/eta_ident.json` |
| Healthy-period check with windowed baselines, all 100 pre-registered regimes (Theorem 1, Table 5, Fig. 1) | `bash theory_worker3.sh 0` and `bash theory_worker3.sh 1`, then `python3 theory3_analyze.py` | `out_v19/theory3/`, `out_v19/theory3_summary.json` |
| Earlier 10-regime checks of Theorem 1 | `theory_flip.py`, `theory_flip2.py` with `theory_analyze.py`, `theory2_analyze.py` | `out_v19/theory*/` |

`theory_worker3.sh` reads `theory3_jobs.txt` (regimes 600 to 699).
| Window length of MFL-SR (W = 150, 300, 600 frames), 100 pre-registered regimes (Table 6) | `bash theory_workerW.sh 0` and `bash theory_workerW.sh 1`, then `python3 theoryW_analyze.py` | `out_v19/theoryW/`, `out_v19/theoryW_summary.json` |
| Computing cost per frame of each detector | `python3 timing.py` | `out_v19/timing.json` |
