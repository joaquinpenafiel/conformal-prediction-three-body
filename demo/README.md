# Short demonstration run

```bash
python demo/run_demo.py
```

Runs the full v7.7-c pipeline on **six configurations instead of sixty**, at
the intermediate control horizon (H = 2, twenty snapshots). Expect roughly
**fifteen to twenty minutes** on a modest laptop, against several days for the
full run.

Requires `numpy`, `scipy`, `scikit-learn` and `matplotlib`. The last one is
pulled in by the frozen v7.7-a module, which produces a results figure; the
demo does not draw anything, but the import must resolve.

## What it shows

1. **Environment gates.** G15 recomputes the canonical dataset hash and G16
   checks the preregistered threshold indices — the same checks the experiment
   runs before computing anything.
2. **Compute.** The frozen method, on real configurations of the dataset.
3. **Criteria and verdict.** How the binding criteria are evaluated and where
   the mechanical verdict comes from.
4. **Contrast against published results.** Compares the cells it just computed
   against `results/v7_7c/mtps_c_v7_7c_table_cells.csv`.

## Which six, and why

Two per subset, the least expensive within each, according to the measured
timings published in `results/v7_7c/tiempos_por_config_v7_7c.csv`:

| Subset | Index | Configuration |
|---|---|---|
| in distribution | 5 | `ID_jerarquico_01` |
| in distribution | 18 | `ID_masas_asimetricas_02` |
| OOD near | 39 | `OOD_NEAR_masas_asimetricas_03` |
| OOD near | 37 | `OOD_NEAR_masas_asimetricas_01` |
| OOD far | 42 | `OOD_FAR_separatriz_02` |
| OOD far | 43 | `OOD_FAR_separatriz_03` |

The `encuentro_violento` family is deliberately excluded: it concentrates
between two thirds and four fifths of the program's total compute, and a single
one of its configurations needs hours.

## How to read the contrast

**Empirical coverage should reproduce exactly.** It is a ratio of integers over
the evaluation ground truths, so it is robust to numerical noise. A difference
there is worth investigating.

**The conformal radius may differ slightly**, and that is expected across
different versions of `scipy` or `numpy`. The demo reports the relative
deviation against the integrator's own declared `rtol` of 1e-8, and reports it
either way — it does not call a deviation a failure. That distinction is not a
convenience: it is what the program learned when a regression gate aborted a
run for demanding of a continuous quantity more precision than the method
guarantees, and it is documented in `docs/actas/ENMIENDA_02_al_ACTA_02_*.docx`.

## What it is not

This does not reproduce the experiment. The scenario verdict is defined over
the full sixty configurations; on six it is not comparable. The demo exists to
show that the machinery runs end to end and can be executed by anyone in a
single session.

For the documentary chain — manifest, canonical dataset and derivation chain —
run `python verify.py` from the repository root.

## A measured result

The first execution of this demo, on a machine and a software stack unrelated
to the one that produced the published results, matched them to `0.000e+00` in
coverage and `8.60e-11` relative in the conformal radius, over 120 cells. The
root `README.md` records the details.
