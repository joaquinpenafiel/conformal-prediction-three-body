# Conformal prediction for chaotic three-body systems

A preregistered experimental program that measures **where a conformal
prediction instrument stops being useful**, and separates that boundary from
the point where it stops being valid.

Coverage is calibrated **per physical configuration**: simulated replicas of
the same initial condition act as a local calibration set, so the guarantee is
conditional on the configuration rather than marginal over a mixed population.
Three closed stages characterise the operating domain along three axes —
ensemble size, calibration-set size and temporal horizon.

Every stage was frozen before execution, guarded by gates that abort the run,
and closed by a **mechanical verdict** computed over criteria fixed in advance.
The full audit chain is anchored by SHA-256 and verifiable in ten seconds.

```
python verify.py
```

[![Verify package](https://github.com/joaquinpenafiel/conformal-prediction-three-body/actions/workflows/verify.yml/badge.svg)](https://github.com/joaquinpenafiel/conformal-prediction-three-body/actions/workflows/verify.yml)

---

## What was measured

| Stage | Question | Verdict |
|---|---|---|
| v7.7-a | Is calibration robust to ensemble size? | 4/4 criteria, 5/5 levels at N ∈ {50, 100, 200} |
| v7.7-b | Can the full five-level grid be enabled? | S11.1 — grid enabled, M = 40 inherited |
| v7.7-c | How deep in time does the instrument stay useful? | S12.1 — boundary characterised |

**The central result of the cycle.** At four times the reference horizon the
regions are still *valid*: empirical coverage holds within tolerance at all
five nominal levels, including the deepest block of snapshots, where the 0.95
level measures 95.47%. What breaks earlier is the geometry — the fraction of
ensemble variance captured by a fixed two-component basis crosses the 0.95
threshold at snapshot 26 and stays below it from snapshot 30 onward.

The characterised limit is therefore one of **geometric informativeness, not
of conformal validity**. The conformal radius grows 567× in absolute terms
across the horizon, but only 1.9× relative to the ensemble's own spread: the
region follows the system's dispersion rather than inflating on its own.

**A preregistered prediction failed, and is reported as failed.** The program
registered, before running, an extrapolation of how effective dimension would
grow. At forty snapshots it predicted 7.35 and the run measured 3.20. The
growth decelerates instead of accelerating — informative about the dynamics,
and recorded precisely because it could have been wrong.

---

## Why this repository might be worth your time

Not for the three-body problem, which is thoroughly studied. For the
experimental machinery, and for the fact that **it actually fired**:

- A gate **aborted a run**. In v7.7-c, the regression gate detected that a run
  failed to reproduce the reference horizon within the inherited tolerance.
  The investigation established that coverage reproduced exactly and only the
  conformal radius differed, at the level of the integrator's own declared
  precision. The tolerance was amended — with the new value derived from the
  integrator's configuration, not from the observed discrepancy, and with the
  author's wrong prior estimate recorded in the amendment itself.
- A binding criterion **bit**, bounding the operating domain instead of
  confirming a hypothesis.
- Cross-auditing — whoever drafts does not audit — **caught a blocking defect
  in every stage of the cycle**. In v7.7-b, a classifier that would have
  emitted a silent, wrong verdict. In v7.7-c, a completeness check missing
  from the regression gates, and a contradiction between an amendment and the
  consolidator, both introduced by the drafter.

Every one of those episodes is documented in the minutes, including the ones
that cost days.

---

## Verify it yourself

```bash
git clone https://github.com/joaquinpenafiel/conformal-prediction-three-body
cd conformal-prediction-three-body
python verify.py
```

No dependencies — standard library only, Python 3.8+. Three independent
checks:

1. **Manifest.** Recomputes the SHA-256 of all 44 packaged files and compares
   against `MANIFEST.sha256`.
2. **Canonical dataset.** Regenerates the dataset hash from
   `data/mtps_c_v7_7a_configs.json` using the stable serialisation defined in
   the freezing minutes, and compares against the value the abort gates
   verified at the start of every run. This checks the *content*, not the
   file.
3. **Derivation chain.** Confirms that the hashes each diff artefact declares
   in its header match the manifest, so that
   v7.7-a → v7.7-b → v7.7-c is verified rather than asserted.

### Measured portability

The short demonstration run (`demo/run_demo.py`) was executed on an
independent machine — Windows on an Intel i3, Python 3.10, numpy 2.2.6, scipy
1.15.3 — against results produced on Linux with Python 3.13 and numpy 2.1.3.
Across 120 compared cells:

| Quantity | Maximum difference |
|---|---|
| empirical coverage | `0.000e+00` — exact |
| explained variance | `2.93e-13` |
| conformal radius | `8.60e-11` relative |

Coverage, being a ratio of integers, reproduces exactly. The radius agrees
well within the integrator's declared `rtol` of 1e-8. Determinism and
portability of the method are therefore measured, not assumed.

---

## Layout

```
docs/INDEX.md        what each document contains, in English
docs/preregistros/   frozen preregistrations, one per stage
docs/actas/          freezing minutes, amendments, authorisation annexes,
                     closing minutes — current versions only
docs/informes/       results reports
docs/notas/          technical note on threshold collapse, literature map,
                     audit note
src/v7_7a|b|c/       common module, runner, consolidator, diff against the
                     previous stage
data/                the 60 canonical configurations
results/v7_7a|b|c/   metrics, per-cell tables, horizon series, verdicts,
                     per-configuration timings
```

Documents are in Spanish and are **not** translated: their SHA-256 is what
makes the chain verifiable, and any re-export would break it. `docs/INDEX.md`
describes each of them in English, with the SHA-256 the manifest records. Do not open the
`.docx` files in a word processor and save them — that alters the hash even if
the text does not change.

---

## Reproducing the runs

The three `*_table_long.csv` files are excluded from the repository: together
they weigh about 30 MB and are regenerated by running the consolidator of each
stage. Everything needed to do so is included.

Requirements: `numpy`, `scipy`, `scikit-learn`, `matplotlib`. The runs were
executed on Python 3.13.15 with numpy 2.1.3 on Linux.

```python
import sys
sys.path.insert(0, "src/v7_7c")
sys.path.insert(0, "src/v7_7b")
sys.path.insert(0, "src/v7_7a")

from mtps_c_v7_7c_runner import run_stage_c
from mtps_c_v7_7c_consolidate import verify_g14

run_stage_c(H=2, drive_base="outputs")   # intermediate control point
verify_g14(H=2, base="outputs")          # gate must pass before the next arm
run_stage_c(H=4, drive_base="outputs")   # operative variable, 40 snapshots
```

The `drive_base` argument matters: the modules default to the Google Drive
path where the original runs were executed, so a local clone must redirect it.
Checkpoints, partials and outputs are written there and the run resumes from
them if interrupted.

**Expect this to be slow, and read the cost note below before starting.**

---

## Cost, and a design lesson that transfers

Registered compute for v7.7-c was 25.25 hours, inside the 17–28 hour budget
declared before running. Wall-clock time was **six days**.

The gap is the interesting part. Four configurations — the `encuentro_violento`
family, where bodies undergo close encounters and the adaptive integrator
shrinks its step drastically — concentrate roughly two thirds to four fifths of
total compute in every stage. One of them needs about nine hours of continuous
computation. The runner checkpointed **per configuration**, so every platform
disconnection discarded the entire attempt. Five days were spent on a single
configuration across attempts of seven, eight and eight and a half hours, none
of which survived.

The fix was to checkpoint **per integration** — the smallest independent,
deterministic unit — via a content-addressed cache of integrator calls,
verified bit-exact against the uncached path. Measured in production: after a
disconnection, a run recovered 60 integrations in 15 seconds where roughly six
hours had previously been lost.

The transferable rule, recorded in the closing minutes: with multi-hour units
of work, checkpoint granularity is not an optimisation — it determines whether
the experiment is executable at all. Registered compute also becomes a *lower
bound*, since interrupted attempts leave no trace.

---

## What this work does not claim

Stated in the reports and repeated here:

- Simulated data only. The method assumes replicas of the same initial
  condition exist — a real observed system does not provide them. Until it is
  validated on a case where the stratum exists without being manufactured,
  this is a well-executed proof of concept.
- Same dataset and seeds across all three stages. These are preregistered
  sensitivity experiments, not independent validations.
- v7.7-c ran without a negative control arm, with the three reasons declared
  in advance and the attribution limit recorded.
- No new knowledge about three-body dynamics is claimed. The contribution is
  methodological.

---

## Citing

See `CITATION.cff`, or:

> Peñafiel González, J. (2026). *MTPS-C: preregistered conformal prediction
> for chaotic three-body systems.* ORCID 0009-0005-2614-9393.

---

## Licence

Code (`src/`, `verify.py`) under **Apache 2.0** — see `LICENSE`.
Documents, data and results (`docs/`, `data/`, `results/`) under
**CC BY 4.0** — see `LICENSE-DOCS`.

---

## Status

The v7.7 characterisation cycle is closed, with no further internal stage
planned. Closing the cycle does not close the program: subsequent work is
applied or extends the domain.
