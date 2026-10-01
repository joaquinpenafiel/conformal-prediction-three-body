# Document index

All documents are in **Spanish** and are deliberately **not translated**: their
SHA-256 is what makes the audit chain verifiable, and any re-export would break
it. This index describes what each one contains, in English, with the hash the
manifest records.

Do not open the `.docx` files in a word processor and save them — that alters
the hash even if the text does not change.

---

## Preregistrations — `preregistros/`

Frozen before execution. Each states the single question, the single variable,
the binding criteria with their thresholds, the abort gates, the scenarios with
their consequence rule, and the compute budget.

| Document | Stage | SHA-256 |
|---|---|---|
| `Hoja_de_Ruta_MTPS-C_v7_7a_Penafiel.docx` | Robustness to ensemble size, N ∈ {50, 100, 200} | `1fc5fcf8…32e009a1` |
| `Hoja_de_Ruta_MTPS-C_v7_7b_BORRADOR8_Penafiel.docx` | Calibration-set size, M ∈ {25, 40, 50}. Eight audit rounds | `b25a08a6…f69cf635` |
| `Hoja_de_Ruta_MTPS-C_v7_7c_BORRADOR5_Penafiel.docx` | Temporal horizon, H = 4 with control point at H = 2. Five audit rounds | `56949b19…6735ee66` |

The v7.7-c preregistration inverts the scenario logic of the earlier two:
measuring the boundary is the favourable outcome, and not finding it within the
explored range is a legitimate but incomplete result.

---

## Minutes, amendments and annexes — `actas/`

Only current versions are published. Superseded issues are listed in the
history section of the root `README.md`.

### Stage v7.7-b

| Document | What it does | SHA-256 |
|---|---|---|
| `ACTA_01_CONGELAMIENTO_MTPS-C_v7_7b.docx` | Freezes the preregistration, rules on the design decisions, sets ten conditions including the canonical dataset hash and the ban on in-place correction | `6f35a27f…f3f2859` |
| `ENMIENDA_01_al_ACTA_01_MTPS-C_v7_7b.docx` | Two changes found while writing the module: seed spacing proportional to M, and a guard on the calibration-set size | `68c93188…1ae805` |
| `ANEXO_C7_al_ACTA_01_MTPS-C_v7_7b.docx` | **Authorises the run.** Records the four code artefacts with their hashes and the cross-audit finding: a classifier that would have made the favourable scenario unreachable | `be9ecc9b…8087db` |
| `ACTA_CIERRE_FORMAL_MTPS-C_v7_7b_v2.docx` | Closes the stage on verdict S11.1 — grid enabled. Anchors report, code and run outputs | `89d11213…0931fb` |

### Stage v7.7-c

| Document | What it does | SHA-256 |
|---|---|---|
| `ACTA_02_CONGELAMIENTO_MTPS-C_v7_7c.docx` | Freezes the preregistration and rules on the design decisions, including running without a control arm for the first time in the program | `00411e57…4bae54` |
| `ENMIENDA_02_al_ACTA_02_MTPS-C_v7_7c.docx` | **Issued after a gate aborted the run.** Separates the tolerance for coverage from the tolerance for the conformal radius, deriving the latter from the integrator's own `rtol` rather than from the observed discrepancy. Records the author's wrong prior estimate | `cabb6299…4df968a` |
| `ENMIENDA_03_al_ACTA_02_MTPS-C_v7_7c.docx` | Operational amendment: checkpointing inside the configuration, after five days lost on a single one. Verified bit-exact against the uncached path | `ca32bf54…6451b78` |
| `ANEXO_C7c_REEMISION3_al_ACTA_02_MTPS-C_v7_7c.docx` | **Authorises the run.** Third issue; the first two were superseded by the amendments above. Records the code hashes and the cross-audit findings | `9f5e21b2…f75e11f` |
| `ACTA_CIERRE_FORMAL_MTPS-C_v7_7c.docx` | Closes the stage on verdict S12.1 — boundary characterised — **and the v7.7 characterisation cycle**. Anchors the full package | `2f16b91e…421a952b` |

---

## Results reports — `informes/`

| Document | Verdict | SHA-256 |
|---|---|---|
| `Informe_de_Resultados_MTPS-C_v7_7a_Penafiel.docx` | 4/4 criteria, 5/5 levels at all three ensemble sizes | `9fafe5fa…b8a527` |
| `Informe_de_Resultados_MTPS-C_v7_7b_Penafiel.docx` | S11.1 — grid enabled, M = 40 inherited | `cf0c5f6f…18677e0` |
| `Informe_de_Resultados_MTPS-C_v7_7c_Penafiel.docx` | S12.1 — boundary characterised at snapshot 26, sustained from 30 | `47e4e1be…77186d8b` |

Each report states, in its own dedicated section, what the result does **not**
claim. The v7.7-c report also contrasts the preregistered prediction, which
failed by a factor of two at its furthest point.

---

## Technical notes — `notas/`

| Document | Contents | SHA-256 |
|---|---|---|
| `Nota_Tecnica_NT01_MTPS-C_Penafiel_rev4.docx` | Why the two highest nominal levels collapsed to the same threshold with a small calibration set, and the prediction — closed 3/3 — that this would persist across ensemble sizes. The hypothesis that motivated stage v7.7-b | `24a31b41…4823e4f` |
| `LIT-00_Mapa_de_Literatura_MTPS-C_v1_0.docx` | Literature map: 17 entries over 12 works, 16 verified against the source PDF. Positions the method as Mondrian conformal prediction with a stratum predefined by physical configuration | `197d295f…35aebe66` |
| `NA-01_Auditoria_Hoja_de_Ruta_MTPS-C_v7_7b.docx` | Audit note on the v7.7-b preregistration, including the amendment recording that M = 50 has lower aggregate structural deviation — so choosing M = 40 is a declared priority, not a dominance claim | `eaa7570a…87b6991` |
| `Nota_Tecnica_NT02_MTPS-C_Penafiel.docx` | **Issued after the cycle closed.** Withdraws an exploratory conjecture from the v7.7-c report that the data refute, and records what the data show instead: the useful horizon varies by dynamical regime by more than an order of magnitude. Four of fifteen regimes never cross the structural criterion; three cross at the first snapshot | `d433a93c…c94d72c` |

---

## Where to start

A reviewer with fifteen minutes:

1. `informes/Informe_de_Resultados_MTPS-C_v7_7c_Penafiel.docx` — sections 4, 5
   and 10: the boundary, the fact that calibration never degrades, and what the
   result does not claim.
2. `actas/ENMIENDA_02_al_ACTA_02_MTPS-C_v7_7c.docx` — how a gate that aborted
   the run was resolved without weakening it after the fact.
3. `notas/Nota_Tecnica_NT02_MTPS-C_Penafiel.docx` — how a conjecture the
   program had itself published was withdrawn once the data refuted it.
4. `python verify.py` at the repository root.
