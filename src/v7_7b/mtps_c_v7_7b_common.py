"""
================================================================================
MTPS-C v7.7-b — MÓDULO COMÚN
Sensibilidad del Conjunto de Calibración — Barrido de M
================================================================================

PRERREGISTRO: Hoja de Ruta v7.7-b, Borrador 8
  SHA-256  b25a08a68b7017ba0040f0f96b5d66f790b27db7f2fa6bf84ad9aab2f69cf635
ACTA:       Acta de Congelamiento 01
  SHA-256  6f35a27f20768fcb95e298b6c1e03038c0b1764c44990aa2a0e7836a7f3f2859
BASE:       mtps_c_v7_7a_common.py (condición C6)
  SHA-256  d0a7b151ff6d456ab2cf51cfbebfddb98b93696d6e676b0c7e8d80930dacf74e

DIFF DECLARADO (C6, con la Enmienda 01 al Acta 01)
--------------------------------------------------
Este módulo NO reimplementa el método. Importa del módulo congelado toda la
física, los generadores, el integrador, el PCA, el score kNN y el umbral
conformal, y solo introduce los siete cambios permitidos:

  1. n_calib = piso(M/2) en reemplazo de la constante N_GT_CALIB = 12.
  2. Puertas G11, G12 y G13.
  3. Bloque de procedencia exigido por C9.
  4. Reportes de C11.f(iii): distribución del proxy por celda y escalera de
     medianas de tau por nivel y M.
  5. Infraestructura de checkpoints y reanudación (en el runner, sin efecto
     sobre el cómputo).
  6. Espaciado de semillas de ground truth proporcional a M (Enmienda 01):
     seed_gt_base[i] = 60000 + i*M en lugar de 60000 + i*25. Con M = 25 es
     idéntico al congelado; con M = 40 o 50 evita que dos configuraciones
     compartan semillas de ground truth.
  7. Guardia de n_calib ante fallos de integración (Enmienda 01):
     min(M//2, n_gt_total//2). En el caso normal es exactamente piso(M/2);
     solo actúa si alguna integración de ground truth falló, que es la
     función que ya cumplía el guardia original.

RETROCOMPATIBILIDAD
-------------------
Con M = 25 este módulo debe reproducir el stage 1 de v7.7-a exactamente:
n_calib = 12, 13 ground truths de evaluación en el brazo principal, 25 en el
de control, y el mismo espaciado de semillas. Esa igualdad es la puerta G11.
================================================================================
"""

import hashlib
import json
import math
import os
import platform
import sys

import numpy as np
from scipy.stats import chi2

from mtps_c_v7_7a_common import (  # base congelada — no se modifica
    P_LEVELS, P_MAIN, P_HIGH, P_LEVEL_FOR_SIZE,
    N_SNAPSHOTS, K_NEIGHBORS, N_SPLITS,
    EPS_PRED_DEFAULT, EPS_REAL_DEFAULT,
    ERRCAL_THRESHOLD, SIZE_PROXY_MAX, SIZE_PROXY_FRACTION_MIN,
    SEED_INDIST, SEED_OOD_NEAR, SEED_OOD_FAR,
    generate_mixed_dataset,
    compute_T_adaptive, integrate_with_snapshots, perturb_state,
    compute_knn_score, compute_conformal_threshold, fit_pca,
)

# ==============================================================================
# CONSTANTES PROPIAS DE v7.7-b (prerregistradas)
# ==============================================================================

M_VALUES = [25, 40, 50]
N_ENSEMBLE_FIXED = 50

BAND_C11B = 0.02                 # +-2 pp alrededor del esperado
RATIO_TAU_MAX_C11F = 1.076       # umbral de agudeza 0,95/0,90
ERRCAL_MAX_C11C = 0.10
STD_MAX_C11D = 0.05

# Error estándar por nivel del bootstrap sobre configuraciones de v7.7-a
# (prerregistrado en la § 3 del Borrador 8; solo para reportar en unidades
# de sigma, no interviene en el criterio vinculante)
SE_BOOTSTRAP_PP = {0.30: 0.537, 0.50: 0.530, 0.80: 0.450, 0.90: 0.345, 0.95: 0.345}

# Índices de umbral prerregistrados (§ 2.2 del Borrador 8) — los verifica G12
EXPECTED_INDICES = {
    25: [3, 6, 10, 11, 11],
    40: [6, 10, 16, 18, 19],
    50: [7, 12, 20, 23, 24],
}

# Hash canónico del dataset (C4 del Acta 01) — lo verifica G13
DATASET_SHA256 = "b2b0937a069d0170d9379ae08919067b9fff3e5ffc967a9f580425a80efedf47"

MODES = [
    ('v75b_aux2b',   'principal', 'PRINCIPAL'),
    ('v75a_control', 'control',   'CONTROL'),
]


class GateFailure(Exception):
    """Fallo de puerta: emite artefacto de aborto y detiene el experimento."""


# ==============================================================================
# ENMIENDA DE MÉTODO
# ==============================================================================

def n_calib_for_M(M, n_gt_total=None):
    """n_calib = piso(M/2), con el guardia original ante fallos de integración.

    Con M = 25 y sin fallos devuelve 12, idéntico al método congelado.
    """
    base = M // 2
    if n_gt_total is None:
        return base
    return min(base, n_gt_total // 2)


def expected_indices_for_M(M):
    """Índices de umbral que la regla produce para cada nivel de la grilla."""
    n = n_calib_for_M(M)
    return [min(max(int(math.ceil((n + 1) * p)) - 1, 0), n - 1) for p in P_LEVELS]


def expected_coverage_for_M(M):
    """Cobertura esperada por nivel: (idx + 1) / (n_calib + 1)."""
    n = n_calib_for_M(M)
    return {p: (i + 1) / (n + 1) for p, i in zip(P_LEVELS, expected_indices_for_M(M))}


def seed_gt_base_for(i, M):
    """Espaciado proporcional a M (Enmienda 01). Con M = 25 es 60000 + i*25."""
    return 60000 + i * M


def seed_ensemble_for(i):
    return 50000 + i


# ==============================================================================
# PUERTA G13 — identidad del dataset
# ==============================================================================

def dataset_canonical_hash(dataset):
    """Serialización estable definida en la § 5 del Acta 01."""
    lineas = [
        f"{c['config_id']}|{c['subset']}|{c['regime_or_subtype']}|{c['source_seed']}"
        for c in dataset
    ]
    blob = ("\n".join(lineas) + "\n").encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def gate_g13(dataset, expected=DATASET_SHA256, context="", abort_dir="."):
    got = dataset_canonical_hash(dataset)
    ok = got == expected
    if not ok:
        _emit_abort("G13", {
            "contexto": context,
            "hash_esperado": expected,
            "hash_obtenido": got,
            "n_configs": len(dataset),
        }, abort_dir)
        raise GateFailure(f"G13 {context}: dataset {got} != {expected}")
    print(f"  G13 OK  {context}  dataset {got[:16]}...  ({len(dataset)} configs)")
    return got


# ==============================================================================
# PUERTA G12 — índices prerregistrados
# ==============================================================================

def gate_g12(M, abort_dir="."):
    got = expected_indices_for_M(M)
    exp = EXPECTED_INDICES[M]
    if got != exp:
        _emit_abort("G12", {"M": M, "indices_esperados": exp,
                            "indices_obtenidos": got}, abort_dir)
        raise GateFailure(f"G12 M={M}: {got} != {exp}")
    cov = expected_coverage_for_M(M)
    print(f"  G12 OK  M={M}  índices {got}  "
          f"esperados {[round(100*cov[p], 2) for p in P_LEVELS]}%")
    return got


# ==============================================================================
# PUERTA G11 — regresión contra v7.7-a en M = 25
# ==============================================================================

G11_TOL = 1e-12


def gate_g11(rows_long_b, table_long_v77a_path, mode_label, abort_dir="."):
    """Compara por configuración y por celda contra table_long.csv de v7.7-a.

    Artefacto de referencia fijado en la condición C5 del Acta 01:
    mtps_c_v7_7a_table_long.csv filtrada a n_ensemble = 50, emparejando por
    config_id, snapshot, split_id y p_level, con tolerancia 1e-12 sobre
    cov_emp y tau_p.
    """
    import csv
    prefix = mode_label.split()[0]           # PRINCIPAL | CONTROL
    ref = {}
    with open(table_long_v77a_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(float(r["n_ensemble"])) != 50:
                continue
            if not r["mode"].startswith(prefix):
                continue
            key = (r["config_id"], int(float(r["snapshot"])),
                   int(float(r["split_id"])), round(float(r["p_level"]), 10))
            ref[key] = (float(r["cov_emp"]), float(r["tau_p"]))

    if not ref:
        _emit_abort("G11", {"motivo": "referencia vacía", "modo": mode_label,
                            "archivo": table_long_v77a_path}, abort_dir)
        raise GateFailure("G11: no se encontraron filas de referencia")

    faltantes, difieren, max_dif = [], [], 0.0
    for r in rows_long_b:
        key = (r["config_id"], int(r["snapshot"]),
               int(r["split_id"]), round(float(r["p_level"]), 10))
        if key not in ref:
            faltantes.append(key)
            continue
        d_cov = abs(r["cov_emp"] - ref[key][0])
        d_tau = abs(r["tau_p"] - ref[key][1])
        max_dif = max(max_dif, d_cov, d_tau)
        if d_cov > G11_TOL or d_tau > G11_TOL:
            difieren.append({"celda": key, "d_cov": d_cov, "d_tau": d_tau})

    n_ref, n_got = len(ref), len(rows_long_b)
    problemas = bool(faltantes) or bool(difieren) or n_ref != n_got
    if problemas:
        _emit_abort("G11", {
            "modo": mode_label,
            "filas_referencia": n_ref, "filas_obtenidas": n_got,
            "celdas_faltantes": len(faltantes),
            "celdas_que_difieren": len(difieren),
            "primeras_diferencias": difieren[:10],
            "max_dif_abs": max_dif, "tolerancia": G11_TOL,
        }, abort_dir)
        raise GateFailure(f"G11 {mode_label}: {len(difieren)} celdas difieren, "
                          f"{len(faltantes)} faltantes, max_dif={max_dif:.3e}")

    print(f"  G11 OK  {mode_label}  {n_got} celdas  max_dif={max_dif:.3e} "
          f"(tol {G11_TOL:.0e})")
    return max_dif


def _emit_abort(gate, detalle, abort_dir="."):
    payload = {"estado": "ABORTADA", "puerta": gate,
               "prerregistro": "v7.7-b Borrador 8", "acta": "01",
               "detalle": detalle,
               "nota": "No se produce veredicto de escenario."}
    os.makedirs(abort_dir, exist_ok=True)
    path = os.path.join(abort_dir, f"ABORTO_{gate}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  *** ABORTO {gate} — artefacto en {path}\n")


# ==============================================================================
# PROCEDENCIA (condición C9)
# ==============================================================================

def provenance_block(dataset, M, module_path=None):
    blk = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "plataforma": platform.platform(),
        "dataset_sha256": dataset_canonical_hash(dataset),
        "M": M,
        "n_calib": n_calib_for_M(M),
        "n_ensemble": N_ENSEMBLE_FIXED,
        "modulo_comun_sha256": None,
    }
    path = module_path or os.path.abspath(__file__)
    try:
        with open(path, "rb") as f:
            blk["modulo_comun_sha256"] = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        pass
    return blk


# ==============================================================================
# EVALUACIÓN POR CONFIG — parametrizada por M
# ==============================================================================

def evaluate_config_v77b(config, seed_ensemble, seed_gt_base, mode,
                         n_ensemble=N_ENSEMBLE_FIXED, M=25):
    """Derivada de evaluate_config_v77a. Único cambio: M y n_calib parametrizados."""
    masses = np.array(config['masses'])
    state_0 = np.array(config['state_0'])
    T_max = compute_T_adaptive(state_0, masses)

    rng_pred = np.random.default_rng(seed=seed_ensemble)
    ensemble_trajs = []
    for _ in range(n_ensemble):
        state_0_pred = perturb_state(state_0, EPS_PRED_DEFAULT, rng_pred)
        traj, _ = integrate_with_snapshots(masses, state_0_pred, T_max)
        if traj is None:
            continue
        ensemble_trajs.append(traj)
    if len(ensemble_trajs) < n_ensemble // 2:
        return {'config_id': config['config_id'], 'subset': config['subset'],
                'integration_ok': False, 'reason': 'ensemble_failed'}
    ensemble_trajs = np.array(ensemble_trajs)

    ground_truths, times = [], None
    for m_idx in range(M):
        rng_gt = np.random.default_rng(seed=seed_gt_base + m_idx)
        state_0_gt = perturb_state(state_0, EPS_REAL_DEFAULT, rng_gt)
        traj_gt, times_gt = integrate_with_snapshots(masses, state_0_gt, T_max)
        if traj_gt is None:
            continue
        times = times_gt
        ground_truths.append(traj_gt)
    if len(ground_truths) < M // 2:
        return {'config_id': config['config_id'], 'subset': config['subset'],
                'integration_ok': False, 'reason': 'gt_failed'}
    ground_truths = np.array(ground_truths)
    n_gt_total = len(ground_truths)

    if mode == 'v75b_aux2b':
        rng_split_master = np.random.default_rng(seed=seed_gt_base + 99999)
        splits = []
        for _ in range(N_SPLITS):
            indices = rng_split_master.permutation(n_gt_total)
            n_calib_actual = n_calib_for_M(M, n_gt_total)
            splits.append((indices[:n_calib_actual], indices[n_calib_actual:]))
    else:
        splits = [(None, np.arange(n_gt_total))]

    snapshots_data = []
    for k in range(N_SNAPSHOTS):
        ensemble_k = ensemble_trajs[:, k, :]
        pca, ensemble_pca, d_eff_raw, d_eff_model = fit_pca(ensemble_k)

        cov_pca_mat = np.cov(ensemble_pca.T) + 1e-10 * np.eye(d_eff_model)
        eigvals_cov = np.linalg.eigvalsh(cov_pca_mat)
        radio_env_95 = float(np.sqrt(chi2.ppf(0.95, df=d_eff_model)
                                     * np.max(eigvals_cov)))

        per_split_data = []
        for s_idx, (calib_idx, eval_idx) in enumerate(splits):
            if mode == 'v75a_control':
                scores_calib = np.array([
                    compute_knn_score(ensemble_pca[i], ensemble_pca,
                                      k=K_NEIGHBORS, exclude_self=True, self_idx=i)
                    for i in range(ensemble_pca.shape[0])
                ])
            else:
                gt_pca_all = pca.transform(ground_truths[:, k, :])
                scores_calib = np.array([
                    compute_knn_score(gt_pca_all[idx], ensemble_pca,
                                      k=K_NEIGHBORS, exclude_self=False)
                    for idx in calib_idx
                ])

            tau_p = {p: compute_conformal_threshold(scores_calib, p) for p in P_LEVELS}

            if mode == 'v75b_aux2b':
                eval_gts_pca = gt_pca_all[eval_idx]
            else:
                eval_gts_pca = pca.transform(ground_truths[:, k, :])

            n_eval = len(eval_gts_pca)
            cov_per_p = {p: 0 for p in P_LEVELS}
            for i in range(n_eval):
                score_gt = compute_knn_score(eval_gts_pca[i], ensemble_pca,
                                             k=K_NEIGHBORS, exclude_self=False)
                for p in P_LEVELS:
                    if score_gt <= tau_p[p]:
                        cov_per_p[p] += 1
            cov_rates = {p: cov_per_p[p] / n_eval for p in P_LEVELS}

            size_proxy = (float((tau_p[P_LEVEL_FOR_SIZE] / radio_env_95) ** d_eff_model)
                          if radio_env_95 > 0 else float('inf'))

            per_split_data.append({
                'split_id': s_idx, 'tau_p': tau_p, 'cov_rates': cov_rates,
                'size_proxy': size_proxy,
                'n_calib': len(calib_idx) if calib_idx is not None else n_ensemble,
                'n_eval': n_eval,
            })

        cov_rates_mean = {p: float(np.mean([s['cov_rates'][p] for s in per_split_data]))
                          for p in P_LEVELS}
        tau_mean = {p: float(np.mean([s['tau_p'][p] for s in per_split_data]))
                    for p in P_LEVELS}
        proxies = [s['size_proxy'] for s in per_split_data if np.isfinite(s['size_proxy'])]
        snapshots_data.append({
            'k': k, 't': times[k],
            'd_eff_raw': d_eff_raw, 'd_eff_model': d_eff_model,
            'cov_rates_mean': cov_rates_mean,
            'errcal_mean': {p: abs(cov_rates_mean[p] - p) for p in P_LEVELS},
            'tau_mean': tau_mean,
            'size_proxy_mean': float(np.mean(proxies)) if proxies else float('inf'),
            'per_split': per_split_data,
        })

    return {
        'config_id': config['config_id'], 'subset': config['subset'],
        'regime_or_subtype': config['regime_or_subtype'],
        'mode': mode, 'n_ensemble': n_ensemble, 'M': M,
        'n_calib': n_calib_for_M(M, n_gt_total),
        'integration_ok': True, 'snapshots': snapshots_data,
    }


# ==============================================================================
# AGREGACIÓN Y CRITERIOS C11
# ==============================================================================

def aggregate_results_v77b(all_results, mode_label, M):
    rows_long, rows_per_cell, n_failed = [], [], 0

    for r in all_results:
        if not r or not r.get('integration_ok', False):
            n_failed += 1
            continue
        for snap in r['snapshots']:
            cell = {
                'mode': mode_label, 'M': M, 'n_ensemble': N_ENSEMBLE_FIXED,
                'config_id': r['config_id'], 'subset': r['subset'],
                'regime_or_subtype': r['regime_or_subtype'],
                'snapshot': snap['k'] + 1, 't': snap['t'],
                'd_eff_raw': snap['d_eff_raw'], 'd_eff_model': snap['d_eff_model'],
                'size_proxy_mean': snap['size_proxy_mean'],
            }
            for p in P_LEVELS:
                cell[f'cov_{int(100*p)}_mean'] = snap['cov_rates_mean'][p]
                cell[f'errcal_{int(100*p)}_mean'] = snap['errcal_mean'][p]
                cell[f'tau_{int(100*p)}_mean'] = snap['tau_mean'][p]
            rows_per_cell.append(cell)

            for sd in snap['per_split']:
                for p in P_LEVELS:
                    rows_long.append({
                        'M': M, 'n_ensemble': N_ENSEMBLE_FIXED, 'mode': mode_label,
                        'subset': r['subset'], 'regime_or_subtype': r['regime_or_subtype'],
                        'config_id': r['config_id'], 'snapshot': snap['k'] + 1,
                        'split_id': sd['split_id'], 'p_level': p,
                        'cov_emp': sd['cov_rates'][p],
                        'errcal': abs(sd['cov_rates'][p] - p),
                        'tau_p': sd['tau_p'][p], 'size_proxy': sd['size_proxy'],
                    })

    if not rows_per_cell:
        return None

    cov_global = {p: float(np.mean([c[f'cov_{int(100*p)}_mean'] for c in rows_per_cell]))
                  for p in P_LEVELS}
    errcal_global = {p: abs(cov_global[p] - p) for p in P_LEVELS}
    esperados = expected_coverage_for_M(M)

    # --- C11.a: separación de coberturas realizadas en el nivel alto
    c11a = abs(cov_global[0.90] - cov_global[P_HIGH]) > 0

    # --- C11.b: cada cobertura dentro de +-2 pp de su esperado, con sigma
    desvios, en_sigma = {}, {}
    for p in P_LEVELS:
        d = cov_global[p] - esperados[p]
        desvios[p] = d
        se = SE_BOOTSTRAP_PP[p] / 100.0
        en_sigma[p] = d / se if se > 0 else float('inf')
    c11b = all(abs(desvios[p]) <= BAND_C11B for p in P_LEVELS)

    # --- C11.c: calibración global
    n_levels_ok = sum(1 for p in P_LEVELS if errcal_global[p] <= ERRCAL_MAX_C11C)
    c11c = n_levels_ok == len(P_LEVELS)

    # --- C11.f: agudeza
    proxies = np.array([c['size_proxy_mean'] for c in rows_per_cell
                        if np.isfinite(c['size_proxy_mean'])])
    frac_size_ok = float(np.mean(proxies <= SIZE_PROXY_MAX)) if proxies.size else 0.0
    c11f_i = frac_size_ok >= SIZE_PROXY_FRACTION_MIN

    escalera_tau = {p: float(np.median([c[f'tau_{int(100*p)}_mean']
                                        for c in rows_per_cell])) for p in P_LEVELS}
    ratio_95_90 = (escalera_tau[P_HIGH] / escalera_tau[0.90]
                   if escalera_tau[0.90] > 0 else float('inf'))
    c11f_ii = (ratio_95_90 <= RATIO_TAU_MAX_C11F) if M in (40, 50) else True

    proxy_dist = {
        'n': int(proxies.size),
        'mediana': float(np.median(proxies)) if proxies.size else float('nan'),
        'q1': float(np.percentile(proxies, 25)) if proxies.size else float('nan'),
        'q3': float(np.percentile(proxies, 75)) if proxies.size else float('nan'),
        'max': float(proxies.max()) if proxies.size else float('nan'),
    }

    # --- C11.d: estabilidad entre splits
    split_ids = sorted({r['split_id'] for r in rows_long})
    errcals_split = []
    for sid in split_ids:
        vals = [r['cov_emp'] for r in rows_long
                if r['split_id'] == sid and abs(r['p_level'] - P_MAIN) < 1e-12]
        if vals:
            errcals_split.append(abs(float(np.mean(vals)) - P_MAIN))
    std_errcal_50 = float(np.std(errcals_split, ddof=1)) if len(errcals_split) > 1 else 0.0
    c11d = std_errcal_50 <= STD_MAX_C11D

    by_subset = {}
    for c in rows_per_cell:
        by_subset.setdefault(c['subset'], []).append(c)
    cov_by_subset = {
        s: {p: float(np.mean([c[f'cov_{int(100*p)}_mean'] for c in cs]))
            for p in P_LEVELS}
        for s, cs in by_subset.items()
    }

    return {
        'mode': mode_label, 'M': M, 'n_ensemble': N_ENSEMBLE_FIXED,
        'n_calib': n_calib_for_M(M),
        'rows_per_cell': rows_per_cell, 'rows_long': rows_long,
        'cov_global': cov_global, 'errcal_global': errcal_global,
        'cov_esperada': esperados, 'desvio_vs_esperado': desvios,
        'desvio_en_sigma': en_sigma,
        'n_levels_ok': n_levels_ok, 'cov_by_subset': cov_by_subset,
        'c11a': c11a, 'c11b': c11b, 'c11c': c11c,
        'c11f_i': c11f_i, 'c11f_ii': c11f_ii, 'c11d': c11d,
        'frac_size_ok': frac_size_ok, 'proxy_dist': proxy_dist,
        'escalera_tau': escalera_tau, 'ratio_tau_95_90': ratio_95_90,
        'std_errcal_50': std_errcal_50, 'n_failed': n_failed,
    }


# ==============================================================================
# ESCENARIO (§ 4 del Borrador 8) — veredicto mecánico
# ==============================================================================

def classify_v77b_scenario(results_principal, gates_ok):
    """Veredicto mecánico según la § 4 del Borrador 8.

    ALCANCE DE CADA CRITERIO (prerregistro § 3) — la separación importa:
      C11.a, C11.b y C11.f(ii): definidos SOLO en M = 40 y M = 50.
      C11.c y C11.f(i):         definidos en los tres M del brazo principal.

    En M = 25 los índices son {3, 6, 10, 11, 11} y los niveles 0,90 y 0,95
    colapsan al mismo umbral por construcción, de modo que C11.a no puede
    cumplirse ahí y no debe exigirse: ese brazo es la puerta de regresión
    G11, no una exigencia de separación. Evaluar C11.a sobre M = 25 haría
    inalcanzable el escenario S11.1.

    results_principal: dict {M: agregado}. gates_ok: G11, G12 y G13.
    """
    if not gates_ok:
        return ("S11.3", "ANOMALÍA — puerta fallida",
                "Alto e investigación; nada avanza a v7.7-c.")

    if any(M not in results_principal or results_principal[M] is None
           for M in M_VALUES):
        return ("S11.3", "ANOMALÍA — brazos incompletos",
                "Alto e investigación; nada avanza a v7.7-c.")

    def ok(a, key):
        return bool(a.get(key, False))

    todos = [results_principal[M] for M in M_VALUES]
    altos = [results_principal[M] for M in (40, 50)]

    c11a_altos = all(ok(a, 'c11a') for a in altos)
    c11b_altos = all(ok(a, 'c11b') for a in altos)
    c11fii_altos = all(ok(a, 'c11f_ii') for a in altos)
    c11c_todos = all(ok(a, 'c11c') for a in todos)
    c11fi_todos = all(ok(a, 'c11f_i') for a in todos)

    if (c11a_altos and c11b_altos and c11fii_altos
            and c11c_todos and c11fi_todos):
        return ("S11.1", "GRILLA HABILITADA",
                "v7.7-c hereda M = 40, conforme a la regla de consecuencia "
                "prerregistrada en la § 4.")

    if c11a_altos:
        return ("S11.2", "SEPARACIÓN CON DEGRADACIÓN",
                "Trade-off documentado; v7.7-c continúa con M = 25 y grilla de "
                "cuatro umbrales declarada.")

    return ("S11.3", "ANOMALÍA — coberturas lejos de los esperados",
            "Alto e investigación; nada avanza a v7.7-c.")
