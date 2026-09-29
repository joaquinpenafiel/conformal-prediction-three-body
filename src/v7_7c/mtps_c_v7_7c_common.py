"""
================================================================================
MTPS-C v7.7-c — MÓDULO COMÚN
Horizonte útil — Profundidad temporal
================================================================================

PRERREGISTRO: Hoja de Ruta v7.7-c, Borrador 5
  SHA-256  56949b19b1c9572e2cd8a56181b3ac72c909642c6d8feb9dd1b601526735ee66
ACTA:       Acta de Congelamiento 02
  SHA-256  00411e57f7492aa7736dd3b7d40f9bf45eebbeb0089b4e1060a7563fd84bae54
BASE:       mtps_c_v7_7b_common.py (condición C6)
  SHA-256  9485b073ed7281f02f4662c686b62426db4cc60fdd4842b198a671dff610bc30

DIFF DECLARADO (C6 del Acta 02)
-------------------------------
Este módulo no reimplementa el método. Importa de la base de v7.7-b, y a
través de ella de la base congelada de v7.7-a, toda la física, los
generadores, el integrador, el PCA, el score kNN y el umbral conformal, y
solo introduce los seis cambios permitidos:

  1. N_SNAPSHOTS = 10*H y horizonte de integración H*T_adaptive, en
     reemplazo de la constante y del horizonte simple. El espaciado entre
     snapshots queda invariante: T_adaptive/10 para todo H.
  2. Puertas G14, G15, G16 y G17, con sus alcances acotados.
  3. Bloque de procedencia de C9, extendido para registrar H.
  4. Los tres reportes obligatorios de la § 3.4 del prerregistro:
     varianza explicada por una base FIJA de dos componentes, radio de la
     envolvente del ensemble por celda, y escalera de tau por nivel y
     snapshot.
  5. Reporte de d_eff_raw junto a d_eff_model (C12.e).
  6. Infraestructura de checkpoints por par (brazo, H) — en el runner —
     y el forzado en código del orden de puertas que exige la C8 del acta:
     el brazo H = 4 no arranca sin que G14 haya pasado sobre H = 2.

RETROCOMPATIBILIDAD
-------------------
Con espaciado constante, los primeros diez snapshots de cualquier H evalúan
exactamente los mismos instantes que v7.7-b con M = 40, y las semillas por
configuración son idénticas. Esa igualdad es la puerta G14. Los veinte
snapshots de H = 2 son los veinte primeros de H = 4: esa igualdad es G17.

Ambas puertas excluyen de su aborto el snapshot terminal del brazo de
referencia — el 10 en G14, el 20 en G17 — porque es frontera de integración
y un paso adaptativo lo trata distinto cuando es punto final que cuando es
punto intermedio. La exclusión está declarada por estructura y no por
tamaño de la discrepancia observada (§ 3.1 del prerregistro, C5 del acta).

Enmienda 02 al Acta 02: las puertas separan dos magnitudes. La cobertura
empírica es discreta —un cociente de enteros sobre los ground truths de
evaluación— y conserva la tolerancia estricta de 1e-12; cualquier diferencia
material en ella aborta. El radio conformal es continuo y se compara con una
tolerancia relativa derivada del rtol del integrador congelado, con piso
absoluto en 1e-12. La enmienda no altera el cómputo: evaluate_config_v77c y
aggregate_results_v77c quedan sin cambio.

La comparación de ambas puertas es bidireccional y de completitud: exige que
el conjunto de celdas de la corrida y el de la referencia coincidan dentro del
alcance de aborto, no solo que las celdas presentes en ambos den lo mismo. Sin
esa exigencia, una configuración que fallara su integración no produciría filas
y la puerta pasaría con las restantes.
================================================================================
"""

import hashlib
import json
import os
import platform
import sys

import numpy as np
from scipy.stats import chi2

# --- de la base congelada de v7.7-a, vía v7.7-b: física y maquinaria ---------
from mtps_c_v7_7a_common import (
    P_LEVELS, P_MAIN, P_HIGH, P_LEVEL_FOR_SIZE,
    K_NEIGHBORS, N_SPLITS,
    EPS_PRED_DEFAULT, EPS_REAL_DEFAULT,
    ERRCAL_THRESHOLD, SIZE_PROXY_MAX, SIZE_PROXY_FRACTION_MIN,
    PCA_VAR_THRESHOLD,
    generate_mixed_dataset,
    compute_T_adaptive, integrate_with_snapshots, perturb_state,
    compute_knn_score, compute_conformal_threshold, fit_pca,
)

# --- de la base de v7.7-b: reglas y utilidades ya congeladas -----------------
from mtps_c_v7_7b_common import (
    GateFailure, _emit_abort,
    DATASET_SHA256,
    dataset_canonical_hash,
    n_calib_for_M, expected_indices_for_M, expected_coverage_for_M,
    seed_ensemble_for, seed_gt_base_for,
    ERRCAL_MAX_C11C, STD_MAX_C11D,
)

# ==============================================================================
# CONSTANTES PROPIAS DE v7.7-c (prerregistradas)
# ==============================================================================

H_OPERATIVA = 4                  # variable operativa
H_CONTROL_INTERMEDIO = 2         # punto de control intermedio
H_VALUES = [H_CONTROL_INTERMEDIO, H_OPERATIVA]

M_FIXED = 40                     # heredado de v7.7-b por regla de consecuencia
N_ENSEMBLE_FIXED = 50
N_SNAPSHOTS_BASE = 10            # snapshots de H = 1, método congelado

# C12.a — criterio estructural primario
VE_BASE_FIJA = 2                 # base fija de dos componentes, no adaptativa
VE_UMBRAL = PCA_VAR_THRESHOLD    # 0,95 heredado del método congelado

TOL_PUERTAS = 1e-12               # tolerancia estricta, para magnitudes discretas

# Enmienda 02 al Acta 02. La tolerancia del radio conformal se deriva de la
# configuración del integrador congelado — solve_ivp con rtol=1e-8 — y no del
# resultado observado en ninguna corrida. Extender el horizonte no altera la
# cobertura, que es discreta y se reproduce de forma exacta, pero sí altera tau
# al nivel del ruido numérico que el propio integrador declara. Exigir 1e-12
# absoluto a una magnitud continua producida con rtol=1e-8 es exigir cuatro
# órdenes más de precisión de la que el método garantiza.
RTOL_INTEGRADOR = 1e-8

def tol_tau(tau_nuevo, tau_ref):
    """Tolerancia del radio conformal, con piso absoluto.

    El piso conserva TOL_PUERTAS para radios cercanos a cero, donde una
    tolerancia puramente relativa se volvería inutilizable.
    """
    return max(TOL_PUERTAS,
               RTOL_INTEGRADOR * max(abs(float(tau_nuevo)), abs(float(tau_ref))))

# Índices de umbral prerregistrados para M = 40 — los verifica G16
EXPECTED_INDICES_M40 = [6, 10, 16, 18, 19]

MODE_PRINCIPAL = 'v75b_aux2b'    # único brazo de esta etapa (D4: sin control)

# Predicción prerregistrada de la § 3.6 — no vinculante
PREDICCION = {
    'tasa_geometrica_d_eff_raw': 1.0383,
    'tasa_geometrica_tau_radio': 1.0221,
    'd_eff_raw': {10: 2.38, 20: 3.47, 40: 7.35},
    'tau_radio': {10: 0.0857, 20: 0.107, 40: 0.165},
    'condicion': 'supone tasa de crecimiento constante; nada garantiza que se mantenga',
}


# ==============================================================================
# ENMIENDA DE MÉTODO — horizonte
# ==============================================================================

def n_snapshots_for_H(H):
    """N_SNAPSHOTS = 10*H. Con H = 1 devuelve el valor congelado."""
    return N_SNAPSHOTS_BASE * int(H)


def T_for_H(state_0, masses, H):
    """Horizonte de integración H*T_adaptive.

    Combinado con n_snapshots = 10*H, el espaciado entre snapshots resulta
    T_adaptive/10 para todo H: los instantes evaluados de un H menor son un
    subconjunto exacto de los de un H mayor.
    """
    return float(H) * compute_T_adaptive(state_0, masses)


def dt_for(state_0, masses):
    """Espaciado entre snapshots, invariante en H."""
    return compute_T_adaptive(state_0, masses) / N_SNAPSHOTS_BASE


# ==============================================================================
# PUERTAS
# ==============================================================================

def gate_g15(dataset, expected=DATASET_SHA256, context="", abort_dir="."):
    """Identidad del dataset. Idéntica en mecánica a G13 de v7.7-b."""
    got = dataset_canonical_hash(dataset)
    if got != expected:
        _emit_abort("G15", {"contexto": context, "hash_esperado": expected,
                            "hash_obtenido": got, "n_configs": len(dataset)}, abort_dir)
        raise GateFailure(f"G15 {context}: dataset {got} != {expected}")
    print(f"  G15 OK  {context}  dataset {got[:16]}...  ({len(dataset)} configs)")
    return got


def gate_g16(abort_dir="."):
    """Índices de umbral prerregistrados para M = 40."""
    got = expected_indices_for_M(M_FIXED)
    if got != EXPECTED_INDICES_M40:
        _emit_abort("G16", {"M": M_FIXED, "indices_esperados": EXPECTED_INDICES_M40,
                            "indices_obtenidos": got}, abort_dir)
        raise GateFailure(f"G16: {got} != {EXPECTED_INDICES_M40}")
    cov = expected_coverage_for_M(M_FIXED)
    print(f"  G16 OK  M={M_FIXED}  índices {got}  n_calib={n_calib_for_M(M_FIXED)}")
    return got


def _cargar_referencia_v77b(path):
    """table_long.csv de v7.7-b filtrada a M = 40, modo principal."""
    import csv
    ref = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(float(r["M"])) != M_FIXED:
                continue
            if not r["mode"].startswith("PRINCIPAL"):
                continue
            key = (r["config_id"], int(float(r["snapshot"])),
                   int(float(r["split_id"])), round(float(r["p_level"]), 10))
            ref[key] = (float(r["cov_emp"]), float(r["tau_p"]))
    return ref


def _comparar(filas, ref, snaps_aborto, snaps_reporte, gate, contexto, abort_dir):
    """Compara filas contra una referencia, separando alcance de aborto y de reporte.

    La verificación es BIDIRECCIONAL y de completitud: no basta con que las
    celdas presentes coincidan, porque una configuración que fallara su
    integración simplemente no produciría filas y la puerta pasaría con las
    que quedan. Se exige por tanto que el conjunto de celdas observadas y el
    de celdas de referencia sean idénticos dentro del alcance de aborto.
    """
    en_alcance = set(snaps_aborto) | set(snaps_reporte)

    keys_ref = {k for k in ref if k[1] in en_alcance}
    keys_obs = {(r["config_id"], int(r["snapshot"]), int(r["split_id"]),
                 round(float(r["p_level"]), 10))
                for r in filas if int(r["snapshot"]) in en_alcance}

    ausentes = keys_ref - keys_obs          # la referencia las tiene, la corrida no
    sobrantes = keys_obs - keys_ref         # la corrida las tiene, la referencia no
    ausentes_aborto = {k for k in ausentes if k[1] in snaps_aborto}
    sobrantes_aborto = {k for k in sobrantes if k[1] in snaps_aborto}

    dif_aborto, dif_reporte = [], []
    max_cov_ab = max_tau_ab = max_ratio_ab = 0.0
    max_cov_rp = max_tau_rp = max_ratio_rp = 0.0
    for r in filas:
        s = int(r["snapshot"])
        if s not in en_alcance:
            continue
        key = (r["config_id"], s, int(r["split_id"]), round(float(r["p_level"]), 10))
        if key not in ref:
            continue                        # ya contabilizada como sobrante
        d_cov = abs(r["cov_emp"] - ref[key][0])
        d_tau = abs(r["tau_p"] - ref[key][1])
        t_tau = tol_tau(r["tau_p"], ref[key][1])
        excede = (d_cov > TOL_PUERTAS) or (d_tau > t_tau)
        detalle = {"celda": key, "dif_cov": d_cov, "dif_tau": d_tau,
                   "tol_tau": t_tau, "tau_en_unidades_de_tol": d_tau / t_tau}
        if s in snaps_aborto:
            max_cov_ab = max(max_cov_ab, d_cov)
            max_tau_ab = max(max_tau_ab, d_tau)
            max_ratio_ab = max(max_ratio_ab, d_tau / t_tau)
            if excede:
                dif_aborto.append(detalle)
        else:
            max_cov_rp = max(max_cov_rp, d_cov)
            max_tau_rp = max(max_tau_rp, d_tau)
            max_ratio_rp = max(max_ratio_rp, d_tau / t_tau)
            if excede:
                dif_reporte.append(detalle)

    if ausentes_aborto or sobrantes_aborto or dif_aborto:
        _emit_abort(gate, {
            "contexto": contexto,
            "celdas_de_referencia_ausentes_en_la_corrida": len(ausentes_aborto),
            "primeras_ausentes": sorted(ausentes_aborto)[:10],
            "celdas_de_la_corrida_ausentes_en_referencia": len(sobrantes_aborto),
            "primeras_sobrantes": sorted(sobrantes_aborto)[:10],
            "celdas_que_difieren_en_alcance_de_aborto": len(dif_aborto),
            "primeras_diferencias": dif_aborto[:10],
            "max_dif_cobertura": max_cov_ab,
            "max_dif_tau": max_tau_ab,
            "max_tau_en_unidades_de_su_tolerancia": max_ratio_ab,
            "tolerancia_cobertura": TOL_PUERTAS,
            "tolerancia_tau": f"max({TOL_PUERTAS}, {RTOL_INTEGRADOR} * |tau|)",
            "snapshots_con_aborto": sorted(snaps_aborto),
            "snapshots_solo_reporte": sorted(snaps_reporte),
        }, abort_dir)
        raise GateFailure(
            f"{gate} {contexto}: {len(ausentes_aborto)} celdas ausentes, "
            f"{len(sobrantes_aborto)} sobrantes, {len(dif_aborto)} difieren "
            f"(max_cov={max_cov_ab:.3e}, max_tau={max_tau_ab:.3e} = "
            f"{max_ratio_ab:.2f}x su tolerancia)")

    print(f"  {gate} OK  {contexto}  {len(keys_obs & keys_ref)} celdas comparadas "
          f"en snapshots {min(snaps_aborto)}-{max(snaps_aborto)}")
    print(f"        cobertura: max_dif={max_cov_ab:.3e} (tol {TOL_PUERTAS:.0e}) — "
          f"{'exacta' if max_cov_ab == 0.0 else 'dentro de tolerancia'}")
    print(f"        radio tau: max_dif={max_tau_ab:.3e} = {max_ratio_ab:.2f}x su "
          f"tolerancia (rtol del integrador = {RTOL_INTEGRADOR:.0e})")
    print(f"        completitud: 0 ausentes, 0 sobrantes en el alcance de aborto")
    if snaps_reporte:
        estado = "coincide" if not dif_reporte else f"{len(dif_reporte)} celdas difieren"
        extra = ""
        if ausentes - ausentes_aborto or sobrantes - sobrantes_aborto:
            extra = (f"; {len(ausentes - ausentes_aborto)} ausentes / "
                     f"{len(sobrantes - sobrantes_aborto)} sobrantes")
        print(f"        frontera de integración, solo reporte "
              f"(snapshot {sorted(snaps_reporte)}): cobertura {max_cov_rp:.3e}, "
              f"tau {max_tau_rp:.3e} = {max_ratio_rp:.2f}x su tolerancia — "
              f"{estado}{extra}")
    return {"max_dif_cobertura": max_cov_ab, "max_dif_tau": max_tau_ab,
            "max_tau_en_unidades_de_tol": max_ratio_ab,
            "max_dif_cobertura_reporte": max_cov_rp,
            "max_dif_tau_reporte": max_tau_rp,
            "celdas_comparadas": len(keys_obs & keys_ref),
            "celdas_reporte_que_difieren": len(dif_reporte)}


def gate_g14(rows_long, ref_path, H, abort_dir="."):
    """Regresión contra el stage M = 40 de v7.7-b.

    Alcance de aborto: snapshots 1 a 9. El snapshot 10 es frontera de
    integración del horizonte de referencia y se reporta sin abortar.
    """
    ref = _cargar_referencia_v77b(ref_path)
    if not ref:
        _emit_abort("G14", {"motivo": "referencia vacía", "archivo": ref_path}, abort_dir)
        raise GateFailure("G14: no se encontraron filas de referencia")
    return _comparar(rows_long, ref, set(range(1, 10)), {10},
                     "G14", f"H={H} vs v7.7-b M=40", abort_dir)


def gate_g17(rows_h2, rows_h4, abort_dir="."):
    """Independencia de camino entre horizontes: H = 2 contra H = 4.

    Alcance de aborto: snapshots 1 a 19. El snapshot 20 es frontera de
    integración de H = 2 y se reporta sin abortar.
    """
    ref = {(r["config_id"], int(r["snapshot"]), int(r["split_id"]),
            round(float(r["p_level"]), 10)): (r["cov_emp"], r["tau_p"])
           for r in rows_h4 if int(r["snapshot"]) <= 20}
    if not ref:
        _emit_abort("G17", {"motivo": "referencia vacía desde H=4"}, abort_dir)
        raise GateFailure("G17: sin filas de referencia en H=4")
    return _comparar(rows_h2, ref, set(range(1, 20)), {20},
                     "G17", "H=2 vs H=4", abort_dir)


# ==============================================================================
# PROCEDENCIA (C9)
# ==============================================================================

def provenance_block(dataset, H, module_path=None):
    blk = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "plataforma": platform.platform(),
        "dataset_sha256": dataset_canonical_hash(dataset),
        "H": int(H),
        "n_snapshots": n_snapshots_for_H(H),
        "M": M_FIXED,
        "n_calib": n_calib_for_M(M_FIXED),
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
# EVALUACIÓN POR CONFIG — parametrizada por H
# ==============================================================================

def evaluate_config_v77c(config, seed_ensemble, seed_gt_base, H,
                         n_ensemble=N_ENSEMBLE_FIXED, M=M_FIXED):
    """Derivada de evaluate_config_v77b. Cambios: horizonte parametrizado por
    H y los reportes estructurales de la § 3.4 del prerregistro."""
    masses = np.array(config['masses'])
    state_0 = np.array(config['state_0'])
    T_max = T_for_H(state_0, masses, H)
    n_snaps = n_snapshots_for_H(H)

    rng_pred = np.random.default_rng(seed=seed_ensemble)
    ensemble_trajs = []
    for _ in range(n_ensemble):
        state_0_pred = perturb_state(state_0, EPS_PRED_DEFAULT, rng_pred)
        traj, _ = integrate_with_snapshots(masses, state_0_pred, T_max,
                                           n_snapshots=n_snaps)
        if traj is None:
            continue
        ensemble_trajs.append(traj)
    if len(ensemble_trajs) < n_ensemble // 2:
        return {'config_id': config['config_id'], 'subset': config['subset'],
                'H': int(H), 'integration_ok': False, 'reason': 'ensemble_failed'}
    ensemble_trajs = np.array(ensemble_trajs)

    ground_truths, times = [], None
    for m_idx in range(M):
        rng_gt = np.random.default_rng(seed=seed_gt_base + m_idx)
        state_0_gt = perturb_state(state_0, EPS_REAL_DEFAULT, rng_gt)
        traj_gt, times_gt = integrate_with_snapshots(masses, state_0_gt, T_max,
                                                     n_snapshots=n_snaps)
        if traj_gt is None:
            continue
        times = times_gt
        ground_truths.append(traj_gt)
    if len(ground_truths) < M // 2:
        return {'config_id': config['config_id'], 'subset': config['subset'],
                'H': int(H), 'integration_ok': False, 'reason': 'gt_failed'}
    ground_truths = np.array(ground_truths)
    n_gt_total = len(ground_truths)

    rng_split_master = np.random.default_rng(seed=seed_gt_base + 99999)
    splits = []
    for _ in range(N_SPLITS):
        indices = rng_split_master.permutation(n_gt_total)
        n_calib_actual = n_calib_for_M(M, n_gt_total)
        splits.append((indices[:n_calib_actual], indices[n_calib_actual:]))

    snapshots_data = []
    for k in range(n_snaps):
        ensemble_k = ensemble_trajs[:, k, :]
        pca, ensemble_pca, d_eff_raw, d_eff_model = fit_pca(ensemble_k)

        # --- reporte estructural: varianza explicada por BASE FIJA (C12.a) ---
        ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
        ve_base_fija = float(ratios[:VE_BASE_FIJA].sum())

        cov_pca_mat = np.cov(ensemble_pca.T) + 1e-10 * np.eye(d_eff_model)
        eigvals_cov = np.linalg.eigvalsh(cov_pca_mat)
        radio_env_95 = float(np.sqrt(chi2.ppf(0.95, df=d_eff_model)
                                     * np.max(eigvals_cov)))

        per_split_data = []
        for s_idx, (calib_idx, eval_idx) in enumerate(splits):
            gt_pca_all = pca.transform(ground_truths[:, k, :])
            scores_calib = np.array([
                compute_knn_score(gt_pca_all[idx], ensemble_pca,
                                  k=K_NEIGHBORS, exclude_self=False)
                for idx in calib_idx
            ])
            tau_p = {p: compute_conformal_threshold(scores_calib, p) for p in P_LEVELS}

            eval_gts_pca = gt_pca_all[eval_idx]
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
            # --- reporte: cociente crudo, sin exponente (§ 3.4) ---
            tau_radio = {p: (float(tau_p[p] / radio_env_95) if radio_env_95 > 0
                             else float('inf')) for p in P_LEVELS}

            per_split_data.append({
                'split_id': s_idx, 'tau_p': tau_p, 'cov_rates': cov_rates,
                'size_proxy': size_proxy, 'tau_radio': tau_radio,
                'n_calib': len(calib_idx), 'n_eval': n_eval,
            })

        cov_rates_mean = {p: float(np.mean([s['cov_rates'][p] for s in per_split_data]))
                          for p in P_LEVELS}
        tau_mean = {p: float(np.mean([s['tau_p'][p] for s in per_split_data]))
                    for p in P_LEVELS}
        tau_radio_mean = {p: float(np.mean([s['tau_radio'][p] for s in per_split_data]))
                          for p in P_LEVELS}
        proxies = [s['size_proxy'] for s in per_split_data
                   if np.isfinite(s['size_proxy'])]

        snapshots_data.append({
            'k': k, 't': float(times[k]),
            'd_eff_raw': int(d_eff_raw), 'd_eff_model': int(d_eff_model),
            've_base_fija': ve_base_fija,
            'radio_env_95': radio_env_95,
            'cov_rates_mean': cov_rates_mean,
            'errcal_mean': {p: abs(cov_rates_mean[p] - p) for p in P_LEVELS},
            'tau_mean': tau_mean, 'tau_radio_mean': tau_radio_mean,
            'size_proxy_mean': float(np.mean(proxies)) if proxies else float('inf'),
            'per_split': per_split_data,
        })

    return {
        'config_id': config['config_id'], 'subset': config['subset'],
        'regime_or_subtype': config['regime_or_subtype'],
        'mode': MODE_PRINCIPAL, 'H': int(H), 'n_snapshots': n_snaps,
        'n_ensemble': n_ensemble, 'M': M,
        'n_calib': n_calib_for_M(M, n_gt_total),
        'integration_ok': True, 'snapshots': snapshots_data,
    }


# ==============================================================================
# AGREGACIÓN Y CRITERIOS C12
# ==============================================================================

def aggregate_results_v77c(all_results, H):
    rows_long, rows_per_cell, n_failed = [], [], 0
    n_snaps = n_snapshots_for_H(H)
    mode_label = f'PRINCIPAL H={H}'

    for r in all_results:
        if not r or not r.get('integration_ok', False):
            n_failed += 1
            continue
        for snap in r['snapshots']:
            cell = {
                'mode': mode_label, 'H': int(H), 'M': M_FIXED,
                'n_ensemble': N_ENSEMBLE_FIXED,
                'config_id': r['config_id'], 'subset': r['subset'],
                'regime_or_subtype': r['regime_or_subtype'],
                'snapshot': snap['k'] + 1, 't': snap['t'],
                'd_eff_raw': snap['d_eff_raw'], 'd_eff_model': snap['d_eff_model'],
                've_base_fija': snap['ve_base_fija'],
                'radio_env_95': snap['radio_env_95'],
                'size_proxy_mean': snap['size_proxy_mean'],
            }
            for p in P_LEVELS:
                k = int(100 * p)
                cell[f'cov_{k}_mean'] = snap['cov_rates_mean'][p]
                cell[f'errcal_{k}_mean'] = snap['errcal_mean'][p]
                cell[f'tau_{k}_mean'] = snap['tau_mean'][p]
                cell[f'tau_radio_{k}_mean'] = snap['tau_radio_mean'][p]
            rows_per_cell.append(cell)

            for sd in snap['per_split']:
                for p in P_LEVELS:
                    rows_long.append({
                        'H': int(H), 'M': M_FIXED, 'n_ensemble': N_ENSEMBLE_FIXED,
                        'mode': mode_label, 'subset': r['subset'],
                        'regime_or_subtype': r['regime_or_subtype'],
                        'config_id': r['config_id'], 'snapshot': snap['k'] + 1,
                        'split_id': sd['split_id'], 'p_level': p,
                        'cov_emp': sd['cov_rates'][p],
                        'errcal': abs(sd['cov_rates'][p] - p),
                        'tau_p': sd['tau_p'][p],
                        'tau_radio': sd['tau_radio'][p],
                        'size_proxy': sd['size_proxy'],
                    })

    if not rows_per_cell:
        return None

    def _por_snapshot(campo, agg=np.median):
        out = {}
        for s in range(1, n_snaps + 1):
            v = [c[campo] for c in rows_per_cell
                 if c['snapshot'] == s and np.isfinite(c[campo])]
            out[s] = float(agg(v)) if v else float('nan')
        return out

    # --- C12.a: varianza explicada por base fija (vinculante) ---------------
    ve_por_snapshot = _por_snapshot('ve_base_fija')
    snaps_incumple = sorted(s for s, v in ve_por_snapshot.items()
                            if v == v and v < VE_UMBRAL)
    c12a = len(snaps_incumple) == 0
    primer_incumplimiento = snaps_incumple[0] if snaps_incumple else None

    # --- C12.c: calibración sostenida por bloques de diez snapshots ---------
    bloques, c12c_por_bloque = {}, {}
    for b in range(n_snaps // N_SNAPSHOTS_BASE):
        lo, hi = b * N_SNAPSHOTS_BASE + 1, (b + 1) * N_SNAPSHOTS_BASE
        cov_b, err_b = {}, {}
        for p in P_LEVELS:
            v = [c[f'cov_{int(100*p)}_mean'] for c in rows_per_cell
                 if lo <= c['snapshot'] <= hi]
            cov_b[p] = float(np.mean(v)) if v else float('nan')
            err_b[p] = abs(cov_b[p] - p)
        n_ok = sum(1 for p in P_LEVELS if err_b[p] <= ERRCAL_MAX_C11C)
        bloques[f'{lo}-{hi}'] = {'cov': cov_b, 'errcal': err_b, 'n_levels_ok': n_ok}
        c12c_por_bloque[f'{lo}-{hi}'] = (n_ok == len(P_LEVELS))
    c12c = all(c12c_por_bloque.values())
    primer_bloque_incumple = next((b for b, ok in c12c_por_bloque.items() if not ok), None)

    # --- instrumentales ------------------------------------------------------
    tau_radio_alto = _por_snapshot(f'tau_radio_{int(100*P_HIGH)}_mean')
    tau_alto = _por_snapshot(f'tau_{int(100*P_HIGH)}_mean')
    d_raw = _por_snapshot('d_eff_raw', np.mean)
    d_model = _por_snapshot('d_eff_model', np.mean)

    proxies = np.array([c['size_proxy_mean'] for c in rows_per_cell
                        if np.isfinite(c['size_proxy_mean'])])
    frac_size_ok = float(np.mean(proxies <= SIZE_PROXY_MAX)) if proxies.size else 0.0

    split_ids = sorted({r['split_id'] for r in rows_long})
    errcals_split = []
    for sid in split_ids:
        v = [r['cov_emp'] for r in rows_long
             if r['split_id'] == sid and abs(r['p_level'] - P_MAIN) < 1e-12]
        if v:
            errcals_split.append(abs(float(np.mean(v)) - P_MAIN))
    std_errcal_50 = float(np.std(errcals_split, ddof=1)) if len(errcals_split) > 1 else 0.0

    escalera_tau = {}
    for s in range(1, n_snaps + 1):
        escalera_tau[s] = {p: float(np.median(
            [c[f'tau_{int(100*p)}_mean'] for c in rows_per_cell if c['snapshot'] == s]
        )) for p in P_LEVELS}

    por_subset = {}
    for sub in sorted({c['subset'] for c in rows_per_cell}):
        por_subset[sub] = {
            'd_eff_raw': {s: float(np.mean([c['d_eff_raw'] for c in rows_per_cell
                                            if c['snapshot'] == s and c['subset'] == sub]))
                          for s in range(1, n_snaps + 1)},
            've_base_fija': {s: float(np.median([c['ve_base_fija'] for c in rows_per_cell
                                                 if c['snapshot'] == s and c['subset'] == sub]))
                             for s in range(1, n_snaps + 1)},
        }

    return {
        'mode': mode_label, 'H': int(H), 'n_snapshots': n_snaps,
        'M': M_FIXED, 'n_calib': n_calib_for_M(M_FIXED),
        'rows_per_cell': rows_per_cell, 'rows_long': rows_long,
        'c12a': c12a, 'primer_snapshot_incumple_c12a': primer_incumplimiento,
        've_por_snapshot': ve_por_snapshot, 've_umbral': VE_UMBRAL,
        'c12c': c12c, 'bloques_calibracion': bloques,
        'primer_bloque_incumple_c12c': primer_bloque_incumple,
        'tau_radio_por_snapshot': tau_radio_alto,
        'tau_por_snapshot': tau_alto,
        'd_eff_raw_por_snapshot': d_raw, 'd_eff_model_por_snapshot': d_model,
        'escalera_tau': escalera_tau, 'por_subset': por_subset,
        'frac_size_ok': frac_size_ok, 'std_errcal_50': std_errcal_50,
        'c12d_instrumental': True, 'c12e_instrumental': True,
        'n_failed': n_failed,
    }


# ==============================================================================
# ESCENARIO (§ 4 del prerregistro) — veredicto mecánico
# ==============================================================================

def classify_v77c_scenario(resultados, gates_ok):
    """Lógica invertida respecto de v7.7-a y v7.7-b: medir la frontera es el
    resultado favorable, y no encontrarla es una cota inferior legítima pero
    incompleta.

    resultados: dict {H: agregado}. gates_ok: G14, G15, G16 y G17.
    """
    if not gates_ok:
        return ("S12.3", "ANOMALÍA — puerta fallida",
                "Alto e investigación; nada avanza.")

    if any(H not in resultados or resultados[H] is None for H in H_VALUES):
        return ("S12.3", "ANOMALÍA — brazos incompletos",
                "Alto e investigación; nada avanza.")

    a = resultados[H_OPERATIVA]
    incumple_estructural = not a['c12a']
    incumple_calibracion = not a['c12c']

    if incumple_estructural or incumple_calibracion:
        cuales = []
        if incumple_estructural:
            cuales.append(f"C12.a estructural desde el snapshot "
                          f"{a['primer_snapshot_incumple_c12a']}")
        if incumple_calibracion:
            cuales.append(f"C12.c de calibración desde el bloque "
                          f"{a['primer_bloque_incumple_c12c']}")
        return ("S12.1", "FRONTERA CARACTERIZADA",
                "El horizonte útil queda acotado empíricamente. Primero se degrada: "
                + "; ".join(cuales) + ". El dominio operativo del instrumento queda "
                "caracterizado en su eje temporal y la etapa aplicada hereda ese "
                "límite como parámetro de diseño.")

    return ("S12.2", "SIN FRONTERA EN EL RANGO",
            f"Los dos criterios vinculantes se cumplen en los {a['n_snapshots']} "
            "snapshots. El resultado es una cota inferior del horizonte útil, no su "
            "caracterización: el instrumento sostiene al menos cuatro veces el "
            "horizonte de referencia. Corresponde prerregistrar la extensión del "
            "rango como etapa propia, con el presupuesto recalculado sobre el costo "
            "medido aquí.")


# ==============================================================================
# CONTRASTE CON LA PREDICCIÓN PRERREGISTRADA (C11-c del Acta 02)
# ==============================================================================

def contrastar_prediccion(agregado):
    """Obligatorio en el informe, acierte o falle (C11-c)."""
    out = {'condicion_declarada': PREDICCION['condicion'], 'puntos': {}}
    for snap in (10, 20, 40):
        if snap > agregado['n_snapshots']:
            continue
        obs_d = agregado['d_eff_raw_por_snapshot'].get(snap)
        obs_t = agregado['tau_radio_por_snapshot'].get(snap)
        out['puntos'][snap] = {
            'd_eff_raw_predicho': PREDICCION['d_eff_raw'][snap],
            'd_eff_raw_observado': obs_d,
            'tau_radio_predicho': PREDICCION['tau_radio'][snap],
            'tau_radio_observado': obs_t,
        }
    return out
