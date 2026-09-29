"""
================================================================================
MTPS-C v7.7-b — CONSOLIDADOR
================================================================================

Carga los tres partial_b_m{M}.pkl, ejecuta la puerta G11 como PRIMER paso,
produce el veredicto mecánico de escenario según la § 4 del Borrador 8 y
escribe las salidas de cierre.

ORDEN DE PUERTAS (condición C8 del Acta 01)
-------------------------------------------
  G13  — al inicio de cada par (brazo, M)      -> en el runner
  G12  — antes de evaluar en M = 40 y M = 50   -> en el runner
  G11  — antes de avanzar a M = 40 y M = 50    -> aquí, y como primer paso
         del consolidado

USO
---
    # después del stage M=25, antes de correr M=40 y M=50:
    from mtps_c_v7_7b_consolidate import verify_g11
    verify_g11()

    # con los tres partial listos:
    !python mtps_c_v7_7b_consolidate.py

SALIDAS en MyDrive/mtps_c/
    mtps_c_v7_7b_metrics.txt
    mtps_c_v7_7b_table_long.csv
    mtps_c_v7_7b_table_cells.csv
    mtps_c_v7_7b_trend.csv
    mtps_c_v7_7b_proxy_tau.csv
    veredicto_prerregistrado_v7_7b.json
================================================================================
"""

import csv
import json
import os
import pickle

import numpy as np

from mtps_c_v7_7b_common import (
    M_VALUES, N_ENSEMBLE_FIXED, P_LEVELS, P_MAIN, P_HIGH,
    BAND_C11B, RATIO_TAU_MAX_C11F,
    GateFailure, gate_g11, classify_v77b_scenario, DATASET_SHA256,
    expected_coverage_for_M, n_calib_for_M,
)

DRIVE_BASE = '/content/drive/MyDrive/mtps_c'
TABLE_LONG_V77A = 'mtps_c_v7_7a_table_long.csv'


def _load_partial(M, base=DRIVE_BASE):
    path = os.path.join(base, f'partial_b_m{M}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"falta {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==============================================================================
# PUERTA G11 — se ejecuta antes que nada
# ==============================================================================

def verify_g11(base=DRIVE_BASE):
    """Compara el par de brazos de M = 25 contra el stage 1 de v7.7-a."""
    print("=" * 80)
    print("PUERTA G11 — regresión de M = 25 contra v7.7-a")
    print("=" * 80)
    ref_path = os.path.join(base, TABLE_LONG_V77A)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"falta el artefacto de referencia: {ref_path}")

    payload = _load_partial(25, base)
    max_difs = {}
    for slug, label in (('principal', 'PRINCIPAL M=25'), ('control', 'CONTROL M=25')):
        agg = payload.get(slug)
        if agg is None:
            raise GateFailure(f"G11: falta el brazo {slug} en partial_b_m25.pkl")
        max_difs[slug] = gate_g11(agg['rows_long'], ref_path, label, abort_dir=base)

    print("\n  G11 PASADA — el camino heredado no se alteró.")
    print("  Autorizado avanzar a M = 40 y M = 50.\n")
    return max_difs


# ==============================================================================
# REPORTE
# ==============================================================================

def _fmt_arm(agg, M):
    lines = [f"\n  M={M}  n_calib={agg['n_calib']}  ({agg['mode']}):"]
    for p in P_LEVELS:
        lines.append(
            f"    p={p:<5} Cov={100*agg['cov_global'][p]:6.2f}%  "
            f"esperado={100*agg['cov_esperada'][p]:6.2f}%  "
            f"desvío={100*agg['desvio_vs_esperado'][p]:+6.2f}pp "
            f"({agg['desvio_en_sigma'][p]:+5.2f}σ)  "
            f"[{'OK' if abs(agg['desvio_vs_esperado'][p]) <= BAND_C11B else 'FUERA'}]"
        )
    lines.append(
        f"    C11.a:{'OK' if agg['c11a'] else 'NO'}  "
        f"C11.b:{'OK' if agg['c11b'] else 'NO'}  "
        f"C11.c:{'OK' if agg['c11c'] else 'NO'} ({agg['n_levels_ok']}/5)  "
        f"C11.f(i):{'OK' if agg['c11f_i'] else 'NO'} (frac={agg['frac_size_ok']:.3f})  "
        f"C11.f(ii):{'OK' if agg['c11f_ii'] else 'NO'} "
        f"(τ0,95/τ0,90={agg['ratio_tau_95_90']:.4f} ≤ {RATIO_TAU_MAX_C11F})  "
        f"C11.d:{'OK' if agg['c11d'] else 'NO'} (std={100*agg['std_errcal_50']:.2f}pp)"
    )
    pd_ = agg['proxy_dist']
    lines.append(f"    proxy: mediana={pd_['mediana']:.4f}  Q1={pd_['q1']:.4f}  "
                 f"Q3={pd_['q3']:.4f}  max={pd_['max']:.4f}  (n={pd_['n']})")
    lines.append("    escalera τ: " + "  ".join(
        f"{p}={agg['escalera_tau'][p]:.4f}" for p in P_LEVELS))
    return lines


def build_report(principal, control, escenario, base=DRIVE_BASE):
    out = []
    out.append("MTPS-C v7.7-b — Sensibilidad del conjunto de calibración (barrido de M)")
    out.append("=" * 78)
    out.append("Prerregistro: Hoja de Ruta v7.7-b, Borrador 8")
    out.append("Acta de Congelamiento 01 + Enmienda 01")
    out.append(f"N = {N_ENSEMBLE_FIXED} fijo; método aux2b congelado; única variable: M")
    out.append("Puertas: G13 (dataset), G12 (índices), G11 (regresión M=25) — PASADAS")
    out.append("")
    out.append("=== BRAZO PRINCIPAL ===")
    for M in M_VALUES:
        if M in principal and principal[M]:
            out += _fmt_arm(principal[M], M)
    out.append("")
    out.append("=== BRAZO DE CONTROL (invariancia esperada) ===")
    for M in M_VALUES:
        if M in control and control[M]:
            a = control[M]
            out.append(f"\n  M={M}: niveles ≤10pp {a['n_levels_ok']}/5   " +
                       "  ".join(f"Cov({p})={100*a['cov_global'][p]:.2f}%"
                                 for p in P_LEVELS))
    out.append("")
    out.append("=== TENDENCIA POR NIVEL × M (PRINCIPAL) ===")
    head = f"  {'Nivel':<8}" + "".join(f"M={M:<12}" for M in M_VALUES)
    out.append(head)
    out.append("  " + "-" * (8 + 14 * len(M_VALUES)))
    for p in P_LEVELS:
        line = f"  p={p:<6.2f}"
        for M in M_VALUES:
            a = principal.get(M)
            line += (f"{100*a['cov_global'][p]:6.2f}%      " if a else " " * 14)
        out.append(line)
    out.append("")
    out.append("=== VEREDICTO MECÁNICO ===")
    out.append(f"  ESCENARIO: {escenario[0]} — {escenario[1]}")
    out.append(f"  {escenario[2]}")
    return "\n".join(out)


def save_outputs(principal, control, escenario, base=DRIVE_BASE):
    all_long, all_cells = [], []
    for d in (principal, control):
        for M, agg in d.items():
            if agg:
                all_long.extend(agg['rows_long'])
                all_cells.extend(agg['rows_per_cell'])

    for nombre, filas in (('mtps_c_v7_7b_table_long.csv', all_long),
                          ('mtps_c_v7_7b_table_cells.csv', all_cells)):
        if filas:
            with open(os.path.join(base, nombre), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
                w.writeheader()
                w.writerows(filas)
            print(f"  Guardado: {nombre} ({len(filas)} filas)")

    trend = []
    for M in M_VALUES:
        a = principal.get(M)
        if not a:
            continue
        row = {'M': M, 'n_calib': a['n_calib'], 'n_ensemble': N_ENSEMBLE_FIXED}
        for p in P_LEVELS:
            k = int(100 * p)
            row[f'cov_{k}'] = a['cov_global'][p]
            row[f'esperado_{k}'] = a['cov_esperada'][p]
            row[f'desvio_pp_{k}'] = 100 * a['desvio_vs_esperado'][p]
            row[f'desvio_sigma_{k}'] = a['desvio_en_sigma'][p]
        row.update({'n_levels_ok': a['n_levels_ok'],
                    'frac_size_ok': a['frac_size_ok'],
                    'ratio_tau_95_90': a['ratio_tau_95_90'],
                    'std_errcal_50': a['std_errcal_50'],
                    'c11a': a['c11a'], 'c11b': a['c11b'], 'c11c': a['c11c'],
                    'c11f_i': a['c11f_i'], 'c11f_ii': a['c11f_ii'], 'c11d': a['c11d']})
        trend.append(row)
    if trend:
        with open(os.path.join(base, 'mtps_c_v7_7b_trend.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(trend[0].keys()))
            w.writeheader()
            w.writerows(trend)
        print(f"  Guardado: mtps_c_v7_7b_trend.csv ({len(trend)} filas)")

    # C11.f(iii): distribución del proxy y escalera de τ
    pt = []
    for M in M_VALUES:
        a = principal.get(M)
        if not a:
            continue
        base_row = {'M': M, 'proxy_n': a['proxy_dist']['n'],
                    'proxy_mediana': a['proxy_dist']['mediana'],
                    'proxy_q1': a['proxy_dist']['q1'],
                    'proxy_q3': a['proxy_dist']['q3'],
                    'proxy_max': a['proxy_dist']['max'],
                    'ratio_tau_95_90': a['ratio_tau_95_90']}
        for p in P_LEVELS:
            base_row[f'tau_mediana_{int(100*p)}'] = a['escalera_tau'][p]
        pt.append(base_row)
    if pt:
        with open(os.path.join(base, 'mtps_c_v7_7b_proxy_tau.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(pt[0].keys()))
            w.writeheader()
            w.writerows(pt)
        print(f"  Guardado: mtps_c_v7_7b_proxy_tau.csv ({len(pt)} filas)")

    veredicto = {
        'prerregistro': 'v7.7-b Borrador 8',
        'acta': '01 + Enmienda 01',
        'puertas': {'G13': 'PASADA', 'G12': 'PASADA', 'G11': 'PASADA'},
        'criterios_por_M': {
            str(M): {k: bool(principal[M][k]) for k in
                     ('c11a', 'c11b', 'c11c', 'c11f_i', 'c11f_ii', 'c11d')}
            for M in M_VALUES if principal.get(M)
        },
        'escenario': escenario[0],
        'veredicto': escenario[1],
        'consecuencia': escenario[2],
        'limite_de_atribucion': (
            'Mismo dataset y mismas seeds que v7.7-a; es un experimento de '
            'sensibilidad prerregistrado, no una validación independiente.'),
    }
    with open(os.path.join(base, 'veredicto_prerregistrado_v7_7b.json'), 'w',
              encoding='utf-8') as f:
        json.dump(veredicto, f, indent=2, ensure_ascii=False)
    print("  Guardado: veredicto_prerregistrado_v7_7b.json")
    return veredicto


def main(base=DRIVE_BASE):
    print("=" * 80)
    print("MTPS-C v7.7-b — CONSOLIDACIÓN")
    print("=" * 80)

    # 1. G11 primero, siempre
    try:
        verify_g11(base)
        gates_ok = True
    except (GateFailure, FileNotFoundError) as e:
        print(f"\n  *** G11 NO PASÓ: {e}")
        print("  No se produce veredicto de escenario.")
        return None

    # 2. Cargar los tres stages
    principal, control, procedencias = {}, {}, {}
    for M in M_VALUES:
        payload = _load_partial(M, base)
        principal[M] = payload.get('principal')
        control[M] = payload.get('control')
        procedencias[M] = payload.get('procedencia')

    # 3. Procedencia coherente entre stages (C9)
    hashes = {M: (p or {}).get('dataset_sha256') for M, p in procedencias.items()}
    if len(set(h for h in hashes.values() if h)) > 1:
        print(f"\n  *** Procedencia inconsistente entre stages: {hashes}")
        print("  No se produce veredicto de escenario.")
        return None
    if any(h != DATASET_SHA256 for h in hashes.values()):
        # Red de seguridad de G13: coherencia entre stages no basta si el
        # dataset compartido no es el canónico registrado en el Acta 01.
        print(f"\n  *** Dataset no canónico en la procedencia: {hashes}")
        print(f"      esperado {DATASET_SHA256}")
        print("  No se produce veredicto de escenario.")
        return None
    print(f"  Procedencia canónica verificada: dataset {DATASET_SHA256[:16]}...")

    # 4. Veredicto mecánico
    escenario = classify_v77b_scenario(principal, gates_ok)

    reporte = build_report(principal, control, escenario, base)
    print("\n" + reporte)
    with open(os.path.join(base, 'mtps_c_v7_7b_metrics.txt'), 'w',
              encoding='utf-8') as f:
        f.write(reporte + "\n")
    print("\n  Guardado: mtps_c_v7_7b_metrics.txt")

    save_outputs(principal, control, escenario, base)
    print(f"\n{'=' * 80}\nCONSOLIDACIÓN COMPLETADA — {escenario[0]}\n{'=' * 80}")
    return escenario


if __name__ == '__main__':
    main()
