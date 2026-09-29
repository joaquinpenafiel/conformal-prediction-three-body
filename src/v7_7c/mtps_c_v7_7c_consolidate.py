"""
================================================================================
MTPS-C v7.7-c — CONSOLIDADOR
================================================================================

Carga los dos partial_c_h{H}.pkl, ejecuta las puertas de regresión como
PRIMER paso, produce el veredicto mecánico de escenario según la § 4 del
prerregistro y escribe las salidas de cierre.

ORDEN DE PUERTAS (condición C8 del Acta 02)
-------------------------------------------
  G15, G16  — al inicio de cada par (brazo, H)      -> en el runner
  G14       — sobre H = 2 antes de lanzar H = 4     -> verify_g14(H=2)
  G14, G17  — tras completar ambos brazos           -> aquí, antes de todo

ALCANCES ACOTADOS (§ 3.1 del prerregistro, C5 del acta)
-------------------------------------------------------
  G14 aborta en los snapshots 1 a 9; el 10 es frontera de integración del
      horizonte de referencia y se reporta sin abortar.
  G17 aborta en los snapshots 1 a 19; el 20 es frontera de integración de
      H = 2 y se reporta sin abortar.
Los alcances están fijados por estructura y no pueden ampliarse ni
restringirse después de correr.

USO
---
    # después del stage H=2, antes de lanzar H=4:
    from mtps_c_v7_7c_consolidate import verify_g14
    verify_g14(H=2)

    # con los dos partial listos:
    !python mtps_c_v7_7c_consolidate.py

SALIDAS en MyDrive/mtps_c/
    mtps_c_v7_7c_metrics.txt
    mtps_c_v7_7c_table_long.csv
    mtps_c_v7_7c_table_cells.csv
    mtps_c_v7_7c_horizonte.csv
    mtps_c_v7_7c_prediccion.json
    veredicto_prerregistrado_v7_7c.json
================================================================================
"""

import csv
import json
import os
import pickle

import numpy as np

from mtps_c_v7_7c_common import (
    H_VALUES, H_OPERATIVA, H_CONTROL_INTERMEDIO,
    M_FIXED, N_ENSEMBLE_FIXED, P_LEVELS, P_HIGH,
    VE_UMBRAL, VE_BASE_FIJA, DATASET_SHA256,
    GateFailure, gate_g14, gate_g17,
    classify_v77c_scenario, contrastar_prediccion,
    n_snapshots_for_H,
)

# Enmienda 02 al Acta 02. El partial de H = 2 se generó con el módulo previo a
# la enmienda y se conserva como cómputo válido, porque la enmienda modifica la
# función de comparación de las puertas y no evaluate_config_v77c. En
# consecuencia, el consolidador no puede exigir un único hash de módulo entre
# stages: debe aceptar la combinación que la enmienda autoriza y rechazar
# cualquier otra.
HASH_MODULO_PRE_ENMIENDA = "5539e083796e26e2254c4bea50cc4a0b45f9c6b0dc7d6b66c8f47da0c657885e"
HASH_MODULO_ENMENDADO = "a6c37a3b8931e87223d25103965ea8392adb61b14831cb1f95b4c00844f038d6"

DRIVE_BASE = '/content/drive/MyDrive/mtps_c'
REFERENCIA_G14 = 'mtps_c_v7_7b_table_long.csv'


def _load_partial(H, base=DRIVE_BASE):
    path = os.path.join(base, f'partial_c_h{H}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"falta {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==============================================================================
# PUERTAS DE REGRESIÓN
# ==============================================================================

def verify_g14(H, base=DRIVE_BASE):
    """Regresión del brazo H contra el stage M = 40 de v7.7-b."""
    print("=" * 80)
    print(f"PUERTA G14 — regresión de H = {H} contra v7.7-b M = 40")
    print("=" * 80)
    ref_path = os.path.join(base, REFERENCIA_G14)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"falta el artefacto de referencia: {ref_path}")

    agg = _load_partial(H, base).get('principal')
    if agg is None:
        raise GateFailure(f"G14: falta el brazo principal en partial_c_h{H}.pkl")
    res = gate_g14(agg['rows_long'], ref_path, H, abort_dir=base)

    # Marcador de puerta pasada: el runner lo exige antes de lanzar H = 4,
    # de modo que el orden de puertas de la C8 quede forzado por el código
    # y no dependa de que el operador siga el protocolo manualmente.
    marcador = os.path.join(base, f'g14_ok_h{H}.json')
    with open(marcador, 'w', encoding='utf-8') as f:
        json.dump({'puerta': 'G14', 'H': int(H), 'estado': 'PASADA',
                   'max_dif_cobertura': res['max_dif_cobertura'],
                   'max_dif_tau': res['max_dif_tau'],
                   'max_tau_en_unidades_de_tol': res['max_tau_en_unidades_de_tol'],
                   'celdas_comparadas': res['celdas_comparadas'],
                   'referencia': REFERENCIA_G14,
                   'enmienda': '02 al Acta 02 — cobertura y tau con tolerancias separadas'},
                  f, indent=2, ensure_ascii=False)

    print(f"\n  G14 PASADA para H = {H} — el horizonte de referencia se reproduce.")
    print(f"  Marcador escrito: {os.path.basename(marcador)}")
    if H == H_CONTROL_INTERMEDIO:
        print("  Autorizado avanzar a H = 4.\n")
    return res


def verify_g17(base=DRIVE_BASE):
    """Independencia de camino: H = 2 contra los primeros 20 snapshots de H = 4."""
    print("=" * 80)
    print("PUERTA G17 — independencia de camino entre horizontes")
    print("=" * 80)
    a2 = _load_partial(H_CONTROL_INTERMEDIO, base).get('principal')
    a4 = _load_partial(H_OPERATIVA, base).get('principal')
    if a2 is None or a4 is None:
        raise GateFailure("G17: falta alguno de los dos brazos")
    res = gate_g17(a2['rows_long'], a4['rows_long'], abort_dir=base)
    print("\n  G17 PASADA — extender el horizonte no altera los instantes previos.\n")
    return res


# ==============================================================================
# REPORTE
# ==============================================================================

def build_report(resultados, escenario, prediccion, base=DRIVE_BASE):
    out = []
    out.append("MTPS-C v7.7-c — Horizonte útil (profundidad temporal)")
    out.append("=" * 78)
    out.append("Prerregistro: Hoja de Ruta v7.7-c, Borrador 5")
    out.append("Acta de Congelamiento 02 + Anexo C7-c")
    out.append(f"M = {M_FIXED} fijo; N = {N_ENSEMBLE_FIXED} fijo; "
               "única variable: el horizonte")
    out.append("Sin brazo de control (D4), con el límite de atribución declarado")
    out.append("Puertas G14, G15, G16 y G17 — PASADAS")
    out.append("")
    out.append(f"=== CRITERIO PRIMARIO C12.a — varianza explicada por base fija de "
               f"{VE_BASE_FIJA} componentes, umbral {VE_UMBRAL} ===")
    for H in H_VALUES:
        a = resultados.get(H)
        if not a:
            continue
        estado = "SE CUMPLE" if a['c12a'] else \
                 f"INCUMPLE desde el snapshot {a['primer_snapshot_incumple_c12a']}"
        out.append(f"\n  H = {H}  ({a['n_snapshots']} snapshots): {estado}")

    out.append("")
    out.append("=== SERIES POR SNAPSHOT (H = 4) ===")
    a = resultados.get(H_OPERATIVA)
    if a:
        out.append(f"  {'snap':>5} {'VE base fija':>13} {'d_eff_raw':>10} "
                   f"{'d_eff_model':>12} {'tau/radio':>11} {'tau':>11} {'C12.a':>7}")
        out.append("  " + "-" * 74)
        for s in range(1, a['n_snapshots'] + 1):
            ve = a['ve_por_snapshot'][s]
            ok = "OK" if ve >= VE_UMBRAL else "NO"
            out.append(f"  {s:>5} {ve:13.4f} {a['d_eff_raw_por_snapshot'][s]:10.2f} "
                       f"{a['d_eff_model_por_snapshot'][s]:12.2f} "
                       f"{a['tau_radio_por_snapshot'][s]:11.4f} "
                       f"{a['tau_por_snapshot'][s]:11.5f} {ok:>7}")

    out.append("")
    out.append("=== CRITERIO C12.c — calibración por bloques de diez snapshots ===")
    for H in H_VALUES:
        a = resultados.get(H)
        if not a:
            continue
        out.append(f"\n  H = {H}:")
        for b, d in a['bloques_calibracion'].items():
            det = "  ".join(f"Cov({p})={100*d['cov'][p]:.2f}%" for p in P_LEVELS)
            out.append(f"    snapshots {b:>7}: {d['n_levels_ok']}/5   {det}")

    out.append("")
    out.append("=== CONTRASTE CON LA PREDICCIÓN PRERREGISTRADA (C11-c) ===")
    out.append(f"  Condición declarada: {prediccion['condicion_declarada']}")
    out.append(f"  {'snap':>5} {'d_eff_raw pred':>15} {'observado':>11} "
               f"{'tau/radio pred':>15} {'observado':>11}")
    for s, d in sorted(prediccion['puntos'].items()):
        obs_d = d['d_eff_raw_observado']
        obs_t = d['tau_radio_observado']
        out.append(f"  {s:>5} {d['d_eff_raw_predicho']:15.2f} "
                   f"{(obs_d if obs_d is not None else float('nan')):11.2f} "
                   f"{d['tau_radio_predicho']:15.4f} "
                   f"{(obs_t if obs_t is not None else float('nan')):11.4f}")

    out.append("")
    out.append("=== POR SUBCONJUNTO — d_eff_raw (H = 4, análisis secundario) ===")
    if a and a.get('por_subset'):
        hitos = [s for s in (1, 10, 20, 40) if s <= a['n_snapshots']]
        out.append(f"  {'subset':<18}" + "".join(f"{'s'+str(s):>10}" for s in hitos))
        for sub, d in a['por_subset'].items():
            out.append(f"  {sub:<18}" + "".join(
                f"{d['d_eff_raw'][s]:10.2f}" for s in hitos))

    out.append("")
    out.append("=== VEREDICTO MECÁNICO ===")
    out.append(f"  ESCENARIO: {escenario[0]} — {escenario[1]}")
    out.append(f"  {escenario[2]}")
    return "\n".join(out)


def save_outputs(resultados, escenario, prediccion, base=DRIVE_BASE, procedencias=None):
    all_long, all_cells = [], []
    for H, agg in resultados.items():
        if agg:
            all_long.extend(agg['rows_long'])
            all_cells.extend(agg['rows_per_cell'])

    for nombre, filas in (('mtps_c_v7_7c_table_long.csv', all_long),
                          ('mtps_c_v7_7c_table_cells.csv', all_cells)):
        if filas:
            with open(os.path.join(base, nombre), 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
                w.writeheader()
                w.writerows(filas)
            print(f"  Guardado: {nombre} ({len(filas)} filas)")

    # Serie de horizonte: una fila por (H, snapshot)
    horiz = []
    for H in H_VALUES:
        a = resultados.get(H)
        if not a:
            continue
        for s in range(1, a['n_snapshots'] + 1):
            fila = {'H': H, 'snapshot': s,
                    've_base_fija': a['ve_por_snapshot'][s],
                    've_umbral': VE_UMBRAL,
                    'c12a_cumple': bool(a['ve_por_snapshot'][s] >= VE_UMBRAL),
                    'd_eff_raw': a['d_eff_raw_por_snapshot'][s],
                    'd_eff_model': a['d_eff_model_por_snapshot'][s],
                    'tau_radio_95': a['tau_radio_por_snapshot'][s],
                    'tau_95': a['tau_por_snapshot'][s]}
            for p in P_LEVELS:
                fila[f'tau_mediana_{int(100*p)}'] = a['escalera_tau'][s][p]
            horiz.append(fila)
    if horiz:
        with open(os.path.join(base, 'mtps_c_v7_7c_horizonte.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(horiz[0].keys()))
            w.writeheader()
            w.writerows(horiz)
        print(f"  Guardado: mtps_c_v7_7c_horizonte.csv ({len(horiz)} filas)")

    with open(os.path.join(base, 'mtps_c_v7_7c_prediccion.json'), 'w',
              encoding='utf-8') as f:
        json.dump(prediccion, f, indent=2, ensure_ascii=False)
    print("  Guardado: mtps_c_v7_7c_prediccion.json")

    a4 = resultados.get(H_OPERATIVA)
    veredicto = {
        'prerregistro': 'v7.7-c Borrador 5',
        'acta': '02 + Anexo C7-c',
        'puertas': {'G14': 'PASADA', 'G15': 'PASADA',
                    'G16': 'PASADA', 'G17': 'PASADA'},
        'criterios_vinculantes': {
            str(H): {'c12a': bool(resultados[H]['c12a']),
                     'c12c': bool(resultados[H]['c12c']),
                     'primer_snapshot_incumple_c12a':
                         resultados[H]['primer_snapshot_incumple_c12a'],
                     'primer_bloque_incumple_c12c':
                         resultados[H]['primer_bloque_incumple_c12c']}
            for H in H_VALUES if resultados.get(H)
        },
        'escenario': escenario[0],
        'veredicto': escenario[1],
        'consecuencia': escenario[2],
        'contraste_prediccion': prediccion,
        'procedencia_de_modulos': {
            'H2': (procedencias or {}).get(H_CONTROL_INTERMEDIO, {}).get('modulo_comun_sha256'),
            'H4': (procedencias or {}).get(H_OPERATIVA, {}).get('modulo_comun_sha256'),
            'nota': ('Si los stages traen módulos distintos, la combinación está '
                     'autorizada por la Enmienda 02 al Acta 02: el partial de H = 2 '
                     'se generó antes de la enmienda, que modifica la función de '
                     'comparación de las puertas y no el cómputo.'),
        },
        'limite_de_atribucion': (
            'Mismo dataset y mismas seeds que v7.7-a y v7.7-b; es un experimento de '
            'horizonte prerregistrado, no una validación independiente. La etapa se '
            'corrió sin brazo de control, de modo que no puede descartarse que una '
            'frontera de cobertura fuera artefacto del brazo principal; el criterio '
            'primario, en cambio, no es de cobertura.'),
        'alcance_de_c12a': (
            'El incumplimiento de C12.a acota el régimen de compresibilidad '
            'bidimensional del ensemble. No declara la invalidez del método, que '
            'puede seguir operando con más componentes.'),
    }
    with open(os.path.join(base, 'veredicto_prerregistrado_v7_7c.json'), 'w',
              encoding='utf-8') as f:
        json.dump(veredicto, f, indent=2, ensure_ascii=False)
    print("  Guardado: veredicto_prerregistrado_v7_7c.json")
    return veredicto


def main(base=DRIVE_BASE):
    print("=" * 80)
    print("MTPS-C v7.7-c — CONSOLIDACIÓN")
    print("=" * 80)

    # 1. Puertas de regresión primero, siempre
    try:
        for H in H_VALUES:
            verify_g14(H, base)
        verify_g17(base)
        gates_ok = True
    except (GateFailure, FileNotFoundError) as e:
        print(f"\n  *** PUERTA NO PASÓ: {e}")
        print("  No se produce veredicto de escenario.")
        return None

    # 2. Cargar los stages
    resultados, procedencias = {}, {}
    for H in H_VALUES:
        payload = _load_partial(H, base)
        resultados[H] = payload.get('principal')
        procedencias[H] = payload.get('procedencia')

    # 3. Procedencia canónica y coherente (C9)
    hashes = {H: (p or {}).get('dataset_sha256') for H, p in procedencias.items()}
    if len(set(h for h in hashes.values() if h)) > 1:
        print(f"\n  *** Procedencia inconsistente entre stages: {hashes}")
        return None
    if any(h != DATASET_SHA256 for h in hashes.values()):
        print(f"\n  *** Dataset no canónico en la procedencia: {hashes}")
        print(f"      esperado {DATASET_SHA256}")
        return None

    # C9 exige registrar módulo, H, M y n_snapshots; se verifican todos, no
    # solo el dataset. Un stage corrido con otro módulo o con otro M no es
    # comparable aunque comparta el dataset.
    modulos = {H: (p or {}).get('modulo_comun_sha256') for H, p in procedencias.items()}
    mod_h2 = modulos.get(H_CONTROL_INTERMEDIO)
    mod_h4 = modulos.get(H_OPERATIVA)

    # Dos combinaciones son admisibles y ninguna otra.
    #   (a) Ambos stages con el módulo enmendado. Es el estado más limpio y el
    #       que resultaría de recorrer H = 2 después de la enmienda.
    #   (b) H = 2 con el módulo previo y H = 4 con el enmendado, que es la
    #       reutilización que la Enmienda 02 autoriza expresamente.
    caso_a = (mod_h2 == HASH_MODULO_ENMENDADO and mod_h4 == HASH_MODULO_ENMENDADO)
    caso_b = (mod_h2 == HASH_MODULO_PRE_ENMIENDA and mod_h4 == HASH_MODULO_ENMENDADO)
    if not (caso_a or caso_b):
        print(f"\n  *** Combinación de módulos no autorizada: {modulos}")
        print(f"      admisibles: ambos {HASH_MODULO_ENMENDADO[:16]}..., o bien "
              f"H=2 con {HASH_MODULO_PRE_ENMIENDA[:16]}... y H=4 con "
              f"{HASH_MODULO_ENMENDADO[:16]}...")
        print("  No se produce veredicto de escenario.")
        return None
    for H in H_VALUES:
        p = procedencias.get(H) or {}
        esperado = {'H': int(H), 'M': M_FIXED,
                    'n_snapshots': n_snapshots_for_H(H),
                    'n_ensemble': N_ENSEMBLE_FIXED}
        obtenido = {k: p.get(k) for k in esperado}
        if obtenido != esperado:
            print(f"\n  *** Procedencia incoherente en H={H}: "
                  f"esperado {esperado}, obtenido {obtenido}")
            return None
    print(f"  Procedencia canónica verificada: dataset {DATASET_SHA256[:16]}...")
    if caso_a:
        print(f"  Módulo común a los dos stages: {HASH_MODULO_ENMENDADO[:16]}... "
              f"(ambos posteriores a la Enmienda 02)")
    else:
        print(f"  Procedencia de módulos aceptada bajo la Enmienda 02:")
        print(f"    H={H_CONTROL_INTERMEDIO} generado con el módulo previo "
              f"{HASH_MODULO_PRE_ENMIENDA[:16]}...")
        print(f"    H={H_OPERATIVA} generado con el módulo enmendado "
              f"{HASH_MODULO_ENMENDADO[:16]}...")
        print(f"    La diferencia entre ambos no afecta evaluate_config_v77c ni "
              f"aggregate_results_v77c.")
    print(f"  H, M, n_snapshots y n_ensemble coherentes en los dos stages.")

    # 4. Veredicto mecánico y contraste de la predicción
    escenario = classify_v77c_scenario(resultados, gates_ok)
    prediccion = contrastar_prediccion(resultados[H_OPERATIVA])

    reporte = build_report(resultados, escenario, prediccion, base)
    print("\n" + reporte)
    with open(os.path.join(base, 'mtps_c_v7_7c_metrics.txt'), 'w',
              encoding='utf-8') as f:
        f.write(reporte + "\n")
    print("\n  Guardado: mtps_c_v7_7c_metrics.txt")

    save_outputs(resultados, escenario, prediccion, base, procedencias)
    print(f"\n{'=' * 80}\nCONSOLIDACIÓN COMPLETADA — {escenario[0]}\n{'=' * 80}")
    return escenario


if __name__ == '__main__':
    main()
