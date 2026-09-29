"""
================================================================================
MTPS-C v7.7-b — RUNNER CON CHECKPOINTS POR CONFIG
================================================================================

Derivado de mtps_c_v77a_ckpt_runner.py. Enmienda OPERATIVA, no metodológica:
importa evaluate_config_v77b y aggregate_results_v77b del módulo común de
v7.7-b y las llama con los mismos argumentos, en el mismo orden, sobre el
mismo dataset.

CAMBIOS RESPECTO DEL RUNNER DE v7.7-a
-------------------------------------
  - M es parámetro del stage; el par (brazo, M) es la unidad reanudable.
  - seed_gt_base[i] = 60000 + i*M   (Enmienda 01 al Acta 01, C6.6).
    Con M = 25 es idéntico al congelado: 60000 + i*25.
  - Puertas G13 y G12 al inicio de cada par, antes de computar nada.
  - Bloque de procedencia en cada partial (condición C9 del Acta 01).
  - Selección de brazos por M: principal en {25, 40, 50}, control en {25, 50}.

EQUIVALENCIA
------------
Las seeds siguen siendo POR CONFIG e independientes del orden de ejecución,
así que reanudar a mitad de camino produce el mismo resultado que correr de
un tirón.

ESTRUCTURA EN DRIVE
-------------------
    MyDrive/mtps_c/
        ckpt_b_m25/   principal_00.pkl ... control_59.pkl
        ckpt_b_m40/   principal_00.pkl ... principal_59.pkl
        ckpt_b_m50/   principal_00.pkl ... control_59.pkl
        partial_b_m25.pkl   partial_b_m40.pkl   partial_b_m50.pkl

USO
---
    from mtps_c_v7_7b_runner import run_stage_b
    run_stage_b(M=25)        # principal + control  (puerta de regresión)
    run_stage_b(M=40)        # principal
    run_stage_b(M=50)        # principal + control

Reanudar tras una desconexión: volver a ejecutar exactamente lo mismo.
================================================================================
"""

import json
import os
import pickle
import time

from mtps_c_v7_7b_common import (
    M_VALUES, N_ENSEMBLE_FIXED, MODES,
    P_LEVELS, P_MAIN, P_HIGH,
    GateFailure,
    generate_mixed_dataset,
    gate_g12, gate_g13,
    seed_ensemble_for, seed_gt_base_for,
    n_calib_for_M, expected_coverage_for_M,
    evaluate_config_v77b, aggregate_results_v77b,
    provenance_block,
)

DRIVE_BASE = '/content/drive/MyDrive/mtps_c'

# Brazos por M (decisión D4 del prerregistro)
ARMS_BY_M = {25: ['principal', 'control'],
             40: ['principal'],
             50: ['principal', 'control']}


def _atomic_dump(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _load_ckpt(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"    checkpoint corrupto, se recalcula: {os.path.basename(path)} ({e})")
        return None


def run_stage_b(M, drive_base=DRIVE_BASE, report_every=1):
    """Ejecuta el stage completo para un valor de M. Reanudable por par (brazo, M)."""
    if M not in M_VALUES:
        raise ValueError(f"M={M} no está en el prerregistro {M_VALUES}")

    ckpt_dir = os.path.join(drive_base, f'ckpt_b_m{M}')
    out_path = os.path.join(drive_base, f'partial_b_m{M}.pkl')
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 80)
    print(f"MTPS-C v7.7-b — STAGE M = {M}   (N = {N_ENSEMBLE_FIXED} fijo)")
    print("=" * 80)
    print("\nPrerregistro: Hoja de Ruta v7.7-b, Borrador 8")
    print("Acta 01 + Enmienda 01. Método aux2b congelado; única variable: M.")
    print(f"n_calib = piso(M/2) = {n_calib_for_M(M)}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Salida:      {out_path}\n")

    dataset = generate_mixed_dataset()

    # ---------------------------------------------------------------- PUERTAS
    print("PUERTAS PREVIAS AL CÓMPUTO")
    gate_g13(dataset, context=f"M={M}", abort_dir=drive_base)
    gate_g12(M, abort_dir=drive_base)
    cov_esp = expected_coverage_for_M(M)
    print(f"  Coberturas esperadas: "
          f"{ {p: round(100*cov_esp[p], 2) for p in P_LEVELS} }\n")

    seeds_ens = [seed_ensemble_for(i) for i in range(len(dataset))]
    seeds_gt = [seed_gt_base_for(i, M) for i in range(len(dataset))]

    print(f"Dataset mixto: {len(dataset)} configs")
    counts = {}
    for c in dataset:
        counts[c['subset']] = counts.get(c['subset'], 0) + 1
    for s, n in counts.items():
        print(f"  {s}: {n} configs")

    arms = ARMS_BY_M[M]
    print(f"Brazos en este stage: {', '.join(arms)}\n")

    aggregated = {}
    total_compute_seconds = 0.0

    for mode, slug, label in MODES:
        if slug not in arms:
            print(f"\n(omitido {label} en M={M}, conforme a D4)")
            continue

        print(f"\n{'=' * 80}")
        print(f"EJECUTANDO {label}  M={M}  N={N_ENSEMBLE_FIXED}")
        print("=" * 80)

        all_results = [None] * len(dataset)
        pending = []
        for i in range(len(dataset)):
            ck = os.path.join(ckpt_dir, f'{slug}_{i:02d}.pkl')
            payload = _load_ckpt(ck)
            if payload is not None:
                r = payload.get('result')
                cid_ck = r.get('config_id') if isinstance(r, dict) else None
                m_ck = r.get('M') if isinstance(r, dict) else None
                if cid_ck != dataset[i]['config_id'] or (m_ck is not None and m_ck != M):
                    print(f"    checkpoint {i:02d} no corresponde a "
                          f"{dataset[i]['config_id']} / M={M} "
                          f"(trae {cid_ck} / M={m_ck}), se recalcula")
                    payload = None
            if payload is not None:
                all_results[i] = payload['result']
                total_compute_seconds += payload.get('seconds', 0.0)
            else:
                pending.append(i)

        done = len(dataset) - len(pending)
        if done:
            print(f"  Reanudando: {done}/{len(dataset)} configs ya en checkpoint")
        if not pending:
            print(f"  {label} completo desde checkpoints, sin recomputar")

        t0 = time.time()
        for count, i in enumerate(pending, start=1):
            t_cfg = time.time()
            result = evaluate_config_v77b(
                dataset[i], seeds_ens[i], seeds_gt[i], mode,
                n_ensemble=N_ENSEMBLE_FIXED, M=M
            )
            secs = time.time() - t_cfg
            total_compute_seconds += secs

            all_results[i] = result
            _atomic_dump({'result': result, 'seconds': secs, 'M': M},
                         os.path.join(ckpt_dir, f'{slug}_{i:02d}.pkl'))

            if count % report_every == 0 or count == len(pending):
                el = time.time() - t0
                eta = el / count * (len(pending) - count)
                print(f"  ...{done + count}/{len(dataset)} "
                      f"[cfg {i:02d} {dataset[i]['subset']}] "
                      f"{secs:.0f}s | acum {el:.0f}s | ETA {eta:.0f}s", flush=True)

        print(f"  {label} completado: {len(dataset)} configs")
        aggregated[slug] = aggregate_results_v77b(
            all_results, f'{label} M={M}', M
        )

    print(f"\n  Tiempo de cómputo acumulado: {total_compute_seconds/60:.1f} min "
          f"({total_compute_seconds/3600:.2f} h)")

    # ------------------------------------------------------------- RESUMEN
    p = aggregated.get('principal')
    if p:
        print(f"\n  PRINCIPAL M={M}  (n_calib={p['n_calib']}):")
        for lv in P_LEVELS:
            print(f"    p={lv:<5} Cov={100*p['cov_global'][lv]:6.2f}%  "
                  f"esperado={100*p['cov_esperada'][lv]:6.2f}%  "
                  f"desvío={100*p['desvio_vs_esperado'][lv]:+6.2f}pp  "
                  f"({p['desvio_en_sigma'][lv]:+5.2f}σ)")
        print(f"    C11.a:{'OK' if p['c11a'] else 'NO'} "
              f"C11.b:{'OK' if p['c11b'] else 'NO'} "
              f"C11.c:{'OK' if p['c11c'] else 'NO'} "
              f"C11.f(i):{'OK' if p['c11f_i'] else 'NO'} "
              f"C11.f(ii):{'OK' if p['c11f_ii'] else 'NO'} "
              f"C11.d:{'OK' if p['c11d'] else 'NO'}")
        print(f"    τ0,95/τ0,90 = {p['ratio_tau_95_90']:.4f}   "
              f"proxy mediana={p['proxy_dist']['mediana']:.4f} "
              f"(Q1={p['proxy_dist']['q1']:.4f}, Q3={p['proxy_dist']['q3']:.4f})")

    ctrl = aggregated.get('control')
    if ctrl:
        print(f"\n  CONTROL M={M}:")
        print(f"    Niveles ≤10pp: {ctrl['n_levels_ok']}/5   "
              f"Cov(0,90)={100*ctrl['cov_global'][0.90]:.2f}%   "
              f"Cov(0,95)={100*ctrl['cov_global'][P_HIGH]:.2f}%")

    payload = {
        'M': M,
        'n_ensemble': N_ENSEMBLE_FIXED,
        'n_calib': n_calib_for_M(M),
        'principal': aggregated.get('principal'),
        'control': aggregated.get('control'),
        'elapsed_seconds': total_compute_seconds,
        'procedencia': provenance_block(dataset, M),
    }
    _atomic_dump(payload, out_path)

    print(f"\n  Resultado guardado: {out_path}")
    print(f"  Procedencia: {json.dumps(payload['procedencia'], ensure_ascii=False)}")
    print(f"\n{'=' * 80}")
    print(f"Stage M={M} completado.")
    print("=" * 80)
    if M == 25:
        print("\n  SIGUIENTE: verificar la puerta G11 antes de correr M=40 y M=50.")
        print("  from mtps_c_v7_7b_consolidate import verify_g11")
        print("  verify_g11()")
    return payload
