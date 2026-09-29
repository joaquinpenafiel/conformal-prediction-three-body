"""
================================================================================
MTPS-C v7.7-c — RUNNER CON CHECKPOINTS POR CONFIG
================================================================================

Derivado de mtps_c_v7_7b_runner.py. Enmienda operativa, no metodológica:
importa evaluate_config_v77c y aggregate_results_v77c del módulo común de
v7.7-c y las llama con los mismos argumentos, en el mismo orden, sobre el
mismo dataset.

CAMBIOS RESPECTO DEL RUNNER DE v7.7-b
-------------------------------------
  - H es parámetro del stage; el par (brazo, H) es la unidad reanudable.
  - Un solo brazo, el principal: v7.7-c no lleva control (D4 del
    prerregistro, § 2.3, con su límite de atribución declarado).
  - Puertas G15 y G16 al inicio de cada par, antes de computar nada.
  - Bloque de procedencia extendido con H (C9 del Acta 02).
  - M = 40 fijo, heredado de v7.7-b por la regla de consecuencia.
  - Enmienda 03: checkpoint dentro de la configuración para la familia
    encuentro_violento, mediante una caché de integraciones indexada por el
    hash de sus entradas. El módulo común no se modifica.

SEMILLAS
--------
Sin cambios respecto de v7.7-b con M = 40:
    seeds_ens[i]     = 50000 + i
    seeds_gt_base[i] = 60000 + i*40
Son por configuración e independientes del orden de ejecución, de modo que
reanudar a mitad de camino produce el mismo resultado que correr de un
tirón. Su invariancia es además condición de la puerta G14: si las semillas
cambiaran, los primeros diez snapshots no podrían reproducir v7.7-b.

ESTRUCTURA EN DRIVE
-------------------
    MyDrive/mtps_c/
        ckpt_c_h2/    principal_00.pkl ... principal_59.pkl
        ckpt_c_h4/    principal_00.pkl ... principal_59.pkl
        partial_c_h2.pkl    partial_c_h4.pkl

USO
---
    from mtps_c_v7_7c_runner import run_stage_c
    run_stage_c(H=2)      # punto de control intermedio
    # verificar G14 antes de continuar:
    #   from mtps_c_v7_7c_consolidate import verify_g14
    #   verify_g14(H=2)
    run_stage_c(H=4)      # variable operativa

Reanudar tras una desconexión: volver a ejecutar exactamente lo mismo.
================================================================================
"""

import hashlib
import json
import os
import pickle
import time

import numpy as np

import mtps_c_v7_7c_common as _C

from mtps_c_v7_7c_common import (
    H_VALUES, H_OPERATIVA, H_CONTROL_INTERMEDIO,
    M_FIXED, N_ENSEMBLE_FIXED, MODE_PRINCIPAL,
    P_LEVELS, P_MAIN, P_HIGH,
    GateFailure,
    generate_mixed_dataset,
    gate_g15, gate_g16,
    seed_ensemble_for, seed_gt_base_for,
    n_calib_for_M, n_snapshots_for_H, expected_coverage_for_M,
    evaluate_config_v77c, aggregate_results_v77c,
    seed_ensemble_for as _seed_ens_v, seed_gt_base_for as _seed_gt_v,
    provenance_block, VE_UMBRAL,
)

DRIVE_BASE = '/content/drive/MyDrive/mtps_c'


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


# ==============================================================================
# CHECKPOINT DENTRO DE LA CONFIGURACIÓN — Enmienda 03 al Acta 02
# ==============================================================================
# Las configuraciones de la familia encuentro_violento requieren más horas
# continuas de las que la plataforma sostiene, y el checkpoint por
# configuración las vuelve todo o nada: una caída a mitad de camino pierde el
# intento entero. Se intercepta la función de integración con una caché en
# disco indexada por el SHA-256 de sus entradas. El integrador es determinista,
# de modo que devolver el resultado guardado es idéntico por construcción a
# recomputarlo. evaluate_config_v77c no se modifica: resuelve
# integrate_with_snapshots en el espacio de nombres de su módulo en tiempo de
# llamada, y el runner sustituye esa referencia solo durante la configuración.
#
# La familia se identifica por el dato de costo de v7.7-b, donde concentró el
# 84,8 % del cómputo, y no por una elección hecha tras observar esta corrida.
FAMILIAS_CKPT_INTRA = ('encuentro_violento',)

_INTEGRADOR_ORIGINAL = _C.integrate_with_snapshots

# La clave incluye el hash del módulo que define el integrador. rtol, atol y
# max_step están escritos dentro de la función congelada y no son argumentos,
# de modo que sin este componente la caché devolvería resultados obsoletos si el
# integrador cambiara y la carpeta de caché se reutilizara. La versión de la
# caché cubre cualquier cambio futuro en el formato de la clave.
import mtps_c_v7_7a_common as _A
with open(_A.__file__, 'rb') as _f:
    _HASH_INTEGRADOR = hashlib.sha256(_f.read()).hexdigest()
_VERSION_CACHE = 'enmienda-03-v1'


def _clave_integracion(masses, state_0, T_max, n_snapshots):
    h = hashlib.sha256()
    h.update(_VERSION_CACHE.encode())
    h.update(_HASH_INTEGRADOR.encode())
    h.update(np.asarray(masses, dtype=np.float64).tobytes())
    h.update(np.asarray(state_0, dtype=np.float64).tobytes())
    h.update(np.float64(T_max).tobytes())
    h.update(np.int64(n_snapshots).tobytes())
    return h.hexdigest()


def _integrador_cacheado(cache_dir, etiqueta, total, cada=5):
    os.makedirs(cache_dir, exist_ok=True)
    estado = {'n': 0, 'hits': 0}

    def integrador(masses, state_0, T_max, n_snapshots=10):
        clave = _clave_integracion(masses, state_0, T_max, n_snapshots)
        ruta = os.path.join(cache_dir, clave + '.pkl')
        previo = _load_ckpt(ruta)
        if previo is not None:
            out = previo
            estado['hits'] += 1
        else:
            out = _INTEGRADOR_ORIGINAL(masses, state_0, T_max, n_snapshots=n_snapshots)
            _atomic_dump(out, ruta)
        estado['n'] += 1
        if estado['n'] % cada == 0 or estado['n'] == total:
            print(f"      {time.strftime('%H:%M:%S')} [{etiqueta}] "
                  f"integración {estado['n']}/{total} "
                  f"({estado['hits']} desde caché)", flush=True)
        return out

    integrador.estado = estado
    return integrador


def _evaluar(config, seed_ens, seed_gt, H, cache_base):
    """Evalúa una configuración, con checkpoint intra-configuración si su
    familia lo requiere. El cómputo es el de evaluate_config_v77c sin cambio."""
    familia = config.get('regime_or_subtype')
    if familia not in FAMILIAS_CKPT_INTRA:
        return evaluate_config_v77c(config, seed_ens, seed_gt, H,
                                    n_ensemble=N_ENSEMBLE_FIXED, M=M_FIXED)
    cache_dir = os.path.join(cache_base, config['config_id'])
    total = N_ENSEMBLE_FIXED + M_FIXED
    integ = _integrador_cacheado(cache_dir, config['config_id'], total, cada=1)
    print(f"    checkpoint intra-configuración activo ({familia}): {cache_dir}")
    _C.integrate_with_snapshots = integ
    try:
        return evaluate_config_v77c(config, seed_ens, seed_gt, H,
                                    n_ensemble=N_ENSEMBLE_FIXED, M=M_FIXED)
    finally:
        _C.integrate_with_snapshots = _INTEGRADOR_ORIGINAL
        print(f"    integraciones: {integ.estado['n']}, "
              f"desde caché: {integ.estado['hits']}")


def run_stage_c(H, drive_base=DRIVE_BASE, report_every=1):
    """Ejecuta el stage completo para un valor de H. Reanudable."""
    if H not in H_VALUES:
        raise ValueError(f"H={H} no está en el prerregistro {H_VALUES}")

    ckpt_dir = os.path.join(drive_base, f'ckpt_c_h{H}')
    out_path = os.path.join(drive_base, f'partial_c_h{H}.pkl')
    os.makedirs(ckpt_dir, exist_ok=True)

    n_snaps = n_snapshots_for_H(H)
    rol = ('variable operativa' if H == H_OPERATIVA
           else 'punto de control intermedio')

    print("=" * 80)
    print(f"MTPS-C v7.7-c — STAGE H = {H}   ({rol})")
    print("=" * 80)
    print("\nPrerregistro: Hoja de Ruta v7.7-c, Borrador 5")
    print("Acta de Congelamiento 02. Método congelado; única variable: el horizonte.")
    print(f"M = {M_FIXED} fijo, N = {N_ENSEMBLE_FIXED} fijo, "
          f"n_calib = {n_calib_for_M(M_FIXED)}")
    print(f"Horizonte: {H} x T_adaptive   Snapshots: {n_snaps}   "
          f"Espaciado: T_adaptive/10 (invariante en H)")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Salida:      {out_path}\n")

    # ------------------------------------------------- ORDEN DE PUERTAS (C8)
    # El brazo operativo no arranca sin que G14 haya pasado sobre el punto de
    # control intermedio. La condición C8 del Acta 02 lo exige, y aquí queda
    # forzada por el código en lugar de depender del protocolo manual.
    if H == H_OPERATIVA:
        marcador = os.path.join(drive_base,
                                f'g14_ok_h{H_CONTROL_INTERMEDIO}.json')
        if not os.path.exists(marcador):
            raise GateFailure(
                f"C8: no se puede lanzar H={H_OPERATIVA} sin que G14 haya pasado "
                f"sobre H={H_CONTROL_INTERMEDIO}. Falta {os.path.basename(marcador)}. "
                f"Ejecutar antes:  from mtps_c_v7_7c_consolidate import verify_g14; "
                f"verify_g14(H={H_CONTROL_INTERMEDIO})")
        with open(marcador, encoding='utf-8') as f:
            mk = json.load(f)
        print(f"C8 OK — G14 pasada sobre H={H_CONTROL_INTERMEDIO} "
              f"({mk.get('celdas_comparadas')} celdas; "
              f"cobertura {mk.get('max_dif_cobertura', 0.0):.3e}, "
              f"tau {mk.get('max_tau_en_unidades_de_tol', 0.0):.2f}x su tolerancia)\n")

    dataset = generate_mixed_dataset()

    # ---------------------------------------------------------------- PUERTAS
    print("PUERTAS PREVIAS AL CÓMPUTO")
    gate_g15(dataset, context=f"H={H}", abort_dir=drive_base)
    gate_g16(abort_dir=drive_base)
    cov_esp = expected_coverage_for_M(M_FIXED)
    print(f"  Coberturas esperadas: "
          f"{ {p: round(100*cov_esp[p], 2) for p in P_LEVELS} }")
    print(f"  Criterio primario C12.a: varianza explicada por base fija de 2 "
          f"componentes >= {VE_UMBRAL}\n")

    seeds_ens = [seed_ensemble_for(i) for i in range(len(dataset))]
    seeds_gt = [seed_gt_base_for(i, M_FIXED) for i in range(len(dataset))]

    print(f"Dataset mixto: {len(dataset)} configs")
    counts = {}
    for c in dataset:
        counts[c['subset']] = counts.get(c['subset'], 0) + 1
    for s, n in counts.items():
        print(f"  {s}: {n} configs")
    print("Brazo en este stage: principal (sin control, conforme a D4)\n")

    # ---------------------------------------------------------------- CÓMPUTO
    print("=" * 80)
    print(f"EJECUTANDO PRINCIPAL  H={H}  {n_snaps} snapshots")
    print("=" * 80)

    all_results = [None] * len(dataset)
    pending = []
    for i in range(len(dataset)):
        ck = os.path.join(ckpt_dir, f'principal_{i:02d}.pkl')
        payload = _load_ckpt(ck)
        if payload is not None:
            r = payload.get('result')
            cid = r.get('config_id') if isinstance(r, dict) else None
            h_ck = r.get('H') if isinstance(r, dict) else None
            if cid != dataset[i]['config_id'] or (h_ck is not None and h_ck != H):
                print(f"    checkpoint {i:02d} no corresponde a "
                      f"{dataset[i]['config_id']} / H={H} "
                      f"(trae {cid} / H={h_ck}), se recalcula")
                payload = None
        if payload is not None:
            all_results[i] = payload['result']
        else:
            pending.append(i)

    done = len(dataset) - len(pending)
    if done:
        print(f"  Reanudando: {done}/{len(dataset)} configs ya en checkpoint")
    if not pending:
        print(f"  Stage completo desde checkpoints, sin recomputar")

    total_compute_seconds = sum(
        (_load_ckpt(os.path.join(ckpt_dir, f'principal_{i:02d}.pkl')) or {}).get('seconds', 0.0)
        for i in range(len(dataset)) if i not in pending
    )

    t0 = time.time()
    for count, i in enumerate(pending, start=1):
        t_cfg = time.time()
        result = _evaluar(dataset[i], seeds_ens[i], seeds_gt[i], H,
                          os.path.join(ckpt_dir, 'intra'))
        secs = time.time() - t_cfg
        total_compute_seconds += secs

        all_results[i] = result
        _atomic_dump({'result': result, 'seconds': secs, 'H': H},
                     os.path.join(ckpt_dir, f'principal_{i:02d}.pkl'))

        if count % report_every == 0 or count == len(pending):
            el = time.time() - t0
            eta = el / count * (len(pending) - count)
            print(f"  ...{done + count}/{len(dataset)} "
                  f"[cfg {i:02d} {dataset[i]['subset']}] "
                  f"{secs:.0f}s | acum {el:.0f}s | ETA {eta:.0f}s", flush=True)

    print(f"  PRINCIPAL completado: {len(dataset)} configs")
    agg = aggregate_results_v77c(all_results, H)

    print(f"\n  Tiempo de cómputo acumulado: {total_compute_seconds/60:.1f} min "
          f"({total_compute_seconds/3600:.2f} h)")

    # ---------------------------------------------------------------- RESUMEN
    if agg:
        print(f"\n  H={H}  ({agg['n_snapshots']} snapshots):")
        print(f"    C12.a estructural: {'OK' if agg['c12a'] else 'INCUMPLE'}"
              + ("" if agg['c12a'] else
                 f" desde el snapshot {agg['primer_snapshot_incumple_c12a']}"))
        print(f"    C12.c calibración: {'OK' if agg['c12c'] else 'INCUMPLE'}"
              + ("" if agg['c12c'] else
                 f" desde el bloque {agg['primer_bloque_incumple_c12c']}"))
        hitos = [s for s in (1, 10, 20, 40) if s <= agg['n_snapshots']]
        print(f"\n    {'snap':>5} {'VE base fija':>13} {'d_eff_raw':>10} "
              f"{'tau/radio':>11} {'tau':>11}")
        for s in hitos:
            print(f"    {s:>5} {agg['ve_por_snapshot'][s]:13.4f} "
                  f"{agg['d_eff_raw_por_snapshot'][s]:10.2f} "
                  f"{agg['tau_radio_por_snapshot'][s]:11.4f} "
                  f"{agg['tau_por_snapshot'][s]:11.5f}")
        print(f"\n    Bloques de calibración:")
        for b, d in agg['bloques_calibracion'].items():
            print(f"      {b:>7}: {d['n_levels_ok']}/5 niveles")
        print(f"    Configuraciones fallidas: {agg['n_failed']}")

    payload = {
        'H': int(H),
        'n_snapshots': n_snaps,
        'M': M_FIXED,
        'n_ensemble': N_ENSEMBLE_FIXED,
        'principal': agg,
        'elapsed_seconds': total_compute_seconds,
        'procedencia': provenance_block(dataset, H),
    }
    _atomic_dump(payload, out_path)

    print(f"\n  Resultado guardado: {out_path}")
    print(f"  Procedencia: {json.dumps(payload['procedencia'], ensure_ascii=False)}")
    print(f"\n{'=' * 80}")
    print(f"Stage H={H} completado.")
    print("=" * 80)
    if H == H_CONTROL_INTERMEDIO:
        print("\n  SIGUIENTE: verificar la puerta G14 antes de correr H=4.")
        print("  from mtps_c_v7_7c_consolidate import verify_g14")
        print("  verify_g14(H=2)")
    else:
        print("\n  SIGUIENTE: consolidar.")
        print("  !cd /content/drive/MyDrive/mtps_c && python mtps_c_v7_7c_consolidate.py")
    return payload


# ==============================================================================
# VERIFICACIÓN EMPÍRICA DE LA ENMIENDA 03
# ==============================================================================

def _aplanar(r):
    out = {}
    for sn in r["snapshots"]:
        k = sn["k"]
        out[("ve", k)] = sn["ve_base_fija"]
        out[("radio", k)] = sn["radio_env_95"]
        out[("draw", k)] = sn["d_eff_raw"]
        for sd in sn["per_split"]:
            for p, v in sd["cov_rates"].items():
                out[("cov", k, sd["split_id"], p)] = v
            for p, v in sd["tau_p"].items():
                out[("tau", k, sd["split_id"], p)] = v
    return out


def _igual(a, b):
    fa, fb = _aplanar(a), _aplanar(b)
    if set(fa) != set(fb):
        return False, "conjuntos de claves distintos"
    dif = [k for k in fa if fa[k] != fb[k]]
    return (not dif), f"{len(dif)} de {len(fa)} valores difieren"


def verificar_ckpt_intra(i, H=4, drive_base=DRIVE_BASE):
    """Prueba la caché de integraciones sobre una configuración ya computada.

    Tres casos, todos contra el checkpoint existente y bit a bit:
      1. caché vacía — cada integración se computa y se guarda;
      2. caché completa — cada integración sale de disco;
      3. caché parcial — se borra la mitad y se reanuda.
    """
    import shutil
    dataset = generate_mixed_dataset()
    cfg = dataset[i]
    ref_path = os.path.join(drive_base, f'ckpt_c_h{H}', f'principal_{i:02d}.pkl')
    ref = _load_ckpt(ref_path)
    if ref is None:
        raise FileNotFoundError(f"no existe el checkpoint de referencia {ref_path}")
    ref = ref['result']

    base = os.path.join(drive_base, '_verif_enm03')
    if os.path.exists(base):
        shutil.rmtree(base)

    def correr(etiqueta):
        cache_dir = os.path.join(base, cfg['config_id'])
        integ = _integrador_cacheado(cache_dir, etiqueta, N_ENSEMBLE_FIXED + M_FIXED, cada=30)
        _C.integrate_with_snapshots = integ
        try:
            out = evaluate_config_v77c(cfg, _seed_ens_v(i),
                                       _seed_gt_v(i, M_FIXED), H,
                                       n_ensemble=N_ENSEMBLE_FIXED, M=M_FIXED)
        finally:
            _C.integrate_with_snapshots = _INTEGRADOR_ORIGINAL
        return out, integ.estado['hits'], cache_dir

    print(f"Verificación de la Enmienda 03 sobre {cfg['config_id']} (índice {i}), H={H}\n")
    resultados = []

    out, hits, cache_dir = correr("caché vacía")
    ok, msg = _igual(ref, out)
    resultados.append(ok)
    print(f"  1. caché vacía:    {hits} desde caché — {'IDÉNTICO' if ok else 'DIFIERE'} ({msg})")

    out, hits, _ = correr("caché completa")
    ok, msg = _igual(ref, out)
    resultados.append(ok)
    print(f"  2. caché completa: {hits} desde caché — {'IDÉNTICO' if ok else 'DIFIERE'} ({msg})")

    archivos = sorted(os.listdir(cache_dir))
    for a in archivos[::2]:
        os.remove(os.path.join(cache_dir, a))
    out, hits, _ = correr("caché parcial")
    ok, msg = _igual(ref, out)
    resultados.append(ok)
    print(f"  3. caché parcial:  {hits} desde caché — {'IDÉNTICO' if ok else 'DIFIERE'} ({msg})")

    shutil.rmtree(base)
    todo = all(resultados)
    print("\n" + ("ENMIENDA 03 VERIFICADA — la caché no altera el cómputo"
                  if todo else "FALLO — la caché altera el cómputo, NO usar"))
    return todo
