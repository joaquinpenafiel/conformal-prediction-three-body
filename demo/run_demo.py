#!/usr/bin/env python3
"""
================================================================================
MTPS-C — corrida corta de demostracion
================================================================================

Ejercita el pipeline completo de v7.7-c sobre seis configuraciones en lugar de
sesenta, con el horizonte del punto de control intermedio (H = 2, veinte
snapshots). Tarda minutos en vez de dias.

    python demo/run_demo.py

NO reproduce el experimento. Su proposito es que cualquiera pueda comprobar,
en una sola sesion, que la maquinaria del programa funciona:

  1. Las puertas de entorno G15 y G16 se ejecutan antes de computar nada.
  2. El metodo congelado corre sobre configuraciones reales del dataset.
  3. Los criterios se evaluan y el veredicto sale de una funcion mecanica.
  4. Los resultados se contrastan contra los publicados en results/.

LAS SEIS CONFIGURACIONES
------------------------
Dos por subconjunto, elegidas por ser las de menor costo dentro del suyo segun
los tiempos medidos de la corrida completa, que estan publicados en
results/v7_7c/tiempos_por_config_v7_7c.csv. La familia encuentro_violento queda
fuera a proposito: concentra entre dos tercios y cuatro quintos del computo
total del programa y una sola de sus configuraciones requiere horas.

SOBRE EL CONTRASTE
------------------
El paso 4 compara contra los valores publicados y reporta la diferencia maxima.
Una diferencia del orden del ruido numerico del integrador es ESPERABLE entre
versiones distintas de scipy o numpy, y no significa que algo este mal: es la
misma leccion que llevo a la Enmienda 02 del programa, donde una puerta abortó
por exigir a una magnitud continua mas precision de la que el metodo declara
garantizar. El contraste informa el desvio; no lo declara fracaso.

La cobertura empirica, en cambio, es discreta y deberia reproducirse de forma
exacta. Si difiere, hay algo de fondo que investigar.

Requiere numpy, scipy, scikit-learn y matplotlib. El ultimo lo arrastra el
modulo congelado de v7.7-a, que genera una figura de resultados; esta demo
no dibuja nada, pero el import debe resolverse.
================================================================================
"""

import csv
import os
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src", "v7_7c"))
sys.path.insert(0, os.path.join(RAIZ, "src", "v7_7b"))
sys.path.insert(0, os.path.join(RAIZ, "src", "v7_7a"))

H_DEMO = 2
PUBLICADO = os.path.join(RAIZ, "results", "v7_7c", "mtps_c_v7_7c_table_cells.csv")

# Dos por subconjunto, las de menor costo del suyo.
INDICES = [5, 18, 39, 37, 42, 43]

# Tolerancia del contraste: el rtol del integrador congelado. No es un umbral
# de aprobacion — solo separa "ruido numerico esperable" de "algo que revisar".
RTOL_INTEGRADOR = 1e-8


def titulo(t):
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


def main():
    print()
    print("MTPS-C — corrida corta de demostracion")
    print(f"Seis configuraciones, H = {H_DEMO} ({10 * H_DEMO} snapshots)")

    try:
        from mtps_c_v7_7c_common import (
            generate_mixed_dataset, gate_g15, gate_g16,
            evaluate_config_v77c, aggregate_results_v77c,
            seed_ensemble_for, seed_gt_base_for,
            M_FIXED, N_ENSEMBLE_FIXED, VE_UMBRAL, VE_BASE_FIJA,
            n_snapshots_for_H, P_LEVELS, ERRCAL_MAX_C11C,
        )
    except ImportError as e:
        print(f"\n  falta una dependencia: {e}")
        print("  instalar con:  pip install numpy scipy scikit-learn matplotlib")
        return 1

    # ---------------------------------------------------------------- puertas
    titulo("1. PUERTAS DE ENTORNO")
    print("  Las mismas que el experimento ejecuta antes de computar nada.")
    print()
    dataset = generate_mixed_dataset()
    gate_g15(dataset, context="demo")
    gate_g16()
    print()
    print(f"  Criterio primario C12.a: varianza explicada por una base fija de")
    print(f"  {VE_BASE_FIJA} componentes >= {VE_UMBRAL}")

    # ---------------------------------------------------------------- computo
    titulo("2. COMPUTO — metodo congelado, seis configuraciones")
    resultados, t0 = [], time.time()
    for n, i in enumerate(INDICES, start=1):
        cfg = dataset[i]
        t1 = time.time()
        r = evaluate_config_v77c(cfg, seed_ensemble_for(i),
                                 seed_gt_base_for(i, M_FIXED), H_DEMO,
                                 n_ensemble=N_ENSEMBLE_FIXED, M=M_FIXED)
        resultados.append(r)
        print(f"  {n}/{len(INDICES)}  idx={i:<3} {cfg['subset']:<16} "
              f"{cfg['config_id']:<32} {time.time() - t1:6.1f} s", flush=True)
    print(f"\n  total: {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------- agregacion
    titulo("3. CRITERIOS Y VEREDICTO")
    agg = aggregate_results_v77c(resultados, H_DEMO)
    if agg is None:
        print("  no se produjo agregado")
        return 1

    n_snaps = n_snapshots_for_H(H_DEMO)
    print(f"  {'snap':>5} {'VE base fija':>13} {'d_eff_raw':>10} "
          f"{'tau/radio':>11} {'C12.a':>7}")
    print("  " + "-" * 50)
    for s in range(1, n_snaps + 1):
        ve = agg["ve_por_snapshot"][s]
        print(f"  {s:>5} {ve:13.4f} {agg['d_eff_raw_por_snapshot'][s]:10.2f} "
              f"{agg['tau_radio_por_snapshot'][s]:11.4f} "
              f"{'OK' if ve >= VE_UMBRAL else 'NO':>7}")

    print()
    print(f"  C12.a estructural : {'se cumple' if agg['c12a'] else 'incumple'}"
          + ("" if agg["c12a"]
             else f" desde el snapshot {agg['primer_snapshot_incumple_c12a']}"))
    print(f"  C12.c calibracion : {'se cumple' if agg['c12c'] else 'incumple'}")
    for b, d in agg["bloques_calibracion"].items():
        det = "  ".join(f"Cov({p})={100 * d['cov'][p]:.2f}%" for p in P_LEVELS)
        print(f"      snapshots {b:>6}: {d['n_levels_ok']}/5   {det}")
    print()
    print(f"  Configuraciones fallidas: {agg['n_failed']}")
    print()
    print("  Sobre seis configuraciones, el veredicto de escenario no es")
    print("  comparable con el de la corrida completa: el prerregistro lo define")
    print("  sobre las sesenta. Los criterios se muestran para exhibir como se")
    print("  evaluan, no como resultado.")

    # -------------------------------------------------------------- contraste
    titulo("4. CONTRASTE CONTRA LOS RESULTADOS PUBLICADOS")
    if not os.path.exists(PUBLICADO):
        print(f"  no se encontro {os.path.relpath(PUBLICADO, RAIZ)}")
        print("  se omite el contraste")
        return 0

    pub = {}
    with open(PUBLICADO, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(float(r["H"])) != H_DEMO:
                continue
            pub[(r["config_id"], int(float(r["snapshot"])))] = r

    ids = {dataset[i]["config_id"] for i in INDICES}
    comparadas = 0
    max_cov, max_tau_rel, max_ve = 0.0, 0.0, 0.0
    peor_tau = None

    for c in agg["rows_per_cell"]:
        clave = (c["config_id"], c["snapshot"])
        if c["config_id"] not in ids or clave not in pub:
            continue
        p = pub[clave]
        comparadas += 1
        max_cov = max(max_cov, abs(c["cov_95_mean"] - float(p["cov_95_mean"])))
        max_ve = max(max_ve, abs(c["ve_base_fija"] - float(p["ve_base_fija"])))
        a, b = c["tau_95_mean"], float(p["tau_95_mean"])
        if max(abs(a), abs(b)) > 0:
            rel = abs(a - b) / max(abs(a), abs(b))
            if rel > max_tau_rel:
                max_tau_rel, peor_tau = rel, (clave, a, b)

    print(f"  celdas comparadas : {comparadas}")
    print()
    print(f"  cobertura empirica   dif. maxima absoluta : {max_cov:.3e}")
    print(f"  varianza explicada   dif. maxima absoluta : {max_ve:.3e}")
    print(f"  radio conformal      dif. maxima relativa : {max_tau_rel:.3e}")
    if peor_tau:
        (cid, s), a, b = peor_tau
        print(f"      peor caso: {cid} snapshot {s}")
        print(f"        esta corrida : {a:.12e}")
        print(f"        publicado    : {b:.12e}")

    print()
    if max_cov == 0.0:
        print("  La cobertura se reproduce de forma EXACTA. Es una magnitud")
        print("  discreta y por tanto robusta al ruido numerico.")
    else:
        print("  ATENCION: la cobertura difiere. Es discreta y deberia")
        print("  reproducirse exactamente; conviene investigar.")

    if max_tau_rel <= RTOL_INTEGRADOR:
        print(f"  El radio conformal coincide dentro del rtol que el integrador")
        print(f"  declara ({RTOL_INTEGRADOR:.0e}).")
    else:
        print(f"  El radio conformal difiere por encima del rtol del integrador")
        print(f"  ({RTOL_INTEGRADOR:.0e}). Es esperable con versiones distintas de")
        print(f"  scipy o numpy: se informa el desvio, no se declara fracaso.")

    titulo("FIN")
    print("  El pipeline corrio de extremo a extremo: puertas, computo,")
    print("  criterios y contraste. Para la cadena documental completa:")
    print("      python verify.py")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
