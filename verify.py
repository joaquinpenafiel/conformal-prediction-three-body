#!/usr/bin/env python3
"""
================================================================================
MTPS-C — verificador de la cadena de integridad
================================================================================

Comprueba que el paquete publicado es el que las actas del programa declaran.
No requiere dependencias: solo la biblioteca estandar de Python 3.8 o superior.

    python verify.py

Hace tres cosas, en este orden:

  1. MANIFIESTO. Recalcula el SHA-256 de cada archivo listado en
     MANIFEST.sha256 y lo compara con el valor registrado. Cualquier
     diferencia significa que el archivo no es el que el programa congelo.

  2. DATASET CANONICO. Regenera el hash canonico del conjunto de 60
     configuraciones a partir de data/mtps_c_v7_7a_configs.json, usando la
     serializacion estable definida en la seccion 5 del Acta de Congelamiento
     01, y lo compara con el valor que las tres etapas verificaron en cada
     corrida mediante las puertas G13 y G15. Es una comprobacion distinta de
     la anterior: el manifiesto verifica el archivo, esto verifica que su
     CONTENIDO produce el hash que los experimentos usaron.

  4. ARCHIVOS NO DECLARADOS. Lista, sin hacer fallar la verificacion, los
     archivos presentes que el manifiesto no declara y que no son
     infraestructura conocida del repositorio.

  3. COHERENCIA INTERNA. Comprueba que los hashes que los artefactos de diff
     declaran en su encabezado coinciden con los del manifiesto, de modo que
     la cadena de derivacion v7.7-a -> v7.7-b -> v7.7-c quede verificada y no
     solo afirmada.

Codigo de salida 0 si todo verifica, 1 si algo falla.
================================================================================
"""

import hashlib
import json
import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
MANIFIESTO = os.path.join(RAIZ, "MANIFEST.sha256")

# Registrado en la condicion C4 del Acta de Congelamiento 01 y verificado por
# las puertas G13 (v7.7-b) y G15 (v7.7-c) al inicio de cada par de la corrida.
DATASET_SHA256 = ("b2b0937a069d0170d9379ae08919067b"
                  "9fff3e5ffc967a9f580425a80efedf47")

# Cadena de derivacion declarada en las condiciones C6 de ambas actas.
CADENA_DIFF = [
    ("src/v7_7b/diff_v77a_to_v77b_common.txt",
     "src/v7_7a/mtps_c_v7_7a_common.py",
     "src/v7_7b/mtps_c_v7_7b_common.py"),
    ("src/v7_7c/diff_v77b_to_v77c_common.txt",
     "src/v7_7b/mtps_c_v7_7b_common.py",
     "src/v7_7c/mtps_c_v7_7c_common.py"),
]

VERDE, ROJO, AMARILLO, FIN = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    VERDE = ROJO = AMARILLO = FIN = ""


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def titulo(texto):
    print()
    print("=" * 74)
    print(texto)
    print("=" * 74)


# ==============================================================================
# 1. Manifiesto
# ==============================================================================

def verificar_manifiesto():
    titulo("1. MANIFIESTO — integridad de cada archivo del paquete")
    if not os.path.exists(MANIFIESTO):
        print(f"{ROJO}  no se encontro MANIFEST.sha256{FIN}")
        return False, {}

    entradas = []
    with open(MANIFIESTO, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            esperado, _, ruta = linea.partition("  ")
            entradas.append((esperado.strip(), ruta.strip()))

    ok, faltan, difieren, hashes = 0, [], [], {}
    for esperado, ruta in entradas:
        completa = os.path.join(RAIZ, ruta)
        if not os.path.exists(completa):
            faltan.append(ruta)
            continue
        obtenido = sha256(completa)
        hashes[ruta] = obtenido
        if obtenido == esperado:
            ok += 1
        else:
            difieren.append((ruta, esperado, obtenido))

    print(f"  archivos en el manifiesto : {len(entradas)}")
    print(f"  verificados               : {ok}")
    if faltan:
        print(f"{ROJO}  ausentes                  : {len(faltan)}{FIN}")
        for r in faltan:
            print(f"      {r}")
    if difieren:
        print(f"{ROJO}  con hash distinto         : {len(difieren)}{FIN}")
        for r, e, o in difieren:
            print(f"      {r}")
            print(f"        esperado  {e}")
            print(f"        obtenido  {o}")

    bien = not faltan and not difieren
    print()
    print(f"  {VERDE}MANIFIESTO VERIFICADO{FIN}" if bien
          else f"  {ROJO}MANIFIESTO NO VERIFICADO{FIN}")
    return bien, hashes


# ==============================================================================
# 2. Dataset canonico
# ==============================================================================

def verificar_dataset():
    titulo("2. DATASET CANONICO — el contenido produce el hash de las corridas")
    ruta = os.path.join(RAIZ, "data", "mtps_c_v7_7a_configs.json")
    if not os.path.exists(ruta):
        print(f"{ROJO}  no se encontro data/mtps_c_v7_7a_configs.json{FIN}")
        return False

    with open(ruta, encoding="utf-8") as f:
        contenido = json.load(f)
    configs = contenido["configs"] if isinstance(contenido, dict) else contenido

    # Serializacion estable, seccion 5 del Acta de Congelamiento 01.
    lineas = [
        f"{c['config_id']}|{c['subset']}|{c['regime_or_subtype']}|{c['source_seed']}"
        for c in configs
    ]
    obtenido = hashlib.sha256(("\n".join(lineas) + "\n").encode("utf-8")).hexdigest()

    print(f"  configuraciones : {len(configs)}")
    print(f"  esperado        : {DATASET_SHA256}")
    print(f"  obtenido        : {obtenido}")
    bien = obtenido == DATASET_SHA256
    print()
    print(f"  {VERDE}DATASET CANONICO VERIFICADO{FIN}" if bien
          else f"  {ROJO}DATASET NO CANONICO{FIN}")
    return bien


# ==============================================================================
# 3. Cadena de derivacion
# ==============================================================================

def verificar_cadena(hashes):
    titulo("3. CADENA DE DERIVACION — los diff declaran lo que el paquete contiene")
    todo_bien = True
    for ruta_diff, ruta_base, ruta_nuevo in CADENA_DIFF:
        completa = os.path.join(RAIZ, ruta_diff)
        if not os.path.exists(completa):
            print(f"{ROJO}  ausente: {ruta_diff}{FIN}")
            todo_bien = False
            continue

        declarados = []
        with open(completa, encoding="utf-8", errors="replace") as f:
            for linea in f:
                if not linea.startswith("#"):
                    break
                partes = linea.split()
                if "SHA-256" in linea and partes:
                    declarados.append(partes[-1])

        if len(declarados) < 2:
            print(f"{AMARILLO}  {ruta_diff}: el encabezado no declara dos hashes{FIN}")
            todo_bien = False
            continue

        d_base, d_nuevo = declarados[0], declarados[1]
        m_base = hashes.get(ruta_base, "")
        m_nuevo = hashes.get(ruta_nuevo, "")

        print(f"  {os.path.basename(ruta_diff)}")
        for etiqueta, declarado, manifiesto, ruta in (
            ("base ", d_base, m_base, ruta_base),
            ("nuevo", d_nuevo, m_nuevo, ruta_nuevo),
        ):
            coincide = declarado == manifiesto and manifiesto != ""
            marca = f"{VERDE}OK{FIN}" if coincide else f"{ROJO}DIFIERE{FIN}"
            print(f"      {etiqueta}  {os.path.basename(ruta):<28} {marca}")
            if not coincide:
                print(f"             declarado en el diff : {declarado}")
                print(f"             en el manifiesto     : {manifiesto or '(ausente)'}")
                todo_bien = False

    print()
    print(f"  {VERDE}CADENA DE DERIVACION VERIFICADA{FIN}" if todo_bien
          else f"  {ROJO}CADENA DE DERIVACION NO VERIFICADA{FIN}")
    return todo_bien


# ==============================================================================

# ==============================================================================
# 4. Archivos no declarados (informativo)
# ==============================================================================

# Infraestructura del repositorio: no forma parte del paquete anclado por las
# actas del programa, y por eso no figura en el manifiesto.
NO_MANIFESTADOS_ESPERADOS = {
    "README.md", "LICENSE", "LICENSE-DOCS", "CITATION.cff",
    "MANIFEST.sha256", "verify.py", "requirements.txt",
    ".gitignore", ".gitattributes",
    "docs/INDEX.md",
    "demo/run_demo.py", "demo/README.md",
    ".github/workflows/verify.yml",
}


def revisar_no_declarados(manifestados):
    """Lista archivos presentes que el manifiesto no declara.

    No hace fallar la verificacion: un repositorio puede incorporar
    infraestructura legitima —un workflow, un archivo de dependencias— sin que
    eso altere el paquete que las actas anclan. La comprobacion existe para que
    la diferencia entre "los archivos del manifiesto verifican" y "el paquete
    contiene exactamente lo declarado" quede a la vista en lugar de suponerse.
    """
    titulo("4. ARCHIVOS NO DECLARADOS — informativo, no hace fallar")
    presentes = set()
    for base, dirs, archivos in os.walk(RAIZ):
        dirs[:] = [d for d in dirs
                   if d not in (".git", "__pycache__", "outputs", ".ipynb_checkpoints")]
        for a in archivos:
            rel = os.path.relpath(os.path.join(base, a), RAIZ).replace(os.sep, "/")
            presentes.add(rel)

    extra = sorted(presentes - set(manifestados) - NO_MANIFESTADOS_ESPERADOS)
    print(f"  archivos en el arbol        : {len(presentes)}")
    print(f"  declarados en el manifiesto : {len(manifestados)}")
    print(f"  infraestructura conocida    : {len(NO_MANIFESTADOS_ESPERADOS)}")
    if extra:
        print(f"{AMARILLO}  no declarados ni conocidos  : {len(extra)}{FIN}")
        for r in extra[:20]:
            print(f"      {r}")
        if len(extra) > 20:
            print(f"      ... y {len(extra) - 20} mas")
    else:
        print(f"  {VERDE}sin archivos fuera de lo declarado{FIN}")
    return extra


def main():
    print()
    print("MTPS-C — verificacion de integridad del paquete publicado")
    print("Motor de Trayectorias Probabilisticas para Sistemas Caoticos")

    ok_manifiesto, hashes = verificar_manifiesto()
    ok_dataset = verificar_dataset()
    ok_cadena = verificar_cadena(hashes)
    extra = revisar_no_declarados(hashes.keys())

    titulo("RESULTADO")
    for etiqueta, estado in (("Manifiesto", ok_manifiesto),
                             ("Dataset canonico", ok_dataset),
                             ("Cadena de derivacion", ok_cadena)):
        marca = f"{VERDE}OK{FIN}" if estado else f"{ROJO}FALLA{FIN}"
        print(f"  {etiqueta:<24} {marca}")
    print(f"  {'Archivos no declarados':<24} "
          + (f"{VERDE}ninguno{FIN}" if not extra
             else f"{AMARILLO}{len(extra)} (informativo){FIN}"))

    todo = ok_manifiesto and ok_dataset and ok_cadena
    print()
    if todo:
        print(f"  {VERDE}El paquete es el que las actas del programa declaran.{FIN}")
    else:
        print(f"  {ROJO}El paquete NO coincide con lo declarado. Revisar arriba.{FIN}")
    print()
    return 0 if todo else 1


if __name__ == "__main__":
    sys.exit(main())
