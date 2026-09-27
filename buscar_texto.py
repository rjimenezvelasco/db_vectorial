# ==========================================
# buscar.py
# Devuelve SOLO la respuesta más cercana
# ==========================================

import sys
import requests
import psycopg2


# ==========================================
# CONFIGURACIÓN
# ==========================================

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
MODEL = "mxbai-embed-large:latest"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "dbbrand",
    "user": "b",
    "password": "a"
}


# ==========================================
# GENERAR EMBEDDING DE LA CONSULTA
# ==========================================

def generar_embedding(texto):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "input": texto
        }
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"][0]


# ==========================================
# CONEXIÓN
# ==========================================

def conectar():
    return psycopg2.connect(**DB_CONFIG)


# ==========================================
# FORMATEAR VECTOR
# ==========================================

def vector_a_str(embedding):
    return "[" + ",".join(map(str, embedding)) + "]"


# ==========================================
# MEJOR RESULTADO — COSENO
# ==========================================

def mejor_coseno(cursor, vector):
    cursor.execute("""
        SELECT
            id,
            descripcion,
            url_imagen,
            embedding <=> %s::vector AS distancia
        FROM marcas_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT 1
    """, (vector, vector))
    return cursor.fetchone()


# ==========================================
# MEJOR RESULTADO — EUCLIDIANA (L2)
# ==========================================

def mejor_euclidiana(cursor, vector):
    cursor.execute("""
        SELECT
            id,
            descripcion,
            url_imagen,
            embedding <-> %s::vector AS distancia
        FROM marcas_embeddings
        ORDER BY embedding <-> %s::vector
        LIMIT 1
    """, (vector, vector))
    return cursor.fetchone()


# ==========================================
# PRESENTACIÓN
# ==========================================

def mostrar_mejor(titulo, fila, tipo):
    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)

    if fila is None:
        print("(sin resultados)")
        return

    id_, desc, url, score = fila

    if tipo == "coseno":
        metrica = f"similitud coseno = {1 - score:.4f}  (distancia = {score:.4f})"
    else:
        metrica = f"distancia L2     = {score:.4f}"

    print(f"ID        : {id_}")
    print(f"Descripción: {desc}")
    print(f"Imagen    : {url}")
    print(f"Métrica   : {metrica}")


# ==========================================
# MAIN
# ==========================================

def main():
    if len(sys.argv) > 1:
        consulta = " ".join(sys.argv[1:]).strip()
    else:
        consulta = input("Descripción a buscar: ").strip()

    if not consulta:
        print("Debes indicar una descripción.")
        return

    print(f"\nConsulta: {consulta}")
    print("Generando embedding...")

    embedding = generar_embedding(consulta)
    vector = vector_a_str(embedding)

    conn = conectar()
    cursor = conn.cursor()

    # ----- Mejor por COSENO -----
    fila_cos = mejor_coseno(cursor, vector)
    mostrar_mejor("MEJOR COINCIDENCIA — SIMILITUD COSENO", fila_cos, "coseno")

    # ----- Mejor por EUCLIDIANA -----
    fila_l2 = mejor_euclidiana(cursor, vector)
    mostrar_mejor("MEJOR COINCIDENCIA — DISTANCIA EUCLIDIANA (L2)", fila_l2, "euclidiana")

    cursor.close()
    conn.close()

def umain():
    if len(sys.argv) > 1:
        consulta = " ".join(sys.argv[1:]).strip()
    else:
        consulta = input("Descripción a buscar: ").strip()

    if not consulta:
        print("Debes indicar una descripción.")
        return

    print(f"\nConsulta: {consulta}")
    embedding = generar_embedding(consulta)
    vector = vector_a_str(embedding)

    conn = conectar()
    cursor = conn.cursor()

    fila = mejor_coseno(cursor, vector)

    if fila is None:
        print("No hay registros indexados.")
    else:
        id_, desc, url, distancia = fila
        print("\n" + "=" * 60)
        print("RESULTADO MÁS CERCANO")
        print("=" * 60)
        print(f"Descripción: {desc}")
        print(f"Imagen     : {url}")
        print(f"Similitud  : {1 - distancia:.4f}")

    cursor.close()
    conn.close()
    
if __name__ == "__main__":
    umain()