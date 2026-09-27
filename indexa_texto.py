import csv
import requests
import psycopg2


# ==========================================
# CONFIGURACIÓN
# ==========================================

CSV_FILE = "datos.csv"

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
# GENERAR EMBEDDING
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
# CONECTAR POSTGRESQL
# ==========================================

def conectar():

    return psycopg2.connect(**DB_CONFIG)


# ==========================================
# CREAR TABLA
# ==========================================

def crear_tabla():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS marcas_embeddings (
            id SERIAL PRIMARY KEY,
            descripcion TEXT,
            url_imagen TEXT,
            embedding vector(1024)
        );
    """)

    conn.commit()

    cursor.close()
    conn.close()


# ==========================================
# INDEXAR CSV
# ==========================================

def indexar():

    crear_tabla()

    conn = conectar()
    cursor = conn.cursor()

    with open(
        CSV_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as archivo:

        reader = csv.DictReader(archivo)

        for i, fila in enumerate(reader, start=1):

            descripcion = fila["descripcion"].strip()
            url_imagen = fila["url_imagen"].strip()

            print(f"{i}. {descripcion}")

            # Generar embedding SOLO de la descripción
            embedding = generar_embedding(descripcion)

            vector = "[" + ",".join(
                map(str, embedding)
            ) + "]"

            cursor.execute("""
                INSERT INTO marcas_embeddings
                (
                    descripcion,
                    url_imagen,
                    embedding
                )
                VALUES (%s, %s, %s::vector)
            """, (
                descripcion,
                url_imagen,
                vector
            ))

            print(
                f"   Embedding generado: {len(embedding)} dimensiones"
            )

    conn.commit()

    cursor.close()
    conn.close()

    print("\n================================")
    print("INDEXACIÓN COMPLETADA")
    print("================================")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    indexar()