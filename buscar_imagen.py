# ==========================================
# buscar_imagen.py
# Dado un archivo de imagen local, encuentra la imagen
# más similar en la tabla marcas_embeddings.
# ==========================================

import sys
import io
import requests
import psycopg2
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch


# ==========================================
# CONFIGURACIÓN
# ==========================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "dbbrand",
    "user": "b",
    "password": "a"
}


# ==========================================
# CLIP
# ==========================================

print("Cargando CLIP...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval()
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)


def embedding_desde_archivo(ruta):
    imagen = Image.open(ruta).convert("RGB")
    inputs = clip_processor(images=imagen, return_tensors="pt")

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy().tolist()


def embedding_desde_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    imagen = Image.open(io.BytesIO(response.content)).convert("RGB")

    inputs = clip_processor(images=imagen, return_tensors="pt")

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy().tolist()


def vector_a_str(embedding):
    return "[" + ",".join(map(str, embedding)) + "]"


# ==========================================
# CONEXIÓN
# ==========================================

def conectar():
    return psycopg2.connect(**DB_CONFIG)


# ==========================================
# BÚSQUEDA POR IMAGEN
# ==========================================

def buscar_por_imagen(consulta_embedding, top_k=3):
    conn = conectar()
    cursor = conn.cursor()

    vector = vector_a_str(consulta_embedding)

    cursor.execute("""
        SELECT
            id,
            descripcion,
            url_imagen,
            embedding_imagen <=> %s::vector AS distancia
        FROM marcas_embeddings
        WHERE embedding_imagen IS NOT NULL
          AND (embedding_imagen <=> %s::vector) < 0.30
        ORDER BY embedding_imagen <=> %s::vector
        LIMIT %s
    """, (vector, vector, vector, top_k))

    resultados = cursor.fetchall()
    cursor.close()
    conn.close()

    return resultados


# ==========================================
# PRESENTACIÓN
# ==========================================

def mostrar_resultados(filas):
    if not filas:
        print("(sin resultados)")
        return

    print("\n" + "=" * 60)
    print("IMÁGENES MÁS SIMILARES")
    print("=" * 60)

    for i, (id_, desc, url, dist) in enumerate(filas, start=1):
        similitud = 1 - dist
        print(f"\n{i}. [{id_}] {desc}")
        print(f"   Similitud: {similitud:.4f}")
        print(f"   URL      : {url}")


# ==========================================
# MAIN
# ==========================================

def main():
    if len(sys.argv) < 2:
        print("Uso: python buscar_imagen.py <ruta_imagen_local>")
        print("  o : python buscar_imagen.py --url <https://...>")
        return

    if sys.argv[1] == "--url":
        if len(sys.argv) < 3:
            print("Falta la URL.")
            return
        print(f"Consultando por URL: {sys.argv[2]}")
        embedding = embedding_desde_url(sys.argv[2])
    else:
        ruta = sys.argv[1]
        print(f"Consultando por archivo: {ruta}")
        embedding = embedding_desde_archivo(ruta)

    print(f"Embedding generado: {len(embedding)} dimensiones")

    resultados = buscar_por_imagen(embedding, top_k=3)
    mostrar_resultados(resultados)


if __name__ == "__main__":
    main()