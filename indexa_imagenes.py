# ==========================================
# indexa_imagenes.py
# Genera embeddings CLIP para todas las imágenes
# de la tabla marcas_embeddings.
# ==========================================

import io
from pathlib import Path
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

BATCH_SIZE = 4  # imágenes por batch


# ==========================================
# CLIP
# ==========================================

print("Cargando CLIP...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval()
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)


def embedding_desde_url(url):
    """Carga y vectoriza una imagen desde URL o ruta local."""
    try:
        if url.startswith(("http://", "https://")):
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            imagen = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            ruta = Path(__file__).resolve().parent / url.lstrip("/\\")
            imagen = Image.open(ruta).convert("RGB")
    except Exception as e:
        print(f"   Error descargando {url}: {e}")
        return None

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
# INDEXAR IMÁGENES
# ==========================================

def indexar_imagenes():
    conn = conectar()
    cursor = conn.cursor()

    # Solo las filas que tienen URL y aún no tienen embedding_imagen
    cursor.execute("""
        SELECT id, descripcion, url_imagen
        FROM marcas_embeddings
        WHERE url_imagen IS NOT NULL
          AND url_imagen <> ''
          AND embedding_imagen IS NULL
        ORDER BY id
    """)

    filas = cursor.fetchall()
    total = len(filas)

    print(f"\n{total} imágenes pendientes de indexar.\n")

    for i, (id_, desc, url) in enumerate(filas, start=1):
        print(f"{i}/{total}. [{id_}] {desc[:50]}...")

        embedding = embedding_desde_url(url)

        if embedding is None:
            continue

        vector = vector_a_str(embedding)

        cursor.execute("""
            UPDATE marcas_embeddings
            SET embedding_imagen = %s::vector
            WHERE id = %s
        """, (vector, id_))

        print(f"   OK — {len(embedding)} dimensiones")

        # Commit parcial cada BATCH_SIZE para no perder progreso
        if i % BATCH_SIZE == 0:
            conn.commit()

    conn.commit()
    cursor.close()
    conn.close()

    print("\n================================")
    print("INDEXACIÓN DE IMÁGENES COMPLETADA")
    print("================================")


if __name__ == "__main__":
    indexar_imagenes()