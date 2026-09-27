CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS imagenes (
    id SERIAL PRIMARY KEY,
    nombre TEXT NOT NULL,
    ruta TEXT NOT NULL,
    embedding vector(512)
);

CREATE INDEX IF NOT EXISTS idx_imagenes_embedding
ON imagenes
USING hnsw (embedding vector_cosine_ops);

ALTER TABLE brand.marcas_embeddings
        ADD COLUMN IF NOT EXISTS embedding_imagen vector(512);