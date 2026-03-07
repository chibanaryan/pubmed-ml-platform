CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    pmid BIGINT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    authors JSONB DEFAULT '[]',
    journal TEXT,
    pub_date DATE,
    mesh_terms JSONB DEFAULT '[]',
    keywords JSONB DEFAULT '[]',
    doi TEXT,
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    pmid BIGINT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    embedding vector,  -- dimension varies by model: 384 for MiniLM, 768 for PubMedBERT
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(pmid, model_name)
);

-- Vector indexes: created AFTER data is loaded.
-- Using HNSW with expression casts since the column is untyped (supports multiple dims).
-- Queries must cast to match: e.embedding::vector(384) <=> query::vector(384)
--
--   CREATE INDEX idx_embeddings_hnsw_minilm
--       ON embeddings USING hnsw ((embedding::vector(384)) vector_cosine_ops)
--       WHERE model_name = 'all-MiniLM-L6-v2';
--
--   CREATE INDEX idx_embeddings_hnsw_pubmedbert
--       ON embeddings USING hnsw ((embedding::vector(768)) vector_cosine_ops)
--       WHERE model_name = 'pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb';

CREATE INDEX IF NOT EXISTS idx_embeddings_model
    ON embeddings(model_name);

CREATE INDEX IF NOT EXISTS idx_papers_pub_date
    ON papers(pub_date);

CREATE INDEX IF NOT EXISTS idx_papers_mesh_terms
    ON papers USING gin(mesh_terms);

-- Track ingestion state
CREATE TABLE IF NOT EXISTS ingestion_state (
    category TEXT PRIMARY KEY,
    last_fetched_date DATE NOT NULL,
    total_fetched BIGINT DEFAULT 0,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
