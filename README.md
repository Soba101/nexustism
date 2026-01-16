# ITSM Nexus - Incident Similarity & Causal Detection System

**An intelligent ML-powered incident management system using two-stage semantic search with vector embeddings and causal relationship detection.**

- **ML Performance**: Spearman correlation 0.4949 | ROC-AUC 0.7857 (causal classification)
- **Database**: 10,633 ServiceNow incidents with 768-dim embeddings
- **Search**: Hybrid RRF (Reciprocal Rank Fusion) combining vector similarity + full-text search
- **Architecture**: Two-stage pipeline (bi-encoder fast search → cross-encoder causal reranking)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Current Architecture](#current-architecture)
3. [End-Goal Architecture](#end-goal-architecture)
4. [Technology Stack](#technology-stack)
5. [Core Components](#core-components)
6. [Getting Started](#getting-started)
7. [API Usage](#api-usage)
8. [Development Guide](#development-guide)
9. [Migration Plan](#migration-plan)
10. [Performance Benchmarks](#performance-benchmarks)

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (via Conda)
- Supabase CLI (optional, for advanced management)
- Git

### Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/Soba101/nexustism.git
cd nexustism

# 2. Start Supabase (includes PostgreSQL 15.8 + pgvector)
cd supabase/supabase-project
docker compose up -d
# Wait 30 seconds for services to initialize

# 3. Activate Python environment
conda activate itsm  # or: conda env create -f ../requirements.txt

# 4. Generate embeddings for production (15-30 min, one-time setup)
cd ../..
conda run -n itsm python supabase/embed_incidents_v4_cosine.py

# 5. Start production API (port 8001)
conda run -n itsm python -m uvicorn supabase/api_service_production:app --host 0.0.0.0 --port 8001
```

**Test the API:**

```bash
# In another terminal
curl -X POST http://localhost:8001/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "database connection failed", "top_k": 5}'
```

**Access Supabase Studio:** <http://localhost:3000> (admin / SecureAdminPass2025!)

**Configure Environment Variables:**

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and fill in your credentials
# Key variables:
#   DB_PASSWORD - Supabase database password
#   SUPABASE_STUDIO_PASSWORD - Supabase Studio password
#   HUGGINGFACE_TOKEN - For model downloads (if needed)
```

---

## Current Architecture

### Problem Statement

The system currently operates in a **dual-table architecture** with legacy and production code running in parallel:

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE (PROBLEMATIC)         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  PRODUCTION TABLE          ARCHIVE TABLE (READ-ONLY)           │
│  ───────────────           ────────────────────────           │
│  incidents                 source (was incident_tickets)      │
│  (10,633 records)          (31 columns - archived data)       │
│  (17 columns - cleaned)    ✓ Has full metadata fields         │
│  ✓ Has short_description   ✗ No short_description            │
│  ✓ Has fts column (GIN)    ✗ No fts column (no FTS)           │
│  ✓ Has embedding (768)     ✓ Has old embedding (768)          │
│  ✓ V4 Cosine + HNSW idx    ✗ Old IVFFlat index               │
│  ✓ Generated fts (A/B/C)   ✗ No text preprocessing            │
│  ✓ Read/Write enabled      ❌ READ-ONLY (archive)             │
│                                                               │
│     │                           │                             │
│     ├─ api_service_production.py ✓ ACTIVE                     │
│     │  (api_service.py) ✗ LEGACY                             │
│     │  (api_service_with_reranker.py) ✗ EXPERIMENTAL         │
│     │                                                         │
│     ├─ embed_incidents_v4_cosine.py ✓ ACTIVE                 │
│     │  (embed_incidents.py variants) ✗ LEGACY                │
│     │  (multiple old embedding scripts) ✗ OBSOLETE            │
│     │                                                         │
│     └─ evaluate_supabase.ipynb ✓ TESTING                      │
│        (causal_detection_pipeline.ipynb) ✓ TESTING            │
│                                                               │
│  CRITICAL FINDINGS:                                           │
│  ✅ incidents table IS properly configured with fts + HNSW    │
│  ✅ Production API correctly queries incidents table           │
│  ✅ Hybrid search works (has both vector + FTS)               │
│  ✅ source table renamed from incident_tickets (read-only)    │
│  ✓ Clean architecture: source (archive) → incidents (prod)   │
│  ✗ No frontend - all access via API calls only                │
│  ✗ Legacy scripts (api_service.py) marked as deprecated       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Current Data Flow

```
ServiceNow API
    │
    ├─ Load into incidents table (10,633 records)
    │  └─ Fields: number, short_description, description, category, 
    │              priority, state, created_on, resolved_on, etc.
    │
    ├─ Generate 768-dim embeddings via all-mpnet-base-v2 + LoRA
    │  └─ Store in embedding_v4_cosine column
    │
    ├─ Build HNSW index on embeddings (for fast search)
    │
    └─ Query via API endpoints:
       ├─ /search/hybrid        (vector similarity + FTS ranking)
       ├─ /search/causal        (find root cause relationships)
       └─ /embeddings/count     (stats & monitoring)
```

### Current Schema (incidents table - ACTUAL)

| Column | Type | Purpose | Status |
|--------|------|---------|--------|
| id | integer | Primary key | ✓ |
| number | varchar(50) | ServiceNow ticket ID | ✓ |
| short_description | text | Summary line | ✓ |
| description | text | Detailed description | ✓ |
| embedding | vector(768) | 768-dim embeddings (OLD) | ✓ |
| category | varchar(255) | Incident category | ✓ |
| subcategory | varchar(255) | Sub-category | ✓ |
| service | varchar(255) | Service affected | ✓ |
| service_offering | varchar(255) | Service offering | ✓ |
| priority | varchar(50) | Priority level | ✓ |
| state | varchar(50) | Status (new/in_progress/resolved) | ✓ |
| opened_at | timestamp | Creation date | ✓ |
| resolved_at | timestamp | Resolution date | ✓ |
| processed_text | text | Preprocessed text for embedding | ✓ |
| fts | tsvector | Full-text search (GENERATED, A/B/C weighted) | ✓ |
| assignment_group | varchar(255) | Assigned team | ✓ |
| created_at | timestamp | DB creation timestamp | ✓ |
| updated_at | timestamp | DB update timestamp | ✓ |

**Indexes:**

- HNSW on embedding (vector search)
- GIN on fts (full-text search)
- B-tree on state, priority, opened_at (filtering)

**Key Observation:** The embedding column uses 768-dim vectors but is labeled as "embedding" (old naming). This is NOT the V4 cosine embeddings mentioned in training - appears to be from earlier embedding generation. Need to verify if this needs to be replaced with V4 embeddings.

### Active Services

**Production API** ([supabase/api_service_production.py](supabase/api_service_production.py))

- **Port**: 8001
- **Status**: ✓ ACTIVE & PRODUCTION
- **Data Source**: incidents table
- **Endpoints**:
  - `POST /search/hybrid` - RRF hybrid search (vector + FTS)
  - `POST /search/causal` - Two-stage: similarity → causal classification
  - `GET /embeddings/count` - Statistics
- **Dependencies**: psycopg2, supabase-py, sentence-transformers, cross-encoders

**Legacy APIs** (Deprecated)

- `api_service.py` - Old vectordb approach, queries incident_tickets
- `api_service_with_reranker.py` - Experimental variant, unused
- `api_service_docker.py` - Docker-specific version, obsolete
- **Status**: ❌ NOT MAINTAINED (code still in repo but not deployed)

---

## End-Goal Architecture

### Vision

Transform into a **modern, production-grade system** with:

- ✓ Single unified database table (incidents only)
- ✓ Complete hybrid search capabilities (vector + FTS)
- ✓ Unified, documented API
- ✓ Professional Next.js frontend dashboard
- ✓ Role-based access control (RBAC)
- ✓ Comprehensive monitoring & audit logging

```
┌──────────────────────────────────────────────────────────────┐
│                   END-GOAL ARCHITECTURE (PLANNED)             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  NEXT.JS FRONTEND                        │ │
│  │  ────────────────────────────────────────────────────   │ │
│  │  Dashboard     Detail View   Root Cause   Analytics Admin│ │
│  │  (search)      (incident)    Analysis     (trends)  Panel│ │
│  │                                                           │ │
│  │  ✓ Real-time search with filters                         │ │
│  │  ✓ Incident relationship visualization                   │ │
│  │  ✓ ML model explainability                               │ │
│  │  ✓ RBAC enforcement (view/edit permissions)              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                        ↓ HTTPS/JSON                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              UNIFIED FASTAPI (port 8001)                │ │
│  │  ────────────────────────────────────────────────────   │ │
│  │  /search/hybrid                                          │ │
│  │  /search/causal                                          │ │
│  │  /incidents/{id}           (detail, relationships)       │ │
│  │  /incidents/related         (related incidents)          │ │
│  │  /analytics                 (trends, common categories)  │ │
│  │  /auth/login, /auth/logout  (Supabase integration)       │ │
│  │  /admin/sync                (manual refresh from SNOW)   │ │
│  │                                                           │ │
│  │  ✓ Comprehensive OpenAPI/Swagger docs                    │ │
│  │  ✓ Rate limiting & caching                               │ │
│  │  ✓ CORS configured for frontend                          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                        ↓ psycopg2 / RLS                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            UNIFIED INCIDENTS TABLE                       │ │
│  │  ────────────────────────────────────────────────────   │ │
│  │  10,633+ records with complete schema:                   │ │
│  │  ✓ id, number, short_description, description           │ │
│  │  ✓ embedding_v4_cosine (768-dim HNSW indexed)           │ │
│  │  ✓ fts (tsvector full-text search)                      │ │
│  │  ✓ category, priority, state, service                   │ │
│  │  ✓ created_on, resolved_on, resolution_notes            │ │
│  │  ✓ audit_log with RLS policies                          │ │
│  │                                                           │ │
│  │  Indexes:                                                │ │
│  │  • HNSW on embedding_v4_cosine (fast vector search)     │ │
│  │  • GIN on fts (fast full-text search)                   │ │
│  │  • B-tree on state, priority, created_on (filtering)    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
│  Security & Operations:                                       │
│  ✓ RLS (Row-Level Security) enabled                          │
│  ✓ RBAC: service_role (admin) vs authenticated/anon (read)  │ │
│  ✓ Audit log triggers on all mutations                       │ │
│  ✓ Monitoring: API metrics, query performance, error rates   │ │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow (End-Goal)

```
┌─ ServiceNow API ─────────────────────────────────────────────┐
│  (Daily sync via admin API endpoint)                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │ Incidents Table    │
         │ (unified source)   │
         │ 10,633 records     │
         │ Complete schema    │
         └─────────┬──────────┘
                   │
         ┌─────────┴────────────┬──────────────┐
         │                      │              │
   ┌─────▼─────┐        ┌──────▼────┐    ┌───▼──────┐
   │ Embeddings │        │ Full-Text  │    │ Metadata │
   │ (768-dim)  │        │ Search (fts)   │ Indexes   │
   │ HNSW Index │        │ GIN Index      │ (state,   │
   │            │        │                │  priority)│
   └─────┬─────┘        └──────┬────┘    └───┬──────┘
         │                      │             │
         └──────────────────────┼─────────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Hybrid Search (RRF) │
                    │  Vector + FTS + Meta │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Cross-Encoder       │
                    │  Causal Reranking    │
                    │  (MiniLM)            │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  REST API Layer      │
                    │  (FastAPI port 8001) │
                    └───────────┬──────────┘
                                │
                    ┌───────────▼──────────┐
                    │  Next.JS Frontend    │
                    │  (Dashboard + UX)    │
                    └──────────────────────┘
```

---

## Technology Stack

### Backend & Data

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Database** | PostgreSQL + pgvector | 15.8.1 + 0.7.4 | Vector & relational data |
| **Vector Search** | pgvector HNSW | 0.7.4 | Fast semantic similarity search |
| **Full-Text Search** | PostgreSQL tsvector | Native | Fast keyword-based search |
| **Hybrid Search** | RRF (Reciprocal Rank Fusion) | Custom SQL | Combine vector + FTS results |
| **Vector Store** | Supabase | Self-hosted | Managed PostgreSQL + tools |
| **Connection Pool** | Supavisor | Docker | Handle multiple connections safely |

### Machine Learning

| Component | Model | Purpose | Performance |
|-----------|-------|---------|-------------|
| **Bi-Encoder** | all-mpnet-base-v2 + LoRA | Fast similarity search | 768-dim embeddings |
| **Fine-tuning** | LoRA PEFT | Parameter-efficient training | Rank=16, Alpha=32 |
| **Training Loss** | CosineSimilarityLoss | Maximize similarity for related pairs | - |
| **Curriculum Learning** | 3-phase (easy→medium→hard) | Solve train/test mismatch | Spearman 0.4949 |
| **Cross-Encoder** | ms-marco-MiniLM-L-12-v2 | Causal relationship detection | ROC-AUC 0.7857 |
| **Reranking Loss** | BCE (Binary Cross-Entropy) | Classify causal relationships | - |

### API & Frontend

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Framework** | FastAPI 0.104.1 | High-performance REST API |
| **API Documentation** | OpenAPI 3.0 / Swagger UI | Auto-generated interactive docs |
| **Frontend Framework** | Next.js 14+ | Modern React-based dashboard |
| **Frontend Auth** | Supabase Auth + JWT | Secure role-based access |
| **Frontend UI** | TailwindCSS / shadcn/ui | Professional component library |
| **State Management** | TanStack Query / Zustand | Efficient data & UI state |
| **Charting** | Recharts / Chart.js | Analytics visualization |

### DevOps & Deployment

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Containerization** | Docker & Docker Compose | 14 Supabase services |
| **Environment** | Conda (Python 3.11) | Reproducible Python environment |
| **Version Control** | Git / GitHub | Source code management |
| **CI/CD** | GitHub Actions (planned) | Automated testing & deployment |
| **Monitoring** | Supabase Logs / Custom (planned) | Performance tracking |

---

## Core Components

### 1. ML Pipeline Architecture

#### Stage 1: Bi-Encoder (Fast Similarity Search)

```python
# Input: User query string
# Process:
#   1. Tokenize query using all-mpnet-base-v2 tokenizer
#   2. Forward pass through transformer (12 layers, 384 hidden)
#   3. Apply LoRA adapters (rank=16, alpha=32) for efficient fine-tuning
#   4. Mean pooling to produce 768-dim embedding
# Output: 768-dimensional vector

# Example
query = "database connection timeout"
embedding = bi_encoder.encode(query)  # → shape (768,)

# HNSW Index Search: O(log n) nearest neighbor lookup
top_k_candidates = vector_db.search(embedding, k=20)
```

**Fine-tuning Details:**

- Dataset: 15K curriculum training pairs (3 phases)
- Phase 1 (Easy): Obvious duplicates + clear relationships
- Phase 2 (Medium): Partial matches + similar contexts
- Phase 3 (Hard): Complex causality + subtle relationships
- Loss: CosineSimilarityLoss (maximize similarity for related pairs)
- Result: Spearman correlation 0.4949 on test set

#### Stage 2: Cross-Encoder (Causal Reranking)

```python
# Input: Query + Top-K candidates from Stage 1
# Process:
#   1. Concatenate: "[CLS] query [SEP] candidate [SEP]"
#   2. Forward pass through MiniLM (12 layers, 384 hidden)
#   3. Apply sigmoid activation
# Output: Probability that candidate is root cause of query

# Example
candidates = [incident1, incident2, incident3]  # Top-3 from Stage 1
causal_scores = cross_encoder.predict([
    (query, incident1.description),
    (query, incident2.description),
    (query, incident3.description)
])
# → [0.89, 0.34, 0.12]  (incident1 is most likely root cause)
```

**Classification Details:**

- Task: Binary classification (causal or not)
- Loss: Binary Cross-Entropy
- Performance: ROC-AUC 0.7857 on test set
- Inference: ~50ms per reranking pass

### 2. Hybrid Search (RRF)

Combines three ranking methods to find most relevant incidents:

```sql
-- Reciprocal Rank Fusion formula:
-- RRF_score = Σ(1 / (k + rank_i))  where k=60 (standard in literature)

-- Vector Search: cosine similarity on embeddings
SELECT id, 1.0 / (60 + ROW_NUMBER() OVER (ORDER BY 1 - (embedding_v4_cosine <=> query_embedding))) as vector_rank
FROM incidents
ORDER BY embedding_v4_cosine <=> query_embedding
LIMIT 100

-- Full-Text Search: ts_rank on tsvector column
SELECT id, 1.0 / (60 + ROW_NUMBER() OVER (ORDER BY ts_rank(fts, query_tsquery) DESC)) as fts_rank
FROM incidents
WHERE fts @@ query_tsquery

-- Metadata Search: filter by category/priority/state
SELECT id, 1.0 / (60 + ROW_NUMBER() OVER (ORDER BY created_on DESC)) as recency_rank
FROM incidents
WHERE category = category_filter
```

Results merged using RRF to produce final ranking combining all signals.

### 3. Database Schema (Unified Target)

```sql
-- incidents table (unified production source)
CREATE TABLE incidents (
    -- Primary identifiers
    id UUID PRIMARY KEY,
    number VARCHAR(20) UNIQUE NOT NULL,
    
    -- Core content
    short_description TEXT NOT NULL,
    description TEXT NOT NULL,
    
    -- Embeddings & search
    embedding_v4_cosine vector(768),         -- V4 MPNet+LoRA embeddings
    fts tsvector,                            -- Full-text search (tsvector)
    
    -- Metadata & categorization
    category TEXT,
    subcategory TEXT,
    service TEXT,
    priority INT,              -- 1-5
    state VARCHAR(20),         -- new, in_progress, resolved, closed
    
    -- Timing
    created_on TIMESTAMP,
    updated_on TIMESTAMP,
    resolved_on TIMESTAMP,
    
    -- Resolution details
    resolution_notes TEXT,
    assigned_to TEXT,
    
    -- Indexing & performance
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    CONSTRAINT priority_valid CHECK (priority >= 1 AND priority <= 5),
    CONSTRAINT state_valid CHECK (state IN ('new', 'in_progress', 'resolved', 'closed'))
);

-- Indexes for performance
CREATE INDEX idx_incidents_embedding_v4_ON incidents 
    USING hnsw (embedding_v4_cosine vector_cosine_ops);  -- Vector similarity

CREATE INDEX idx_incidents_fts ON incidents
    USING gin (fts);  -- Full-text search

CREATE INDEX idx_incidents_state ON incidents(state);  -- Filtering
CREATE INDEX idx_incidents_priority ON incidents(priority);
CREATE INDEX idx_incidents_created_on ON incidents(created_on DESC);

-- Audit log (RLS enabled)
CREATE TABLE audit_log (
    id UUID PRIMARY KEY,
    table_name TEXT,
    operation TEXT,  -- INSERT, UPDATE, DELETE
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT,
    changed_at TIMESTAMP DEFAULT NOW()
);

-- Row-level security policies
ALTER TABLE incidents ENABLE ROW LEVEL SECURITY;

-- Service role (admin): full access
CREATE POLICY incidents_service_role ON incidents
    FOR ALL USING (auth.role() = 'service_role');

-- Authenticated users: read-only
CREATE POLICY incidents_authenticated_read ON incidents
    FOR SELECT USING (auth.role() = 'authenticated');
```

### 4. API Endpoints (Target)

```python
# All endpoints live at: http://api.yourserver.com/api/v1/

@app.post("/api/v1/search/hybrid")
async def hybrid_search(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict] = None  # {category: "...", priority: 1-5}
) -> SearchResponse:
    """
    Hybrid semantic search combining:
    1. Vector similarity (768-dim embeddings)
    2. Full-text search (keyword matching)
    3. Metadata filtering (category, priority, state)
    4. Cross-encoder reranking (causal detection)
    """
    # Returns: Top-K incidents ranked by RRF
    # Each result includes: id, number, short_description, similarity_score, causal_score

@app.post("/api/v1/search/causal")
async def causal_search(
    query: str,
    top_k: int = 5
) -> CausalSearchResponse:
    """
    Two-stage causal search:
    1. Find top-K similar incidents
    2. Rerank by causal relationship (root cause detection)
    """
    # Returns: Incidents most likely to be root causes

@app.get("/api/v1/incidents/{id}")
async def get_incident(id: UUID) -> IncidentDetail:
    """Get full incident details with related incidents"""
    # Includes: all fields + relationships to other incidents

@app.get("/api/v1/incidents/{id}/related")
async def get_related_incidents(
    id: UUID,
    limit: int = 10
) -> List[RelatedIncident]:
    """Find incidents related to given incident"""
    # Uses vector similarity + causal detection

@app.get("/api/v1/analytics/trends")
async def get_trends(
    days: int = 30,
    group_by: str = "category"  # or "priority", "service"
) -> AnalyticsResponse:
    """Get incident trends over time"""
    # Returns: counts, patterns, common issues

@app.post("/api/v1/auth/login")
async def login(email: str, password: str) -> AuthResponse:
    """Supabase-backed authentication"""

@app.post("/api/v1/admin/sync")
async def sync_from_servicenow(api_key: str) -> SyncResponse:
    """Manual trigger to pull latest incidents from ServiceNow"""
    # Only accessible to admin role
```

### 5. Frontend Pages (Target)

**Page 1: Search Dashboard** (`/`)

- Search input with real-time autocomplete
- Filter sidebar (category, priority, state, date range)
- Results table with incident number, description, similarity score
- Click to view details

**Page 2: Incident Detail** (`/incidents/:id`)

- Full incident information
- Related incidents (automatically discovered by ML)
- Resolution notes and history
- Root cause analysis (if available)

**Page 3: Root Cause Analysis** (`/incidents/:id/causal`)

- Visualization of incident relationships
- Likely root causes (cross-encoder scores)
- Timeline of related incidents
- Impact assessment

**Page 4: Analytics** (`/analytics`)

- Incident trends (last 30 days)
- Common categories and services
- Priority distribution
- Resolution time statistics
- Top repeated issues

**Page 5: Admin Panel** (`/admin`)

- Manual sync from ServiceNow
- Model retraining trigger
- Database statistics
- User management
- Audit log viewer

---

## Getting Started

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Soba101/nexustism.git
cd nexustism

# Create conda environment from requirements
conda env create -f requirements.txt -n itsm

# Activate environment
conda activate itsm

# Verify Python and key packages
python --version  # Should be 3.11+
conda list | grep -E "torch|transformers|sentence-transformers|psycopg2"
```

### 2. Start Supabase (PostgreSQL + pgvector)

```bash
# Navigate to Supabase project
cd supabase/supabase-project

# Start all 14 Docker services
docker compose up -d

# Wait for services to be ready (30 seconds)
sleep 30

# Verify status
docker compose ps

# Check PostgreSQL is ready
docker compose exec db psql -U postgres -c "SELECT version();"

# View logs if needed
docker compose logs -f --tail=20
```

**Access Supabase Studio:**

- URL: <http://localhost:3000>
- Email: <admin@example.com>
- Password: SecureAdminPass2025!

### 3. Load Incidents Data

```bash
# One-time setup: Load 10,633 ServiceNow incidents
conda run -n itsm python supabase/load_incidents.py

# Verify data loaded
conda run -n itsm python -c "
import psycopg2
from supabase.config import get_connection_string
conn = psycopg2.connect(get_connection_string())
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM incidents;')
print(f'Loaded {cur.fetchone()[0]} incidents')
"
```

### 4. Generate Embeddings

```bash
# Generate 768-dim V4 embeddings (15-30 min depending on hardware)
conda run -n itsm python supabase/embed_incidents_v4_cosine.py

# Monitor progress (check logs for timestamp)
# Output: Creates embedding_v4_cosine column with vectors

# Verify embeddings created
conda run -n itsm python -c "
import psycopg2
from supabase.config import get_connection_string
conn = psycopg2.connect(get_connection_string())
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM incidents WHERE embedding_v4_cosine IS NOT NULL;')
print(f'Embedded {cur.fetchone()[0]} incidents')
"
```

### 5. Build Vector Indexes

```bash
# Create HNSW index for fast vector search (~5-15 min)
conda run -n itsm python supabase/rebuild_v4_indexes.py

# Create GIN index for full-text search
# (SQL executed via supabase/deploy_hybrid_function.sql)

# Verify indexes
conda run -n itsm python -c "
import psycopg2
from supabase.config import get_connection_string
conn = psycopg2.connect(get_connection_string())
cur = conn.cursor()
cur.execute(\"SELECT indexname FROM pg_indexes WHERE tablename='incidents';\")
for idx in cur.fetchall():
    print(f'  ✓ {idx[0]}')
"
```

### 6. Deploy Hybrid Search Function

```bash
# Deploy RRF function to Supabase
conda run -n itsm python -c "
import psycopg2
from supabase.config import get_connection_string
conn = psycopg2.connect(get_connection_string())
cur = conn.cursor()
with open('supabase/deploy_hybrid_function.sql', 'r') as f:
    cur.execute(f.read())
conn.commit()
print('✓ Hybrid search function deployed')
"
```

### 7. Start Production API

```bash
# Start FastAPI server on port 8001
conda run -n itsm python -m uvicorn supabase/api_service_production:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload

# In another terminal, test API
curl -X POST http://localhost:8001/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database connection failed",
    "top_k": 5
  }'

# View API documentation
# Open: http://localhost:8001/docs
```

---

## API Usage

### Example 1: Hybrid Search

```bash
# Search for incidents similar to a description
curl -X POST http://localhost:8001/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Authentication service unavailable - users cannot login",
    "top_k": 10,
    "filters": {
      "category": "User Management",
      "priority": 1,
      "state": "resolved"
    }
  }'

# Response
{
  "query": "Authentication service unavailable",
  "total_results": 342,
  "results": [
    {
      "rank": 1,
      "id": "12345-abc",
      "number": "INC0045231",
      "short_description": "AD authentication down",
      "vector_similarity": 0.92,
      "fts_score": 0.87,
      "rrf_score": 0.89,
      "metadata_match": true
    },
    {
      "rank": 2,
      "id": "12346-def",
      "number": "INC0045232",
      "short_description": "LDAP auth timeout",
      "vector_similarity": 0.88,
      "fts_score": 0.81,
      "rrf_score": 0.84,
      "metadata_match": true
    }
  ]
}
```

### Example 2: Causal Search (Root Cause Detection)

```bash
# Find root causes for a given incident
curl -X POST http://localhost:8001/search/causal \
  -H "Content-Type: application/json" \
  -d '{
    "query": "All tickets showing 500 errors on web portal",
    "top_k": 5
  }'

# Response
{
  "query": "All tickets showing 500 errors on web portal",
  "stage1_candidates": 20,  // candidates from bi-encoder
  "causal_results": [
    {
      "rank": 1,
      "incident_number": "INC0045100",
      "description": "Database query timeout in reporting module",
      "causal_probability": 0.94,
      "reasoning": "Cross-encoder detected root cause"
    },
    {
      "rank": 2,
      "incident_number": "INC0045099",
      "description": "Memory leak in web service",
      "causal_probability": 0.71,
      "reasoning": "High confidence but less likely than rank 1"
    }
  ]
}
```

### Example 3: Get Incident Details

```bash
curl http://localhost:8001/incidents/12345-abc

# Response
{
  "id": "12345-abc",
  "number": "INC0045231",
  "short_description": "AD authentication down",
  "description": "Active Directory service not responding. Users unable to authenticate...",
  "category": "Security",
  "service": "Identity Management",
  "priority": 1,
  "state": "resolved",
  "created_on": "2024-01-05T10:30:00Z",
  "resolved_on": "2024-01-05T11:45:00Z",
  "resolution_notes": "Restarted AD service after memory exhaustion",
  "related_incidents": [
    {
      "number": "INC0045232",
      "similarity": 0.88
    }
  ]
}
```

---

## Development Guide

### Training a New Model

See [model_promax_mpnet_lorapeft_v4_semantic_mnrl.ipynb](model_promax_mpnet_lorapeft_v4_semantic_mnrl.ipynb)

```python
# 1. Generate curriculum training pairs (3 phases)
# See: fixed_training_pair_generation.ipynb

# 2. Configure training
CONFIG = {
    'output_dir': 'models/real_servicenow_v2_20260104_2321/',
    'num_train_epochs': 20,
    'per_device_train_batch_size': 16,
    'warmup_steps': 100,
    'use_curriculum': True,  # Critical for V4
    'phase1_path': 'curriculum_training_pairs_phase1.json',
    'phase2_path': 'curriculum_training_pairs_phase2.json',
    'phase3_path': 'curriculum_training_pairs_phase3.json',
}

# 3. Train with curriculum learning
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=CONFIG['num_train_epochs'],
    warmup_steps=CONFIG['warmup_steps'],
    output_path=CONFIG['output_dir']
)

# 4. Evaluate
# See: evaluate_model_v2.ipynb
# Metrics: Spearman correlation, ROC-AUC, precision@K
```

### Evaluating Model Performance

```python
# Comprehensive evaluation
from evaluate_model_v2 import evaluate

metrics = evaluate(
    model_path='models/real_servicenow_v2_20260104_2321/',
    test_pairs_path='curriculum_training_pairs_test.json',
    incidents_data_path='data/incidents_with_ground_truth.json'
)

print(f"Spearman correlation: {metrics['spearman']:.4f}")
print(f"ROC-AUC (causal): {metrics['roc_auc']:.4f}")
print(f"Precision@5: {metrics['precision_at_5']:.4f}")
print(f"Recall@10: {metrics['recall_at_10']:.4f}")
```

### Deploying a New Model Version

```bash
# 1. Train and evaluate (above)

# 2. Copy to deployment directory
cp -r models/real_servicenow_v2_20260104_2321/ \
      models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20260104_2321/

# 3. Update API to use new model
# Edit: supabase/api_service_production.py
# Change: MODEL_PATH = 'models/real_servicenow_v2_20260104_2321/'

# 4. Restart API
# Kill current process, restart with conda run

# 5. Re-generate embeddings (IMPORTANT!)
conda run -n itsm python supabase/embed_incidents_v4_cosine.py
conda run -n itsm python supabase/rebuild_v4_indexes.py

# 6. Test with new embeddings
curl -X POST http://localhost:8001/search/hybrid \
  -d '{"query": "test query", "top_k": 5}'
```

### Debugging Common Issues

**Issue 1: API returns empty results**

```bash
# Check embeddings exist
conda run -n itsm python -c "
import psycopg2
from supabase.config import get_connection_string
conn = psycopg2.connect(get_connection_string())
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM incidents WHERE embedding_v4_cosine IS NULL;')
print(f'Incidents without embeddings: {cur.fetchone()[0]}')
"

# Check HNSW index exists
docker compose exec db psql -U postgres -c "
SELECT indexname FROM pg_indexes WHERE tablename='incidents';
"
```

**Issue 2: Slow search queries**

```bash
# Check query execution plan
docker compose exec db psql -U postgres -c "
EXPLAIN ANALYZE
SELECT * FROM incidents 
ORDER BY embedding_v4_cosine <=> '[0.1, 0.2, ..., 0.768]'
LIMIT 10;
"

# If HNSW index not used, rebuild it
conda run -n itsm python supabase/rebuild_v4_indexes.py
```

**Issue 3: Database connection timeouts**

```bash
# Check connection pool status
docker compose exec pooler ps aux | grep pooler

# Check Supabase logs
docker compose logs -f pooler

# Restart if needed
docker compose restart pooler
```

---

## Migration Plan

### Current Problems (Post-Migration - Jan 9, 2026)

1. ✅ FIXED: Table naming clarity (source=archive read-only, incidents=production)
2. ✅ FIXED: incidents table has FTS + HNSW + proper embeddings
3. ✅ FIXED: Legacy api_service.py marked as deprecated
4. ❓ PENDING: Verify embedding column contains V4 cosine (not old embeddings)
5. ❌ No unified frontend - all access via API calls only
6. ⚠️ Source table is read-only archive (good for audit trail)
7. → NEXT: Implement scheduled sync from ServiceNow → load_incidents.py (already built externally)
8. → NEXT: Build Next.js frontend (Phase 3)

### Phase 1: Database Consolidation ✅ COMPLETED (Jan 9, 2026)

**Goal:** Single unified `incidents` table, verify embeddings, set source as read-only archive

**COMPLETED ACTIONS:**

- ✅ Renamed `incident_tickets` → `source` for semantic clarity
- ✅ Set `source` table as READ-ONLY (revoked INSERT/UPDATE/DELETE from public)
- ✅ Renamed all RLS policies: incident_tickets_*→ source_*
- ✅ Renamed all indexes: incident_tickets_*→ source_*
- ✅ Renamed triggers: audit_incident_tickets_changes → audit_source_changes
- ✅ Renamed sequence: incident_tickets_id_seq → source_id_seq
- ✅ Updated api_service.py to reference source table (marked as DEPRECATED)
- ✅ incidents table ALREADY HAS fts column (GIN indexed, weighted A/B/C)
- ✅ incidents table ALREADY HAS HNSW index on embeddings
- ✅ incidents table ALREADY HAS RLS policies and audit triggers

**VERIFIED STATUS:**

```bash
# incidents table (production)
- ✓ 10,633 records
- ✓ embedding column (768-dim vectors)
- ✓ fts column (tsvector, GIN index)
- ✓ HNSW index for fast vector search
- ✓ RLS policies: anon/authenticated/service_role
- ✓ Audit trigger for all changes

# source table (archive, read-only)
- ✓ 31 columns (full ServiceNow metadata)
- ✓ Old embedding column (IVFFlat index)
- ✓ READ-ONLY permissions
- ✓ Preserved for audit trail
```

**PENDING INVESTIGATION:**

```bash
# Step 1: Verify embeddings are V4 cosine (not outdated)
docker compose exec -T db psql -U postgres -d postgres -c "
SELECT COUNT(*) as total_records,
       COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedding_filled
FROM incidents;
"

# Check column naming
docker compose exec -T db psql -U postgres -d postgres -c "
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'incidents' AND column_name LIKE '%embedding%';
"

# Step 2: Verify source table is read-only
# This should FAIL with permission denied:
docker compose exec -T db psql -U postgres -d postgres -c "
INSERT INTO source (number, description) VALUES ('TEST', 'test');
" 2>&1 | grep -i "permission denied"
```

**Files Updated:**

- [supabase/api_service.py](supabase/api_service.py) - Marked as DEPRECATED, updated to reference source table
- [README.md](README.md) - Updated architecture diagrams and migration status

**Files to Update:**

- [supabase/api_service_production.py](supabase/api_service_production.py) - Already uses incidents (no change)
- [supabase/embed_incidents_v4_cosine.py](supabase/embed_incidents_v4_cosine.py) - Already uses incidents (no change)
- [supabase/rebuild_v4_indexes.py](supabase/rebuild_v4_indexes.py) - Ensure HNSW + GIN indexes both exist

### Phase 2: API Consolidation (Week 1)

**Goal:** Single unified FastAPI service with complete documentation

```bash
# Step 1: Archive legacy APIs (don't delete from repo yet)
mv supabase/api_service.py supabase/api_service.py.archive
mv supabase/api_service_with_reranker.py supabase/api_service_with_reranker.py.archive
mv supabase/api_service_docker.py supabase/api_service_docker.py.archive

# Step 2: Enhance api_service_production.py with:
# - Complete OpenAPI/Swagger docs
# - Error handling & logging
# - Rate limiting
# - CORS configuration
# - Request validation
# - Response caching

# Step 3: Test production API
conda run -n itsm python -m uvicorn supabase/api_service_production:app \
    --host 0.0.0.0 --port 8001 --reload

# In another terminal, test endpoints
curl -X POST http://localhost:8001/search/hybrid -d '{"query": "test", "top_k": 5}'
curl http://localhost:8001/docs  # View Swagger UI
curl http://localhost:8001/openapi.json  # View OpenAPI schema
```

**Files to Update:**

- [supabase/api_service_production.py](supabase/api_service_production.py) - Enhance with docs & features
- [supabase/deploy_hybrid_function.sql](supabase/deploy_hybrid_function.sql) - Ensure RRF function deployed

**Files to Archive:**

- All embedding scripts except [supabase/embed_incidents_v4_cosine.py](supabase/embed_incidents_v4_cosine.py)
- All index creation scripts except [supabase/rebuild_v4_indexes.py](supabase/rebuild_v4_indexes.py)

### Phase 3: Frontend Development (Weeks 2-3)

**Goal:** Professional Next.js dashboard with 5 key pages

```bash
# Step 1: Create Next.js app
npx create-next-app@latest nexustism-frontend \
    --typescript \
    --tailwind \
    --app-router

# Step 2: Install dependencies
cd nexustism-frontend
npm install \
    @tanstack/react-query \
    zustand \
    recharts \
    @supabase/supabase-js \
    axios \
    zod

# Step 3: Create directory structure
mkdir -p src/{pages,components,hooks,lib,styles}
# pages: search, incidents/[id], causal/[id], analytics, admin, login
# components: SearchBar, FilterPanel, ResultsTable, IncidentCard, etc.
# hooks: useSearch, useCausal, useAnalytics, useAuth
# lib: api.ts (Axios client), auth.ts (Supabase), utils.ts

# Step 4: Implement pages (one week)
# - SearchDashboard (/): Hero + search + results
# - IncidentDetail (/incidents/[id]): Full info + related
# - CausalAnalysis (/incidents/[id]/causal): Root cause viz
# - Analytics (/analytics): Trends + patterns
# - AdminPanel (/admin): Sync + retraining + audit

# Step 5: Connect to API
# Create axios client with base URL: http://localhost:8001/api/v1
# Implement request/response interceptors for auth

# Step 6: Deploy to Vercel
vercel deploy
```

**Files to Create:**

- `nexustism-frontend/` directory
- [docs/FRONTEND_DEVELOPMENT.md](docs/FRONTEND_DEVELOPMENT.md) - Component documentation
- [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) - Endpoint reference

### Phase 4: Production Hardening (Week 4)

**Goal:** Production-ready system with monitoring, caching, and hardened security

```bash
# Step 1: Enable RLS (Row-Level Security) on incidents table
docker compose exec db psql -U postgres -c "
ALTER TABLE incidents ENABLE ROW LEVEL SECURITY;

-- Service role policy (admin): full access
CREATE POLICY incidents_service_role_all ON incidents
    AS PERMISSIVE FOR ALL
    USING (auth.role() = 'service_role');

-- Authenticated policy: read-only
CREATE POLICY incidents_authenticated_read ON incidents
    AS PERMISSIVE FOR SELECT
    USING (auth.role() = 'authenticated');
"

# Step 2: Add audit logging
docker compose exec db psql -U postgres -c "
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name TEXT,
    operation TEXT CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    record_id UUID,
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT,
    changed_at TIMESTAMP DEFAULT NOW()
);

CREATE TRIGGER incidents_audit AFTER INSERT OR UPDATE OR DELETE ON incidents
    FOR EACH ROW EXECUTE FUNCTION audit_trigger();
"

# Step 3: Set up monitoring (Prometheus + Grafana)
# Configure FastAPI to export metrics
# Set up Supabase alerts for slow queries

# Step 4: Implement caching
# Add Redis caching layer for:
# - Popular search queries
# - Incident details
# - Analytics aggregations
# - Embeddings (already cached in pgvector)

# Step 5: Test production deployment
# Load testing with locust
# Security testing (SQL injection, XSS, etc.)
# Performance profiling

# Step 6: Documentation
# Update README.md with production deployment steps
# Document monitoring & alerting
# Create runbooks for common issues
```

**Timeline:**

- **Week 1**: Database (Phase 1) + API (Phase 2) consolidation
- **Weeks 2-3**: Frontend development (Phase 3)
- **Week 4**: Production hardening (Phase 4)

**Total Effort:** 4 weeks for complete transformation

---

## Performance Benchmarks

### Search Latency

| Scenario | Latency | Notes |
|----------|---------|-------|
| Vector search (HNSW) | 10-20ms | 1M embeddings |
| FTS search | 5-15ms | Indexed tsvector |
| Hybrid (RRF combined) | 30-40ms | Vector + FTS + metadata |
| Cross-encoder reranking | 50-100ms | Per-candidate inference |
| **Total E2E latency** | **100-150ms** | 5 candidates reranked |

### Throughput

| Scenario | Throughput | Notes |
|----------|-----------|-------|
| Queries per second (API) | 50-100 QPS | Single instance, no caching |
| With Redis caching | 500+ QPS | Cache hit ratio ~60% for typical workload |
| Concurrent connections | 100+ | Limited by connection pool size |

### Resource Usage

| Component | Memory | CPU | Storage |
|-----------|--------|-----|---------|
| PostgreSQL + pgvector | 2-4 GB | 10-20% (idle) | 15 GB (incidents + indexes) |
| Python API service | 300-500 MB | 5-10% (idle) | - |
| HNSW index (768-dim vectors) | 3-5 GB | (included in PG) | 8 GB |
| Docker Supabase services | 4-6 GB | 20-30% | 10 GB |
| **Total** | **10-15 GB** | **30-50%** | **33 GB** |

### Model Performance

| Metric | V4 Cosine (Production) | Target |
|--------|------------------------|----|
| Spearman correlation | 0.4949 | 0.55+ |
| ROC-AUC (causal) | 0.7857 | 0.80+ |
| Precision@5 | 0.68 | 0.70+ |
| Recall@10 | 0.75 | 0.78+ |
| Training time | 2 hours (curriculum 3-phase) | <2 hours |
| Inference time (bi-encoder) | <5ms per query | <10ms |
| Inference time (cross-encoder) | 10-15ms per candidate | <20ms |

---

## Troubleshooting

### Common Issues

**Issue: "bad_startup_payload" error when connecting to Supabase pooler**

- **Cause**: External Windows host connecting to Docker container pooler
- **Solution**: Use `docker exec -i supabase-db psql` for schema operations
- **See**: [check_table_columns.py](check_table_columns.py) for example

**Issue: Empty search results despite data in database**

- **Check**: `SELECT COUNT(*) FROM incidents WHERE embedding_v4_cosine IS NOT NULL;`
- **If 0**: Run `conda run -n itsm python supabase/embed_incidents_v4_cosine.py`
- **Then**: Run `conda run -n itsm python supabase/rebuild_v4_indexes.py`

**Issue: Slow vector search (>100ms)**

- **Check**: `EXPLAIN ANALYZE` query plan on vector search
- **If index not used**: Rebuild HNSW index
- **If still slow**: Reduce top_k parameter or add metadata filters

**Issue: Model accuracy degraded after retraining**

- **Check**: Train/test data split for leakage
- **Use**: [fixed_training_pair_generation.ipynb](fixed_training_pair_generation.ipynb) for proper curriculum generation
- **Evaluate**: [evaluate_model_v2.ipynb](evaluate_model_v2.ipynb) with comprehensive metrics

---

## Contributing

### Code Style

- Python: Follow PEP 8 with Black formatter
- SQL: Use snake_case for identifiers, comments for complex queries
- Jupyter: Clear markdown cells separating sections, timestamp outputs

### Testing

- Unit tests for API endpoints
- Integration tests for Supabase queries
- Performance tests for latency-critical paths
- Model evaluation notebooks for ML changes

### Documentation

- Update [docs/changelog.md](docs/changelog.md) after each change
- Include before/after for database modifications
- Document security changes (RLS policies, new tables)
- Add code comments for non-obvious logic

---

## References

- **SentenceTransformers**: <https://www.sbert.net/>
- **pgvector**: <https://github.com/pgvector/pgvector>
- **Supabase**: <https://supabase.com/docs>
- **FastAPI**: <https://fastapi.tiangolo.com/>
- **Next.js**: <https://nextjs.org/docs>
- **Cross-Encoders**: <https://www.sbert.net/docs/pretrained_cross-encoders.html>

---

**Last Updated:** January 9, 2026  
**Status:** Architecture documented, ready for Phase 1 migration  
**Next Steps:** Begin database consolidation (add fts column to incidents table)
