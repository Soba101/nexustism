#!/bin/bash
# Script to run embedding generation inside Docker Compose network
# Execute from: supabase/supabase-project directory
# Usage: docker compose exec -T -u root db bash /workspace/scripts/run_embedding.sh

set -e

echo "=========================================="
echo "Installing Python dependencies..."
echo "=========================================="
pip install -q torch transformers sentence-transformers peft psycopg2-binary numpy tqdm 2>/dev/null || echo "[!] Some packages may already be installed"

echo ""
echo "=========================================="
echo "Generating V4 embeddings..."
echo "=========================================="

cd /workspace
export PYTHONPATH=/workspace:/workspace/notebook-fixes:$PYTHONPATH

# Modify the script to use internal Docker hostname
python3 << 'PYTHON_SCRIPT'
import sys
import os
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add to path
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/notebook-fixes')
sys.path.insert(0, '/workspace/supabase')

from deploy_model_v4 import V4CosineModelDeployment
from supabase.config import get_db_config

# Database config using internal Docker network (uses env vars)
DB_CONFIG = get_db_config(internal=True)

print("=" * 80)
print("V4 COSINE MODEL - SUPABASE EMBEDDING GENERATION (Docker Execution)")
print("=" * 80)

# Load model
print("\nLoading V4 Cosine model...")
try:
    model = V4CosineModelDeployment(
        model_path='/workspace/models/real_servicenow_finetuned_mpnet_lora/real_servicenow_v2_20260104_2321'
    )
    print(f"[OK] Model loaded successfully")
    print(f"   Embedding dimension: {model.embedding_dim}")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    sys.exit(1)

# Connect to database
print("\nConnecting to PostgreSQL (via Docker network)...")
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("[OK] Connected to database")
except Exception as e:
    print(f"[ERROR] Database connection failed: {e}")
    sys.exit(1)

# Fetch all incidents
print("\nFetching incidents from database...")
cursor.execute("SELECT id, short_description, description, category FROM incidents ORDER BY id")
incidents = cursor.fetchall()
print(f"[OK] Loaded {len(incidents)} incidents")

if len(incidents) == 0:
    print("[ERROR] No incidents found in database")
    cursor.close()
    conn.close()
    sys.exit(1)

# Generate embeddings
print("\nGenerating V4 embeddings (this may take 15-30 minutes)...")
embeddings_list = []

for inc_id, short_desc, desc, category in tqdm(incidents, desc="Embedding"):
    # Preprocess text (CONFIG pattern: content first, then context)
    text = f"{short_desc}\n{desc}\n\nCategory: {category}" if short_desc and desc else str(short_desc or desc or category)
    
    # Generate embedding
    embedding = model.encode(text)
    embeddings_list.append({
        'id': inc_id,
        'embedding': embedding
    })

print(f"[OK] Generated {len(embeddings_list)} embeddings")

# Upload embeddings to database
print("\nUploading embeddings to database...")
batch_size = 100
for batch_start in tqdm(range(0, len(embeddings_list), batch_size), desc="Uploading batches"):
    batch_end = min(batch_start + batch_size, len(embeddings_list))
    batch = embeddings_list[batch_start:batch_end]
    
    # Prepare data for pgvector
    for item in batch:
        embedding_list = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
        cursor.execute(
            "UPDATE incidents SET embedding = %s WHERE id = %s",
            (embedding_list, item['id'])
        )

conn.commit()
print(f"[OK] Uploaded {len(embeddings_list)} embeddings")

# Verify
print("\nVerifying embeddings...")
cursor.execute("SELECT COUNT(*) FROM incidents WHERE embedding IS NOT NULL")
embedded_count = cursor.fetchone()[0]
print(f"[OK] {embedded_count}/{len(incidents)} incidents have embeddings")

if embedded_count < len(incidents):
    print(f"[WARNING] {len(incidents) - embedded_count} incidents missing embeddings")

# Summary
print("\n" + "=" * 80)
print("DEPLOYMENT SUMMARY")
print("=" * 80)
print(f"✓ Model: V4 Cosine (real_servicenow_v2_20260104_2321)")
print(f"✓ Embedding dimension: {model.embedding_dim}")
print(f"✓ Incidents embedded: {embedded_count}/{len(incidents)}")
print(f"✓ Timestamp: {datetime.now().isoformat()}")
print("\nNext steps:")
print("1. Build HNSW indexes: conda run -n itsm python supabase/rebuild_v4_indexes.py")
print("2. Start production API: conda run -n itsm python -m uvicorn supabase/api_service_production:app --port 8001")

cursor.close()
conn.close()

PYTHON_SCRIPT
