# Findings – ITSM Insight Nexus

## What the project is

- AI-powered IT Service Management analytics platform that layers semantic search and relationship discovery on top of ServiceNow-style incident data.
- Frontend: Vite + React with shadcn/tailwind; Backends: Node.js auth, Python FastAPI AI service, PostgreSQL + pgvector; LM Studio supplies local embeddings.
- Core AI capabilities: ticket similarity search, automated parent/child (duplicate/related) linking, configurable thresholds, and a background worker that keeps embeddings fresh.
- Supporting scripts cover batch embedding, relationship establishment, benchmarking, and fine-tuning to adapt models to the ITSM domain.

## What we’re trying to achieve

- Reduce ticket backlog noise by surfacing near-duplicate incidents and building parent/child ticket graphs automatically.
- Deliver fast, privacy-preserving semantic search without external APIs by running embeddings locally (LM Studio).
- Identify which embedding model best balances quality and speed for ITSM text, with a path to fine-tune and improve domain fit.
- Provide measurable evaluation (latency, separation gap, relationship quality) so model choices are evidence-based.

## Architecture (end-to-end)

- Frontend (Vite/React) talks to FastAPI AI backend (port 8000) and Node.js auth (port 3001); PostgreSQL + pgvector stores tickets and embeddings.
- LM Studio runs locally and serves OpenAI-compatible embedding APIs; Docker compose wires LM Studio → AI backend → Postgres, with a background worker consuming the embedding queue.
- Dual-vector schema supports A/B: `embedding` (768-dim Gemma) and `embedding_4096` (4096-dim Qwen3); similarity queries choose the column based on requested model, and 4096-dim uses brute-force search (no HNSW index cap >2000 dims).
- Scripts drive batch embedding, relationship establishment, benchmarking, and model comparison; FastAPI exposes similarity search and family graph endpoints consumed by the frontend.

## Model intent and training plan

- Primary goal: high-quality embeddings that separate semantically similar ITSM tickets from unrelated ones, enabling duplicate detection and family graphing.
- Current production options: `text-embedding-embeddinggemma-300m-qat` (768-dim, fast) and `text-embedding-qwen3-embedding-8b` (4096-dim, higher-capacity). Dual-column storage supports A/B testing (`embedding` vs `embedding_4096`).
- Fine-tuning notebook (`backend-python/scripts/finetuning/finetune_model.ipynb`) focuses on `sentence-transformers/all-mpnet-base-v2` using contrastive learning:
  - Builds positive pairs from tickets in the same category and negative pairs from different categories (source: `data/servicenow_incidents_full.json`).
  - Converts pairs to `InputExample`s, splits train/eval, trains with cosine similarity loss, evaluates mid-epoch, and saves best model + metadata to `models/all-mpnet-finetuned`.
  - Adds a relationship classifier: encodes labeled ticket pairs (`duplicate/related/causal/none`), trains multinomial logistic regression on combined embedding features, and saves to `relationship_classifier/`.
  - Includes optional LLM-assisted labeling (GPT-4o-mini) to tag candidate ticket pairs and build `relationship_pairs.csv/json`.
- Target integration: replace or complement LM Studio embeddings with the fine-tuned model, regenerate embeddings, and rerun quality/performance benchmarks.

## Training methods (fine-tuning workflow)

- **Contrastive embedding training:** all-mpnet-base-v2 encoder trained with cosine similarity loss on balanced positive/negative ticket pairs (category-based positives, cross-category negatives); shuffled split with `eval_split=0.1`, mid-epoch evaluation via `EmbeddingSimilarityEvaluator`, best checkpoint saved under `models/all-mpnet-finetuned` with training metadata.
- **Hyperparameters (default in notebook):** epochs=100, batch_size=32, lr=2e-5, warmup_steps=100; output path configurable via `CONFIG`.
- **Data prep:** ticket text = `short_description + description`; pairs generated from `data/servicenow_incidents_full.json`, persisted to `data/training_pairs.json` for reuse.
- **Relationship classifier:** reuse fine-tuned embeddings, build pairwise features `[emb_a, emb_b, |emb_a-emb_b|, emb_a*emb_b]`, train multinomial logistic regression, and persist model + label map to `relationship_classifier/`.
- **LLM-assisted labeling (optional):** GPT-4o-mini labels top-K similar ticket pairs into `duplicate/related/causal/none`, producing `relationship_pairs.csv/json` to strengthen supervised relationship training.

## What has been tested

- Benchmark suite (Phase 1 scripts) executed against the 78-ticket dataset; results logged in `docs/S2/model-results.md`:
  - **EmbeddingGemma-300m-qat (768-dim)**
    - Speed: mean 26.2 ms per ticket; batch throughput ~65.9 tickets/sec.
    - Similarity search: ~1.2 ms for top-10 (sequential scan, no HNSW index).
    - End-to-end (embed → store → search): ~68.7 ms total.
    - Quality: inverted separation gap (-0.0402) with same-category mean 0.4933 vs different-category 0.5335; parent/child links average similarity ~0.497 with 38.9% category agreement. Fails quality criteria despite excellent speed.
  - **Qwen3-8B (4096-dim) – initial run**
    - Speed: mean 436 ms per ticket (~16x slower than Gemma); ~2.9 tickets/sec throughput.
    - Similarity search: ~1.4 ms on small dataset.
    - End-to-end: failed due to 4096-dim vs 768-dim schema mismatch (could not store embeddings).
    - Quality: not measurable in this run (data stored with Gemma embeddings).
  - **Qwen3-8B – dual-column follow-up (docs/S2/DUAL_MODEL_SETUP_COMPLETE.md)**
    - Added `embedding_4096` column and routing to support 4096-dim vectors; no HNSW index (pgvector limit 2000 dims) but brute-force acceptable at small scale.
    - Early sample (20 tickets) shows improved separation gap (+0.0947 vs Gemma +0.0326) and higher same-category similarity (0.591 vs 0.496), indicating better quality at the cost of latency.
- Evaluation tooling in place: scripts for embedding speed, similarity search, E2E pipeline, similarity distribution, parent-child link quality, and compare_models; quickstart instructions in `docs/S2/QUICKSTART_MODEL_TESTING.md`.

## Status snapshot

- Platform functionality (similarity search, background embeddings, relationship graphing) is implemented; quality of embeddings is the main blocker for production-grade duplicate detection.
- Fine-tuning path (all-mpnet-base-v2 contrastive + relationship classifier) is prepared but results need to be generated and benchmarked.
- Dual-model A/B infrastructure exists to keep testing Qwen3-4096 vs Gemma-768; decision hinges on re-running benchmarks with the updated schema and larger sample sizes.
