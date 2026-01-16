# Migration Summary: incident_tickets → source

**Date:** January 9, 2026  
**Status:** ✅ COMPLETED

## Changes Executed

### 1. Database Schema (PostgreSQL)

```sql
-- Table renamed
incident_tickets → source

-- Objects renamed:
incident_tickets_id_seq → source_id_seq
audit_incident_tickets_changes → audit_source_changes
incident_tickets_pkey → source_pkey
incident_tickets_embedding_idx → source_embedding_idx

-- RLS Policies renamed:
incident_tickets_anon_read → source_anon_read
incident_tickets_authenticated_read → source_authenticated_read
incident_tickets_service_all → source_service_all
```

### 2. Code Updates

**Updated Files:**
- ✅ [supabase/api_service.py](supabase/api_service.py) - Marked DEPRECATED, JOIN source table
- ✅ [README.md](README.md) - Architecture diagrams updated
- ✅ [docs/changelog.md](docs/changelog.md) - Migration documented

**Created Files:**
- ✅ [supabase/verify_migration.py](supabase/verify_migration.py) - Verification script

## Verification Results

```
✅ source table exists
✅ incidents table exists (production)
✅ incidents has 10,633 records
✅ source has 10,633 records
✅ incidents has HNSW index (fast vector search)
✅ incidents has GIN/FTS index (full-text search)
✅ All embeddings populated in incidents table
```

## Architecture Status

**Before Migration:**
```
incident_tickets (confusing legacy name)
incidents (production but unclear relationship)
```

**After Migration:**
```
source (archive/import table)
    ↓
incidents (production source of truth)
    ↓
api_service_production.py (port 8001)
    ↓
Frontend (planned Next.js)
```

## Production Impact

- ✅ **ZERO downtime** - no production services affected
- ✅ api_service_production.py still queries incidents table (unchanged)
- ✅ embed_incidents_v4_cosine.py still updates incidents table (unchanged)
- ✅ Training notebooks use incidents table (unchanged)
- ⚠️ Legacy api_service.py now references source table (marked deprecated)

## Read-Only Status Note

The source table has SELECT grants for authenticated/anon users, but the postgres superuser can still write to it (this is expected PostgreSQL behavior). To enforce true read-only for superuser, you would need to:

```sql
-- Create read-only view (optional, for strict enforcement)
ALTER TABLE source RENAME TO source_internal;
CREATE VIEW source AS SELECT * FROM source_internal;
REVOKE ALL ON source_internal FROM public, authenticated, anon;
GRANT SELECT ON source TO authenticated, anon;
```

However, for the current architecture, this is **not necessary** since:
1. Application code (api_service_production.py) doesn't touch the source table
2. Only postgres superuser (you) has write access, providing operational flexibility
3. Source table serves as archive/backup of original ServiceNow data

## Next Steps

### Phase 2: API Consolidation (Week 1)
- Archive deprecated API files (api_service.py, api_service_with_reranker.py)
- Keep only api_service_production.py as unified endpoint
- Add comprehensive OpenAPI/Swagger documentation

### Phase 3: Frontend Development (Weeks 2-3)
- Create Next.js application
- 5 pages: search dashboard, incident detail, root cause analysis, analytics, admin
- Integrate with api_service_production.py endpoints

### Phase 4: Integration (Week 4)
- Bring scheduled ServiceNow sync into project folder
- Implement automated load_incidents.py runs
- Set up monitoring and alerts

## Rollback Plan (If Needed)

If you ever need to rollback:

```sql
-- Rename back to original
ALTER TABLE source RENAME TO incident_tickets;
ALTER SEQUENCE source_id_seq RENAME TO incident_tickets_id_seq;
ALTER TRIGGER audit_source_changes ON incident_tickets RENAME TO audit_incident_tickets_changes;
ALTER POLICY "source_anon_read" ON incident_tickets RENAME TO "incident_tickets_anon_read";
ALTER POLICY "source_authenticated_read" ON incident_tickets RENAME TO "incident_tickets_authenticated_read";
ALTER POLICY "source_service_all" ON incident_tickets RENAME TO "incident_tickets_service_all";
ALTER INDEX source_pkey RENAME TO incident_tickets_pkey;
ALTER INDEX source_embedding_idx RENAME TO incident_tickets_embedding_idx;
```

**However, rollback is NOT recommended** since the new naming convention provides much better semantic clarity.

---

**Migration completed successfully!** 🎉

Architecture is now clear: **source (archive) → incidents (production) → API → Frontend (planned)**
