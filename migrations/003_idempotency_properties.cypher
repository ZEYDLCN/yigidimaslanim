// Replays must not depend on relationships to revisions and jobs: a purge removes those nodes after
// retention, and a missing relationship would make a replayed request look new. Store the ids on
// the record itself and backfill records created before this migration.
MATCH (i:IdempotencyRecord)-[:FOR_NOTE]->(n:Note) SET i.note_id = n.note_id;
MATCH (i:IdempotencyRecord)-[:FOR_REVISION]->(r:NoteRevision) SET i.revision_id = r.revision_id;
MATCH (i:IdempotencyRecord)-[:FOR_JOB]->(j:IngestionJob) SET i.job_id = j.job_id;
CREATE INDEX idempotency_note_lookup IF NOT EXISTS FOR (i:IdempotencyRecord) ON (i.workspace_id, i.created_at);
