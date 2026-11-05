// Idempotency records are separate from jobs so that several requests can map to one revision.
CREATE CONSTRAINT idempotency_request_unique IF NOT EXISTS FOR (i:IdempotencyRecord) REQUIRE (i.workspace_id, i.request_key) IS UNIQUE;
CREATE INDEX job_revision_lookup IF NOT EXISTS FOR (j:IngestionJob) ON (j.workspace_id, j.revision_id);
CREATE INDEX chunk_revision_lookup IF NOT EXISTS FOR (c:Chunk) ON (c.workspace_id, c.revision_id);
CREATE INDEX claim_revision_lookup IF NOT EXISTS FOR (c:Claim) ON (c.workspace_id, c.revision_id);
