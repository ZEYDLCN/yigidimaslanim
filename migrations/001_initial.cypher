// Embedding dimension 1024 belongs to the proposed Qwen3-Embedding-0.6B profile.
// Verify against the served model before applying (architecture section 5.5).
CREATE CONSTRAINT workspace_id_unique IF NOT EXISTS FOR (w:Workspace) REQUIRE w.workspace_id IS UNIQUE;
CREATE CONSTRAINT note_id_unique IF NOT EXISTS FOR (n:Note) REQUIRE (n.workspace_id, n.note_id) IS UNIQUE;
CREATE CONSTRAINT revision_id_unique IF NOT EXISTS FOR (r:NoteRevision) REQUIRE (r.workspace_id, r.revision_id) IS UNIQUE;
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.workspace_id, c.chunk_id) IS UNIQUE;
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.workspace_id, e.entity_id) IS UNIQUE;
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS FOR (c:Claim) REQUIRE (c.workspace_id, c.claim_id) IS UNIQUE;
CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:IngestionJob) REQUIRE (j.workspace_id, j.job_id) IS UNIQUE;
CREATE CONSTRAINT job_request_unique IF NOT EXISTS FOR (j:IngestionJob) REQUIRE (j.workspace_id, j.request_key) IS UNIQUE;
CREATE CONSTRAINT inference_gate_unique IF NOT EXISTS FOR (g:InferenceGate) REQUIRE g.endpoint_id IS UNIQUE;
CREATE INDEX entity_name_lookup IF NOT EXISTS FOR (e:Entity) ON (e.workspace_id, e.kind, e.normalized_name);
CREATE INDEX job_schedule_lookup IF NOT EXISTS FOR (j:IngestionJob) ON (j.status, j.available_at);
CREATE FULLTEXT INDEX chunk_text_v1 IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text];
CREATE FULLTEXT INDEX entity_names_v1 IF NOT EXISTS FOR (e:Entity) ON EACH [e.canonical_name, e.aliases];
CREATE VECTOR INDEX chunk_embedding_v1 IF NOT EXISTS FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};
