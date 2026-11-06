// Uniqueness of requests belongs to IdempotencyRecord (architecture 7.2). Once a record expires,
// the same key is a new operation and must not collide with an old job that still exists.
DROP CONSTRAINT job_request_unique IF EXISTS;
