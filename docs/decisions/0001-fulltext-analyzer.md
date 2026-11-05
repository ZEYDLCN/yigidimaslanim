# 0001: Full-text analyzer for Turkish and English notes

Status: accepted for the MVP. Measured on Neo4j 5.26.32 Community, 2026-11-05.

## Context

Architecture section 5.5 requires choosing and testing a full-text analyzer for Turkish and English,
then recording it in the migration. Changing an analyzer requires an index rebuild.

## What was measured

Index: `CREATE FULLTEXT INDEX ... ON EACH [n.t] OPTIONS {indexConfig: {`fulltext.analyzer`: ...}}`
over two sample nodes: "İstanbul ofisinde Ada çalışıyor. Ada Project Atlas üzerinde çalışır." and a
second Turkish sentence. Hit counts per query (`standard-no-stop-words`):

| Query | Hits | Reading |
| --- | --- | --- |
| `istanbul` | 1 | lowercase query matches capital dotted İ |
| `İstanbul` | 1 | exact form matches |
| `ada` | 1 | English-style token |
| `ofisinde` | 1 | exact inflected form |
| `ofis` | 0 | no stemming: the base form does not match the inflected token |
| `çalışıyor` | 1 | exact Turkish token with diacritics |
| `calisiyor` | 0 | no ASCII folding in this analyzer |

The `turkish` and `standard-folding` analyzers could not be created in the probe (the probe script
failed to create them), so they were not compared. This is a gap, not a result.

## Decision

Use `standard-no-stop-words` for both full-text indexes. It handles the dotted-I case correctly and
keeps mixed Turkish and English notes in one index without a language-specific stemmer.

## Consequences and known limits

- No stemming: `backup` does not match `backups`, and `ofis` does not match `ofisinde`. The
  evaluation set shows this on question q07. Query-side and index-side stemming are later changes
  that require a rebuild.
- Query terms are folded in application code (`retrieval/lexical.py: fold_for_search`) so that
  `İSTANBUL` finds `İstanbul`. Entity identity deliberately does not use that folding (architecture 6.4).
- Vector retrieval partly compensates for the missing stemming, but it is only as good as the
  embedding model, which has not been measured here.
