# Evaluation set

`notes.jsonl` (9 notes, English and Turkish) and `questions.jsonl` (16 questions: 11 answerable,
5 unanswerable). Each answerable question lists the notes that must appear in its evidence.
`hops` marks questions that need a connection through a shared entity or a second note.

## What the runner measures

`scripts/evaluate.py` ingests the notes and asks every question with three retrieval modes
(vector only, hybrid lexical+vector, hybrid+graph). It reports:

- evidence recall: share of required notes present in the evidence pack (answerable questions);
- abstention accuracy: share of unanswerable questions answered with `insufficient_evidence`;
- answer status accuracy and citation validity from the end-to-end answer path;
- retrieval latency p50/p95.

## Important limits

The default run uses the **demo providers** (rule-based fakes in `providers/fakes.py`). Those
results validate the harness, lifecycle and retrieval plumbing. They do **not** measure extraction
quality, Turkish language understanding, or model answer quality. Architecture section 16 targets
(precision, groundedness, latency on a GPU) require a live LM Studio run, which is not part of this
repository's CI and was not executed in the development environment.

The set is intentionally small. With 16 questions, one question changes a rate by 6 percentage
points; report counts alongside percentages.
