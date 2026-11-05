"""Run the evaluation set against the retrieval modes (architecture section 16).

Usage: uv run python scripts/evaluate.py [--out reports/evaluation]

Uses the demo providers unless --live is given. Live mode needs LM Studio configured in .env
and is not run by CI.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notes_graphrag.answering.service import AnswerService  # noqa: E402
from notes_graphrag.core.config import Settings  # noqa: E402
from notes_graphrag.ingestion.service import NoteService  # noqa: E402
from notes_graphrag.ingestion.worker import Worker  # noqa: E402
from notes_graphrag.persistence.memory import MemoryStore  # noqa: E402
from notes_graphrag.persistence.memory_source import MemoryKnowledge  # noqa: E402
from notes_graphrag.providers.fakes import DemoChat, FakeEmbeddings  # noqa: E402
from notes_graphrag.retrieval.service import Retriever  # noqa: E402

MODES = ("vector", "hybrid", "graph")
EVAL_IDS: dict[int, dict[str, str]] = {}  # store id -> {note_id: evaluation id}
DIMS, PROFILE = 64, "fake-v1"


@dataclass
class QResult:
    qid: str
    answerable: bool
    expected: list[str]
    found: list[str]
    status: str
    citations_valid: bool
    latency_ms: float
    extra: dict = field(default_factory=dict)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def build(
    notes: list[dict],
) -> tuple[MemoryStore, MemoryKnowledge, Settings, NoteService, DemoChat]:
    s = Settings(_env_file=None, embedding_dimensions=DIMS, embedding_profile=PROFILE)
    store = MemoryStore()
    know = MemoryKnowledge(store)
    svc = NoteService(store, s)
    chat = DemoChat()
    worker = Worker(store, s, chat, FakeEmbeddings(DIMS, PROFILE))
    ids: dict[str, str] = {}
    for n in notes:
        sub = svc.create(n["title"], n["text"], f"eval-{n['id']}")
        ids[sub.note.note_id] = n["id"]
    while worker.run_once():
        pass
    EVAL_IDS[id(store)] = ids
    return store, know, s, svc, chat


def run_mode(mode: str, notes: list[dict], questions: list[dict]) -> dict:
    store, know, s, _, chat = build(notes)
    retriever = Retriever(know, FakeEmbeddings(DIMS, PROFILE), s, mode=mode)
    qa = AnswerService(know, retriever, chat, s)
    results: list[QResult] = []
    for q in questions:
        t0 = time.perf_counter()
        r = retriever.retrieve(q["question"])
        latency = (time.perf_counter() - t0) * 1000
        found_notes = sorted({_note_of(store, i.revision_id) for i in r.pack.items})
        ans = qa.ask(q["question"], include_graph=False)
        valid = all(
            store.get_revision(c.revision_id) is not None
            and store.get_revision(c.revision_id).text[c.start_char : c.end_char] == c.quote
            for c in ans.citations
        )
        results.append(
            QResult(
                q["id"],
                q["answerable"],
                q["expected_notes"],
                found_notes,
                ans.status,
                valid,
                latency,
                {"hops": q["hops"], "lang": q["lang"]},
            )
        )
    return summarise(mode, results)


def _note_of(store: MemoryStore, revision_id: str) -> str:
    rev = store.get_revision(revision_id)
    assert rev is not None
    return EVAL_IDS[id(store)][rev.note_id]


def summarise(mode: str, rs: list[QResult]) -> dict:
    ans = [r for r in rs if r.answerable]
    unans = [r for r in rs if not r.answerable]
    recall = [len(set(r.expected) & set(r.found)) / len(r.expected) for r in ans]
    lat = sorted(r.latency_ms for r in rs)
    p95_idx = max(0, int(round(0.95 * (len(lat) - 1))))
    hop2 = [r for r in ans if r.extra["hops"] == 2]
    return {
        "mode": mode,
        "counts": {"questions": len(rs), "answerable": len(ans), "unanswerable": len(unans)},
        "evidence_recall_mean": round(statistics.mean(recall), 3),
        "full_evidence_hit": sum(1 for r in ans if set(r.expected) <= set(r.found)),
        "multi_hop_full_coverage": sum(1 for r in hop2 if set(r.expected) <= set(r.found)),
        "multi_hop_total": len(hop2),
        "abstention_correct": sum(1 for r in unans if r.status == "insufficient_evidence"),
        "answered_when_answerable": sum(1 for r in ans if r.status != "insufficient_evidence"),
        "citations_all_valid": all(r.citations_valid for r in rs),
        "latency_ms_p50": round(statistics.median(lat), 2),
        "latency_ms_p95": round(lat[p95_idx], 2),
        "per_question": [r.__dict__ for r in rs],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "evaluation"))
    args = ap.parse_args()
    notes = load(ROOT / "evaluation" / "notes.jsonl")
    questions = load(ROOT / "evaluation" / "questions.jsonl")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = [run_mode(m, notes, questions) for m in MODES]
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print("providers: DEMO (rule-based fakes); results validate plumbing, not model quality")
    print(
        f"{'mode':8} {'recall':>7} {'full':>7} {'hop2':>7} {'abstain':>9} {'answered':>9} "
        f"{'p50ms':>7} {'p95ms':>7}"
    )
    for s in summaries:
        n_un = s["counts"]["unanswerable"]
        n_an = s["counts"]["answerable"]
        print(
            f"{s['mode']:8} {s['evidence_recall_mean']:7.3f} "
            f"{s['full_evidence_hit']:>3}/{n_an:<3} "
            f"{s['multi_hop_full_coverage']:>3}/{s['multi_hop_total']:<3} "
            f"{s['abstention_correct']:>4}/{n_un:<4} "
            f"{s['answered_when_answerable']:>4}/{n_an:<4} "
            f"{s['latency_ms_p50']:7.2f} {s['latency_ms_p95']:7.2f}"
        )
    print(f"wrote {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
