import json

import httpx
import pytest
from pydantic import SecretStr

from notes_graphrag.core.config import Settings
from notes_graphrag.core.errors import PayloadTooLarge, ProviderBadResponse, ValidationFailed
from notes_graphrag.domain.ontology import EntityKind, predicate_allows
from notes_graphrag.ingestion.chunking import chunk_text
from notes_graphrag.ingestion.entity_resolution import KnownEntity, normalize_name, resolve
from notes_graphrag.ingestion.extraction import validate_extraction
from notes_graphrag.ingestion.validation import decode_text, validate_filename
from notes_graphrag.providers.lmstudio_provider import LMStudioProvider
from notes_graphrag.retrieval.fusion import overfetch, rrf
from notes_graphrag.retrieval.lexical import escape_lucene

S = Settings(_env_file=None)
NOTE = "Ada works on Project Atlas. Project Atlas uses Neo4j."


def test_settings_reject_bad_context():
    with pytest.raises(ValueError):
        Settings(_env_file=None, max_context_tokens=8000)


def test_decode_rules():
    assert decode_text(b"a\r\nb", S) == "a\nb"
    with pytest.raises(ValidationFailed):
        decode_text(b"\xff\xfe", S)
    with pytest.raises(ValidationFailed):
        decode_text(b"  \n", S)
    with pytest.raises(PayloadTooLarge):
        decode_text(b"x" * (S.max_note_bytes + 1), S)
    with pytest.raises(ValidationFailed):
        validate_filename("a.pdf")


@pytest.mark.parametrize(
    "text", [NOTE, "Çalışıyor. İstanbul 😀 ofisi.\n\nYeni paragraf. " * 400, "x" * 9000]
)
def test_chunks_are_exact_slices_and_deterministic(text):
    a = chunk_text(text, "rev1", max_chunks=1000)
    assert a == chunk_text(text, "rev1", max_chunks=1000)
    for c in a:
        assert text[c.start_char : c.end_char] == c.text
    assert a[0].start_char == 0 and a[-1].end_char == len(text.rstrip())
    assert all(a[i].start_char < a[i + 1].start_char for i in range(len(a) - 1))
    assert all(a[i + 1].start_char <= a[i].end_char for i in range(len(a) - 1))  # no gaps


def test_chunk_limit():
    with pytest.raises(PayloadTooLarge):
        chunk_text("word. " * 5000, "r", max_chunks=2)


def test_turkish_normalization_keeps_distinct():
    assert normalize_name("I") != normalize_name("İ")
    assert normalize_name("ı") != normalize_name("i")
    assert normalize_name("  Project   Atlas ") == "project atlas"


def test_ontology_direction_and_kinds():
    assert predicate_allows("USES", EntityKind.PROJECT, EntityKind.TECHNOLOGY)
    assert not predicate_allows("USES", EntityKind.TECHNOLOGY, EntityKind.PROJECT)
    assert not predicate_allows("PART_OF", EntityKind.PROJECT, EntityKind.CONCEPT)
    assert not predicate_allows("NOPE", EntityKind.PROJECT, EntityKind.PROJECT)


def _out(**over):
    claim = {
        "subject_local_id": "e1",
        "predicate": "WORKS_ON",
        "object_local_id": "e2",
        "evidence": {"start": 0, "end": 27, "quote": NOTE[:27]},
    }
    claim.update(over)
    return json.dumps(
        {
            "entities": [
                {"local_id": "e1", "name": "Ada", "kind": "Person"},
                {"local_id": "e2", "name": "Project Atlas", "kind": "Project"},
            ],
            "claims": [claim],
        }
    )


def _v(raw, start=0, chunk=NOTE):
    return validate_extraction(
        raw, revision_id="r", revision_text=NOTE, chunk_start=start, chunk_text=chunk
    )


def test_extraction_accepts_and_offsets_global():
    r = _v(_out())
    assert r.ok and (r.claims[0].evidence_start, r.claims[0].evidence_end) == (0, 27)
    # chunk-local -> global conversion
    chunk = NOTE[28:]
    raw = _out(subject_local_id="e1")
    raw = json.dumps(
        {
            "entities": [
                {"local_id": "e1", "name": "Project Atlas", "kind": "Project"},
                {"local_id": "e2", "name": "Neo4j", "kind": "Technology"},
            ],
            "claims": [
                {
                    "subject_local_id": "e1",
                    "predicate": "USES",
                    "object_local_id": "e2",
                    "evidence": {"start": 0, "end": 25, "quote": chunk[:25]},
                }
            ],
        }
    )
    r = _v(raw, 28, chunk)
    assert r.ok and r.claims[0].evidence_start == 28


@pytest.mark.parametrize(
    "over",
    [
        {"predicate": "USES"},  # incompatible kinds
        {"object_local_id": "zz"},  # unknown entity
        {"evidence": {"start": 0, "end": 27, "quote": "Ada works on Project Atlaz."}},
        {"evidence": {"start": 0, "end": 999, "quote": "x"}},
    ],
)
def test_extraction_rejects(over):
    assert not _v(_out(**over)).ok


def test_extraction_rejects_bad_schema_and_dup_ids():
    assert not _v("not json").ok
    assert not _v(
        json.dumps(
            {
                "entities": [
                    {"local_id": "a", "name": "A", "kind": "Person"},
                    {"local_id": "a", "name": "B", "kind": "Person"},
                ]
            }
        )
    ).ok
    assert _v(json.dumps({"entities": [], "claims": []})).ok  # zero claims allowed


def test_entity_resolution_conservative():
    ada = KnownEntity("1", EntityKind.PERSON, "Ada", context={"project atlas"})
    ada2 = KnownEntity("2", EntityKind.PERSON, "Ada", context={"project zeta"})
    assert resolve("Ada", EntityKind.PERSON, set(), [ada]) is None  # single token, no context
    assert resolve("Ada", EntityKind.PERSON, {"Project Atlas"}, [ada, ada2]) == "1"
    assert resolve("Ada", EntityKind.PERSON, {"Other"}, [ada, ada2]) is None
    full = KnownEntity("3", EntityKind.PERSON, "Ada Lovelace")
    assert resolve("ada  lovelace", EntityKind.PERSON, set(), [full]) == "3"
    assert resolve("Ada Lovelace", EntityKind.ORGANIZATION, set(), [full]) is None


def test_rrf_and_overfetch():
    assert rrf([["a", "b"], ["b", "c"]])[0] == "b"
    hits, limited = overfetch(
        lambda n: [str(i) for i in range(n)], lambda h: int(h) % 100 == 0, want=3
    )
    assert limited and hits == ["0", "100"]
    hits, limited = overfetch(lambda n: [str(i) for i in range(n)], lambda h: True, want=5)
    assert not limited and len(hits) == 5


def test_lucene_escape():
    assert escape_lucene("a:b (c) AND d") == r'a\:b \(c\) "AND" d'


def _provider(handler, key=None):
    return LMStudioProvider(
        base_url="http://x/v1",
        api_key=key,
        embedding_model="e",
        embedding_dimensions=3,
        transport=httpx.MockTransport(handler),
    )


def test_embeddings_validation_and_no_auth_header():
    seen = {}

    def ok(req):
        seen["auth"] = req.headers.get("authorization")
        n = len(json.loads(req.content)["input"])
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [3.0, 4.0, 0.0]} for i in reversed(range(n))]},
        )

    p = _provider(ok)
    vecs = p.embed(["a"] * 10)  # 2 batches
    assert len(vecs) == 10 and vecs[0] == pytest.approx([0.6, 0.8, 0.0])
    assert seen["auth"] is None
    _provider(ok, SecretStr("tok")).embed(["a"])
    assert seen["auth"] == "Bearer tok"

    bad_dim = _provider(
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    )
    with pytest.raises(ProviderBadResponse):
        bad_dim.embed(["a"])
    zero = _provider(
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0, 0, 0]}]})
    )
    with pytest.raises(ProviderBadResponse):
        zero.embed(["a"])


def test_chat_rejects_truncation_and_non_json():
    def reply(content, finish="stop"):
        return lambda r: httpx.Response(
            200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]}
        )

    kw = dict(model="m", system="s", user="u", schema_name="n", schema={}, max_tokens=10)
    assert _provider(reply('{"ready": true}')).chat_json(**kw) == '{"ready": true}'
    for bad in (reply('{"a":', "length"), reply("```json\n{}\n```"), reply("")):
        with pytest.raises(ProviderBadResponse):
            _provider(bad).chat_json(**kw)
    assert _provider(reply("{}")) is not None
