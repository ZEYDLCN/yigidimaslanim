import json

import httpx

from notes_graphrag.core.config import Settings
from notes_graphrag.providers.capability_probes import FIXTURE_DOCS, run_probes
from notes_graphrag.providers.lmstudio_provider import LMStudioProvider

S = Settings(
    _env_file=None,
    extraction_model="chat-m",
    answer_model="chat-m",
    embedding_model="emb-m",
    embedding_dimensions=4,
)


def vec_for(text: str) -> list[float]:
    # Toy embedding: the Neo4j/database passage and the question share axis 0.
    t = text.lower()
    return [
        1.0 if ("database" in t or "neo4j" in t) else 0.0,
        1.0 if "recipe" in t else 0.0,
        1.0 if "yağmur" in t else 0.0,
        0.1,
    ]


def make(handler_overrides=None):
    over = handler_overrides or {}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path in over:
            return over[path](req)
        if path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "chat-m"}, {"id": "emb-m"}]})
        body = json.loads(req.content)
        if path.endswith("/chat/completions"):
            content = '{"ready": true}' if "response_format" in body else "ok"
            return httpx.Response(
                200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
            )
        return httpx.Response(
            200,
            json={
                "data": [{"index": i, "embedding": vec_for(t)} for i, t in enumerate(body["input"])]
            },
        )

    return LMStudioProvider(
        base_url="http://x/v1",
        embedding_model="emb-m",
        embedding_dimensions=4,
        transport=httpx.MockTransport(handler),
    )


def by_name(results):
    return {r.name: r for r in results}


def test_all_probes_pass():
    res = run_probes(make(), S, query_template="{question}")
    assert all(r.ok for r in res) and len(res) == 5


def test_missing_model_stops_early_with_code():
    s = Settings(
        _env_file=None, extraction_model="nope", answer_model="nope", embedding_model="emb-m"
    )
    res = run_probes(make(), s)
    assert len(res) == 1 and res[0].code == "MODEL_MISSING"


def test_specific_failure_codes():
    trunc = {
        "/v1/chat/completions": lambda r: httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ready":'}, "finish_reason": "length"}]},
        )
    }
    r = by_name(run_probes(make(trunc), S, query_template="{question}"))
    assert r["structured_output"].code == "OUTPUT_TRUNCATED"

    auth = {"/v1/embeddings": lambda r: httpx.Response(401)}
    assert by_name(run_probes(make(auth), S))["embeddings"].code == "AUTH_REJECTED"

    bad_dim = {
        "/v1/embeddings": lambda r: httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [1.0, 0.0]}
                    for i in range(len(json.loads(r.content)["input"]))
                ]
            },
        )
    }
    assert by_name(run_probes(make(bad_dim), S))["embeddings"].code == "DIMENSION_MISMATCH"

    def wrong_rank(r):  # every text maps to the same vector -> expected passage not uniquely first
        n = len(json.loads(r.content)["input"])
        vecs = [[0.0, 0.0, 1.0, 0.1]] * n
        if n == 1:
            vecs = [[0.0, 1.0, 0.0, 0.1]]
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]}
        )

    res = by_name(run_probes(make({"/v1/embeddings": wrong_rank}), S, query_template="{question}"))
    assert res["retrieval_fixture"].code == "RETRIEVAL_FIXTURE_FAILED"
    assert FIXTURE_DOCS


def test_unreachable_endpoint():
    def down(req):
        raise httpx.ConnectError("refused")

    p = LMStudioProvider(base_url="http://x/v1", transport=httpx.MockTransport(down))
    res = run_probes(p, S)
    assert res[0].code == "ENDPOINT_UNREACHABLE"
