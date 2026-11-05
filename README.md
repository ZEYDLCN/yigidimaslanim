# Notes → Knowledge Graph (GraphRAG + Neo4j + LM Studio)

Mimari: bkz. mimari belge (Bölüm 1–21). Geliştirme fazları: `docs/DEVELOPMENT_PLAN.md`.

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest
```

## Durum (dürüst özet)

Çalışan ve test edilen: ayarlar, ontoloji, durum makineleri, doğrulama, chunker, extraction
doğrulaması, entity resolution, RRF, LM Studio HTTP adaptörü (mock transport ile), bellek içi
referans depo (idempotency, lease fencing, atomik yayın, silme), worker.

Henüz çalıştırılmayan/yazılmayan: Neo4j repository (migrasyon dosyası yazıldı, uygulanmadı),
FastAPI, Streamlit, graf genişletme, yanıt üretimi, InferenceGate, canlı LM Studio testleri.
