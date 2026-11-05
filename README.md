# Notes → Knowledge Graph (GraphRAG + Neo4j + LM Studio)

Mimari: bkz. mimari belge (Bölüm 1–21). Geliştirme fazları: `docs/DEVELOPMENT_PLAN.md`.

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest
```

## Çalıştırma (demo: LM Studio ve Neo4j gerekmez)

```bash
uv sync --group dev --group ui
uv run python -m notes_graphrag.cli serve --demo          # API + süreç içi worker, 127.0.0.1:8000
API_BASE_URL=http://127.0.0.1:8000 uv run streamlit run ui/app.py
uv run python scripts/smoke.py --api-base http://127.0.0.1:8000
```

Gerçek LM Studio ile: `.env` doldurup `uv run python scripts/check_lmstudio.py` (capability probe'ları),
ardından `serve` komutunu `--demo` olmadan çalıştırın. Depo **bellek içi**dir: süreç kapanınca veri gider.

## Durum (dürüst özet)

Çalışan ve test edilen: ayarlar, ontoloji, durum makineleri, doğrulama, chunker, extraction
doğrulaması, entity resolution, RRF, LM Studio HTTP adaptörü + capability probe'ları (sahte HTTP
taşıması ile), bellek içi referans depo (idempotency, lease fencing, atomik yayın, silme), worker,
varlık bağlama, sınırlı graf genişletme, kanıt paketleme, atıf doğrulamalı yanıt, FastAPI uçları
(hata zarfı, Idempotency-Key, sayfalama, yükleme limitleri), Streamlit arayüzü (Notes/Ask/Graph/System),
gerçek HTTP üzerinden smoke testi (demo sağlayıcılarla).

Henüz yapılmayan: Neo4j repository (migrasyon dosyası yazıldı, uygulanmadı), ayrı worker süreci,
`InferenceGate`, canlı LM Studio testleri (bu ortamda LM Studio yok), değerlendirme seti/baseline'lar,
yedekleme/geri yükleme. Demo modundaki çıkarıcı/yanıtlayıcı kural tabanlı sahtedir; model kalitesi
hakkında hiçbir şey kanıtlamaz.
