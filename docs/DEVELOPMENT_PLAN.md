# Geliştirme Planı (Fazlar)

Kaynak: mimari belge (Notes → Knowledge Graph: GraphRAG + Neo4j + LM Studio).
Bu plan, mimarideki Bölüm 17 yol haritasını uygulanabilir iş paketlerine böler.
Her faz bir **çıkış kapısı** ile biter; kapı geçmeden sonraki faza geçilmez.

## Ortam kısıtları (bu bulut oturumu)

- LM Studio bu makinede yok. Canlı model testleri (`live_lmstudio`) çalıştırılamaz;
  bunun yerine **deterministik sahte sağlayıcılar** ve HTTP-sözleşme testleri (mock transport) kullanılır.
- Neo4j için Docker gerekir; entegrasyon testleri Neo4j erişilebilirse çalışır, değilse `skip` edilir ve
  raporda "çalıştırılmadı" olarak belirtilir (başarılı sayılmaz).
- Mimari ilke: önce dikey dilim (sahte sağlayıcı ile), sonra gerçek LM Studio adaptörü.

## Fazlar

| Faz | İçerik | Çıkış kapısı | Durum |
| --- | --- | --- | --- |
| 0 | Paket iskeleti, `pyproject`, tipli ayarlar, hatalar, loglama, domain modelleri, ontoloji, durum makineleri, `.env.example`, Compose, CI iskeleti | `ruff`, `mypy`, `pytest` yeşil | Bu turda |
| 1 | Giriş doğrulama (UTF-8, limitler), idempotency anahtarları, not/revizyon/iş servisleri, Neo4j migrasyon + repository (veya bellek içi repo) | Create/update/delete + idempotency testleri | Bu turda (bellek içi repo + Neo4j migrasyon dosyası) |
| 2 | Deterministik chunker (tam dilim, kod noktası ofsetleri), sahte extraction/embedding, worker iş döngüsü, lease fencing, atomik yayın | Worker yalnızca tam revizyonu yayınlar | Bu turda (bellek içi) |
| 3 | LM Studio HTTPX adaptörü, capability probe'ları, extraction şeması + doğrulama sırası, entity resolution | Doğrulama/ontoloji birim testleri; canlı test sonra | Bu turda (adaptör + doğrulama) |
| 4 | Arama: RRF, lexical/vector aday birleştirme, aktif revizyon filtresi | Eski/silinmiş kanıt asla dönmez | Bu turda (RRF + filtre mantığı) |
| 5 | Graf genişletme (2 semantik sıçrama, limitler), kanıt paketleme, token bütçesi | Çok-notlu 2-hop fikstürü | Sonraki tur |
| 6 | Yanıt üretimi, atıf doğrulama, abstention, graph_version kontrolü | Çalışan örnek soru-cevap | Sonraki tur |
| 7 | FastAPI uçları + Streamlit arayüzü (Notes/Ask/Graph/System) | Uçtan uca kullanıcı akışı | Sonraki tur |
| 8 | Güvenilirlik: InferenceGate, yeniden deneme bütçeleri, restart kurtarma, yedekleme | Hata enjeksiyon testleri | Sonraki tur |
| 9 | Değerlendirme seti, baseline'lar, canlı smoke, operasyon rehberi | Raporlanmış sonuçlar | Sonraki tur |

## Bu turun teslimatı

Faz 0 tamamı; Faz 1–4'ün **saf-Python çekirdeği** (Neo4j'den bağımsız test edilebilir mantık) ve
Neo4j migrasyonu. Dürüst raporlama: yazılan ama çalıştırılmayan hiçbir şey "test edildi" sayılmaz.
