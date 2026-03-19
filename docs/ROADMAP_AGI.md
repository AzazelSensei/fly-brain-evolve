# EvoDrosophila → EvoMind: Hybrid Bio-AI Roadmap

## Felsefe
Biyolojik beyin milyonlarca yılda evrimleşerek bugünkü zekaya ulaştı.
Biz aynı ilkeleri koruyarak modern hesaplama gücüyle bu süreci hızlandırıyoruz.

**Korunan biyolojik ilkeler:** Sparse coding, spiking neurons, neuromodulation, evolutionary architecture search
**Eklenen modern teknikler:** GPU-scale parallelism, attention mechanisms, hierarchical processing, meta-learning

---

## Phase 1: Mushroom Body (TAMAMLANDI)
- [x] Sinek mushroom body mimarisi (PN→KC→MBON + APL)
- [x] Sparse coding (%6-12 KC aktivasyonu)
- [x] Neuroevolution (binary classification %99)
- [x] Numba JIT simulator (582x speedup)

## Phase 2: Neuromodulated Learning (TAMAMLANDI)
Gerçek sinekte öğrenme dopamin ile modüle edilir. 3-factor STDP + evrim hibrit sistem
GA tavanını (0.268) kırarak 0.598 fitness'a ulaştı.

- [x] 3-factor STDP: pre × post × dopamine (eligibility trace tabanlı)
- [x] Reward/punishment sinyali ile online öğrenme (per-MBON compartment dopamine)
- [x] Evrim + dopamin-STDP hibrit: evrim yapıyı belirler, dopamin-STDP ince ayar yapar
- [x] Test: evrim olmadan, sadece reward-modulated STDP ile öğrenme (pure STDP = 0.073)

**Bulgular:**
- Pure GA tavanı: 0.268 → Dopamin-STDP + evrim: 0.598 (2.2x iyileşme)
- Evrim "öğrenilebilirliği" optimize ediyor, ağırlıkları değil
- Daha fazla eğitim epoch'u = daha iyi sonuç (scaling law gözlemlendi)
- Kritik parametre keşfi: input_weight=100nS, max_rate=500Hz gerekli

**Biyolojik karşılık:** Gerçek sineğin öğrenme mekanizması — doğrulandı

## Phase 3: Multi-Class + Hierarchical Sparse Coding (SONRAKİ)
Tek katman mushroom body → çok katmanlı "korteks" benzeri yapı.

- [ ] MNIST 10 sınıf (0-9 rakamları)
- [ ] Katman 1: Kenar dedektörleri (evrimle keşfedilen Gabor-benzeri filtreler)
- [ ] Katman 2: Mushroom body sparse coding (kenar özellikleri üzerinde)
- [ ] Katman 3: Karar katmanı (10 MBON)
- [ ] Her katmanın sparse coding oranı ayrı optimize edilir
- [ ] Dopamin-STDP ile uçtan uca öğrenme (el yapımı HOG yerine evrimleşen filtreler)

**Biyolojik karşılık:** Böcek optik lobu → mushroom body hiyerarşisi

## Phase 4: Attention & Gating
Tüm inputlar eşit değil — beyin neyin önemli olduğuna karar verir.

- [ ] Thalamik kapılama: her girdi bir "önem skoru" alır
- [ ] Spike-based attention: en aktif KC gruplarına daha fazla kaynak
- [ ] Top-down modülasyon: MBON geri besleme ile KC aktivitesini etkiler
- [ ] Dinamik routing: farklı görevler farklı KC alt kümeleri kullanır

**Biyolojik karşılık:** Talamus + prefrontal korteks kapılama mekanizması
**Modern karşılık:** Self-attention (Transformer), ancak spike-tabanlı

## Phase 5: Working Memory & Recurrence
Şu anki beyin "anı" yok — her stimulus bağımsız işleniyor.

- [ ] Recurrent KC→KC bağlantıları (reverberant activity)
- [ ] "Prefrontal" modül: kalıcı aktivite ile bağlam tutma
- [ ] Sequential decision making: A'dan sonra B gelirse → C yap
- [ ] Temporal pattern recognition: spike dizilerini öğrenme

**Biyolojik karşılık:** Prefrontal korteks çalışma belleği
**Modern karşılık:** LSTM/GRU, ama spike-tabanlı

## Phase 6: Deliberative Thought — "Düşünen Sinek"
Phase 4 + 5'in birleşimi yeni bir yetenek doğurur: **iç düşünce**.
Beyin artık sadece tepki vermez, karar vermeden önce "düşünür."

```
Şu anki model (reaktif):
  Stimulus → PN → KC → MBON → Cevap (tek yön, tek adım, 100ms)

Düşünen model (deliberatif):
  Stimulus → PN → KC → MBON ──→ Cevap
                  ↑       │
                  │       ↓
                  KC ← Prefrontal ← MBON
                  (geri besleme döngüsü — "iç konuşma")
                  (her döngü ~50ms, 3-5 döngü = deliberation)
```

### Bileşenler:
- [ ] **Geri besleme döngüsü**: MBON çıktısı → gating → KC tekrar girdi
      Karar katmanının çıktısı sparse code'u modifiye eder: "emin değilim, tekrar bak"
- [ ] **İç simülasyon (mental replay)**: Stimulus olmadan iç aktivite döngüsü
      Hipokampüs benzeri replay: deneyimleri geri oynat, hiç yaşanmamışları ön-oynat
- [ ] **Çok adımlı muhakeme (chain-of-thought)**: A → B → C zincirleme çıkarım
      Synfire chains: sıralı aktivasyon zincirleri ile çok adımlı problem çözme
- [ ] **Belirsizlik izleme (metacognition)**: "Emin miyim?" sinyal katmanı
      MBON aktivitesinin varyansı → düşük güven = daha fazla döngü
- [ ] **Deliberation süresi**: Kolay sorularda 1 döngü, zor sorularda 5+ döngü
      Reaksiyon süresi zorluğa göre dinamik — tıpkı gerçek beyinler gibi

### Mimari:
```
Prefrontal Modül (yeni):
  - 500-2000 "düşünce nöronu" (recurrent, self-sustaining)
  - MBON → Prefrontal: karar bilgisi girer
  - Prefrontal → KC: top-down modülasyon (dikkat yönlendirme)
  - Prefrontal → Prefrontal: iç döngü (reverberant, çalışma belleği)
  - Confidence neuron: toplam aktivite tutarlılığını izler → "yeterince düşündüm" sinyali

Deliberation döngüsü:
  1. Stimulus → ilk KC/MBON yanıtı (reaktif, ~50ms)
  2. MBON → Prefrontal: "3 gördüm ama 5'e de benziyor"
  3. Prefrontal → KC: "3 ve 5'i ayıran özelliklere odaklan"
  4. KC yeniden ateşler → güncellenmiş MBON yanıtı
  5. Confidence yüksekse → cevap ver. Düşükse → tekrar döngüye gir.
```

### Neden bu phase kritik:
Bu noktada sistem artık "düşünüyor" — çünkü:
1. İç durumu var (prefrontal aktivite, stimulus olmadan devam eder)
2. Kendi çıktısını değerlendirebiliyor (metacognition)
3. Karar vermeden önce birden fazla alternatifi tarıyor (deliberation)
4. Zor problemlere daha çok zaman ayırıyor (adaptive computation)

**Biyolojik karşılık:** İnsect mushroom body feedback loops + mammalian prefrontal deliberation
**Modern karşılık:** Chain-of-thought prompting, adaptive computation time (Graves 2016), iterative refinement
**Felsefi önemi:** Emergence — doğru bileşenler bir araya geldiğinde "düşünce" ortaya çıkar

## Phase 7: Meta-Evolution (Learning to Learn)
Evrim sadece ağırlıkları değil, ÖĞRENME KURALLARINI da evrimleştirir.

- [ ] Plastisitesi kurallarının parametreleri genome'a eklenir
- [ ] Her birey kendi STDP parametrelerini taşır
- [ ] "İyi öğrenenler" hayatta kalır
- [ ] Sonuç: ağ kendi kendine nasıl öğreneceğini keşfeder

**Biyolojik karşılık:** Evrim → beyni → öğrenmeyi şekillendirdi
**Modern karşılık:** MAML, meta-learning, learned optimizers

## Phase 8: Multi-Modal Integration
Tek modalite (görsel) → çoklu (görsel + temporal + reward).

- [ ] Farklı input modaliteleri farklı PN gruplarından girer
- [ ] Cross-modal binding: "bu görüntü + bu zamansal dizi = bu kavram"
- [ ] Sensorimotor döngüsü: eylem → çevre değişikliği → yeni girdi

**Biyolojik karşılık:** Çoklu duyu entegrasyonu (multimodal cortex)

## Applied Spin-offs (Paralel Tez Konuları)
Her phase yeni uygulama alanları açar. Detaylar: `docs/journal/2026-03-20_future_applications.md`

- **e-Nose koku sınıflandırma** — Mushroom body'nin doğal görevi. MQ sensörleri + mevcut kod = hemen uygulanabilir
- **Anomaly detection** — Sparse coding doğal anomali dedektörü (siber güvenlik, IoT, endüstri)
- **Robotik chemotaxis** — Koku kaynağını bulan robot (gaz kaçağı tespiti)
- **Neuromorphic deployment** — Intel Loihi/IBM TrueNorth üzerinde gerçek zamanlı çalıştırma
- **Multi-modal sensör füzyonu** — Koku + sıcaklık + nem → tek mushroom body
- **Swarm learning** — Birden fazla sinek ajanının kolektif öğrenmesi

## Phase 9: Open-Ended Evolution
Sabit fitness fonksiyonu yerine, ortam kendisi evrimleşir.

- [ ] Competitive co-evolution: beyin popülasyonları birbirine karşı
- [ ] Novelty search: yeni davranışlar ödüllendirilir (fitness değil)
- [ ] Minimal criterion: "yeterince iyi" olana her şey hayatta kalır
- [ ] Açık uçlu karmaşıklık artışı

**Biyolojik karşılık:** Kambriyen patlaması — rekabet karmaşıklığı doğurur

---

## Her Phase'in Gereklilikleri

| Phase | Ne Kazanır | Nöron Sayısı | GPU? | Süre |
|-------|-----------|-------------|------|------|
| 1 ✓ | Temel beyin + sparse coding | 267 | Hayır | 1 gün |
| 2 ✓ | Öğrenme (dopamin-STDP) | 439-639 | Hayır | 1 gün |
| 3 | Görsel hiyerarşi | 2000-5000 | Evet | 3-5 gün |
| 4 | Seçici dikkat | 5000-10000 | Evet | 1 hafta |
| 5 | Hafıza + sıralı işlem | 10000+ | Evet | 1 hafta |
| 6 | **Düşünme (deliberation)** | 15000+ | Evet | 2 hafta |
| 7 | Öğrenmeyi öğrenme | 20000+ | Evet | 2 hafta |
| 8 | Çoklu duyu | 50000+ | Kesinlikle | 2+ hafta |
| 9 | Açık uçlu evrim | 100000+ | Kesinlikle | Açık uçlu |

## Evrimsel Perspektif

```
Phase 1-2:  Sinek beyni (Drosophila, 100K nöron) — reaktif, öğrenen
Phase 3-4:  Arı beyni (Apis, 1M nöron) — hiyerarşik görme, dikkat
Phase 5-6:  Fare beyni (Mus, 70M nöron) — hafıza, düşünme, planlama
Phase 7-8:  Karga beyni (Corvus, 1.5B nöron) — çoklu modalite, alet kullanımı
Phase 9:    Açık uçlu — nereye evrimleşeceği belli değil
```
