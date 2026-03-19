# Future Applications: EvoDrosophila Beyond MNIST

**Date:** 2026-03-20
**Type:** Application roadmap / Thesis spin-off ideas

---

## Core Insight

The mushroom body is an olfactory circuit — we forced it into vision.
Returning it to its natural domain (chemical sensing) would be easier AND more impactful.

---

## Application 1: Electronic Nose (e-nose) — En Doğal Uygulama

**Neden uygun:** PN→KC→MBON tam olarak koku işleme devresi. 8-32 sensör kanalı,
128 HOG özelliğinden çok daha düşük boyutlu. Sparse coding koku için ideal.

**Donanım:**
- Raspberry Pi / Arduino
- MQ serisi gaz sensörleri (MQ-2, MQ-3, MQ-135 vb.) — tanesi 2-3 USD
- Servo motor (yönelim çıkışı için)

**Yazılım:** Mevcut `dopamine_stdp.py` aynen çalışır. HOG yerine sensör okumaları girer.

**Mimari:**
```
Gaz sensör dizisi (8-32 kanal)
    → Poisson spike encoding (sensör değeri = firing rate)
    → 8-32 PN → 200-500 KC → N MBON
    → Dopamin-STDP ile online öğrenme
    → Motor çıkış: yönel / kaç / alarm
```

**Uygulama senaryoları:**

| Senaryo | Sensörler | Sınıf | Zorluk | Tez potansiyeli |
|---------|-----------|-------|--------|----------------|
| Gaz kaçağı tespiti | MQ-2, MQ-4 | 2 (güvenli/tehlike) | Düşük — Phase 1 yeterli | Lisans bitirme |
| Yiyecek kalite kontrolü | MQ-3, MQ-135 | 3-5 (taze/bayat/bozuk) | Orta | Yüksek lisans |
| Parfüm/aroma sınıflandırma | 16-32 kanal dizi | 10-20 sınıf | Yüksek | Yüksek lisans |
| Çevre kirliliği izleme | VOC sensörleri | 5-10 kirletici | Orta-Yüksek | Yüksek lisans |
| Tıbbi nefes analizi | Özel sensör dizi | 2-5 (hasta/sağlıklı) | Çok yüksek | Doktora |

**Avantaj CNN'e göre:** Online öğrenme — yeni koku sınıfı eklemek için tüm modeli
yeniden eğitmeye gerek yok. Dopamin sinyali ile anında öğrenir. Edge deployment
için ideal (düşük güç, küçük model).

---

## Application 2: Anomaly Detection — Sineğin "Tehlike Hissi"

**Konsept:** Sinek bilmediği bir koku aldığında kaçar — "novel = tehlikeli" prensibi.
KC sparse coding doğal bir anomali detektörü: bilinen paternler bilinen KC alt kümelerini
aktive eder, bilinmeyen paternler farklı/dağınık aktivasyon yaratır.

**Uygulamalar:**
- Siber güvenlik: network traffic anomaly detection
- Endüstriyel: makine titreşim anomalisi (predictive maintenance)
- Finansal: fraud detection
- IoT: sensör arıza tespiti

**Tez açısı:** "Bio-inspired anomaly detection using sparse coding in spiking neural networks"

---

## Application 3: Robotik Koku Takibi (Chemotaxis)

**Konsept:** Sineğin kokuya doğru uçması — gradient following + öğrenme.

**Mimari:**
```
Sol sensör + Sağ sensör → 2 PN grubu → KC → MBON
                                            ↓
                                    Sol motor / Sağ motor
                                    (Braitenberg vehicle + öğrenme)
```

**Senaryo:** Gaz kaçağı kaynağını bulan robot. Ödül: koku güçlendikçe dopamin.
Phase 5 (working memory) ile kombinle: "son 5 saniyede sola gittim ve koku arttı"
→ sola gitmeye devam et.

**Tez açısı:** "Evolved spiking controllers for chemotactic navigation"

---

## Application 4: Neuromorphic Chip Deployment

**Konsept:** Modelimizi Intel Loihi veya IBM TrueNorth gibi neuromorphic donanıma taşımak.

**Avantaj:** Spiking model zaten native — ANN→SNN dönüşümüne gerek yok.
439-639 nöron Loihi'nin 128K nöronluk kapasitesinin %0.5'i. Gerçek zamanlı,
mikroWatt güç tüketimi.

**Tez açısı:** "Deployment of evolved spiking mushroom body circuits on neuromorphic hardware"

---

## Application 5: Çoklu Sensör Füzyonu (Phase 8 Preview)

**Konsept:** Koku + sıcaklık + nem + ışık → tek bir mushroom body'de birleşim.

**Mimari:**
```
Koku sensörleri (8 kanal)   → PN grubu A (8 nöron)  ↘
Sıcaklık sensörü (1 kanal)  → PN grubu B (4 nöron)   → KC (500) → MBON
Nem sensörü (1 kanal)       → PN grubu C (4 nöron)   ↗
Işık sensörü (1 kanal)      → PN grubu D (4 nöron)  ↗
```

KC sparse coding farklı modaliteleri doğal olarak birleştirir — her KC birden
fazla modaliteden girdi alabilir. "Sıcak + nemli + tatlı koku = çürük meyve"
öğrenimini KC düzeyinde yapar.

**Tez açısı:** "Multi-modal sensory integration in bio-inspired sparse coding networks"

---

## Application 6: Federated / Swarm Learning

**Konsept:** Birden fazla "sinek" aynı ortamda farklı deneyimler yaşar.
Her birinin KC→MBON ağırlıkları farklı evrimleşir. Ağırlıkları birleştirerek
kolektif bilgi oluştur.

**Biyolojik karşılık:** Arı kolonisinde keşifçi arılar farklı çiçekleri ziyaret eder,
dönüşte dans ederek bilgi paylaşır.

**Tez açısı:** "Swarm learning in populations of spiking mushroom body agents"

---

---

## Sensör Gerçekçilik Analizi: e-Nose Uygulanabilirliği

### MQ Sensörlerinin Sınırlamaları

| Sorun | Detay |
|-------|-------|
| Seçicilik düşük | MQ-2 hem metan, hem propan, hem duman algılar — tek başına ayırt edemez |
| Drift | Kalibrasyon zamanla kayar (sıcaklık, nem, sensör yaşlanması) |
| Tepki süresi | 1-10 saniye (yavaş) |
| Tekrarlanabilirlik | Aynı kokuya %10-20 farklı okuma verebilir |
| Çapraz hassasiyet | Her sensör birden fazla gaza tepki verir |

### Neden Mushroom Body Tam Bu Sorunu Çözer

Gerçek sineğin koku reseptörleri de "mükemmel" değil — her reseptör birden fazla
moleküle tepki verir (çapraz hassasiyet). Sinek bunu **kombinatoryal kodlama** ile çözer:

```
Tek sensör:     MQ-2 = "bir gaz var" (hangisi belli değil)
Sensör dizisi:  MQ-2 yüksek + MQ-3 düşük + MQ-135 orta = "metan"
                MQ-2 orta  + MQ-3 yüksek + MQ-135 düşük = "etanol"
```

Sparse coding tam bunu yapar — her KC farklı bir sensör KOMBİNASYONUNA yanıt verir.
Tek sensörün ayırt edemediğini, birden fazla sensörün birleşik deseni ayırt eder.
Bu biyolojinin 500 milyon yıllık çözümü: ucuz, gürültülü reseptörlerden
kombinatoryal kodlama ile maksimum bilgi çıkarmak.

### Sensör Seçenekleri ve Maliyetler

| Sensör Tipi | Fiyat | Hassasiyet | Kullanım Alanı |
|------------|-------|-----------|---------------|
| MQ serisi (MQ-2,3,4,5,6,7,8,135) | 2-5$ / adet | ppm seviye | Hobi, prototip, demo |
| BME680 (Bosch) | 10-15$ | VOC indeks + sıcaklık/nem | IoT, iç mekan kalitesi |
| SGP40 (Sensirion) | 5-8$ | VOC kalite indeksi | Hava kalitesi izleme |
| PID sensör | 200-500$ | ppb seviye (1000x hassas) | Endüstriyel güvenlik |
| Metal oksit dizi (8-16 kanal) | 50-100$ | Patern bazlı ayrım | e-nose araştırma standardı |
| Kütle spektrometresi | 50.000$+ | Molekül bazlı kesin tanı | Laboratuvar referans |

### Maliyet Bazlı Stratejiler

**Düşük maliyet — Demo/Hobi (~25$):**
8 farklı MQ sensör dizisi. Her biri farklı gazlara farklı hassasiyette.
Mushroom body'nin kombinatoryal kodlaması ile 5-10 koku sınıfı ayırt edilebilir.
Gaz kaçağı / taze-bayat yiyecek / alkol tespiti için yeterli.

**Orta maliyet — Araştırma (~50$):**
BME680 + SGP40 + 4-8 MQ sensör. Sıcaklık ve nem kompanzasyonu dahil.
Drift sorununu dopamin-STDP'nin online öğrenme özelliği çözer — sensör
kayarsa sinek anında yeniden öğrenir, klasik ML modeli çöker.

**Yüksek maliyet — Endüstriyel (~100-500$):**
16 kanallı metal oksit sensör dizisi. e-nose araştırmasının standart donanımı.
Bizim avantajımız: CNN yerine spiking mushroom body = anında öğrenme,
yeni koku sınıfı ekleme, edge deployment.

### Mushroom Body'nin Rakiplere Karşı 3 Ezici Avantajı

**1. Online öğrenme (CNN'de yok)**
Yeni koku sınıfı eklemek: birkaç örnek göster + dopamin sinyali ver = bitti.
CNN'i baştan yeniden eğitmek gerekir (saatler/günler).
Mushroom body anında öğrenir (saniyeler).

**2. Drift adaptasyonu (Klasik ML'de yok)**
Sensör zamanla kayar (sıcaklık, nem, yaşlanma). Klasik ML modeli bu kaymayı
tanımaz ve yanlış sınıflandırmaya başlar — yeniden kalibrasyon gerekir.
Dopamin-STDP sürekli adapte olur: sensör kayarsa, ağırlıklar güncellenir.
Sinek asla "kalibre edilmez" — sürekli öğrenir.

**3. Edge deployment (Bulut modellerde yok)**
639 nöron = mikrokontrolörde çalışır (Arduino, ESP32, Raspberry Pi).
WiFi yok, bulut bağlantısı yok, gizlilik sorunu yok, gerçek zamanlı.
Neuromorphic chip (Intel Loihi) üzerinde mikrowatt güç tüketimi.
Saha koşullarında (fabrika, tarla, maden) bağlantısız çalışır.

### Sonuç: Hassasiyet Yarışını Kazanmaya Gerek Yok

Mesele "en hassas sensör" değil — mesele **ucuz, gürültülü sensörlerden maksimum
bilgiyi çıkarmak**. Sinek beyni 500 milyon yıldır tam da bunu yapıyor.
Pahalı, hassas sensör + basit algoritma yerine, ucuz sensör + akıllı beyin.
Bu yaklaşım özellikle düşük maliyetli IoT ve gelişmekte olan ülkelerdeki
endüstriyel uygulamalar için devrim niteliğinde.

---

## Tez Spin-off Özeti

| # | Konu | Seviye | Mevcut Altyapı |
|---|------|--------|---------------|
| 1 | e-nose koku sınıflandırma | YL | %90 hazır — sensör verisi ekle |
| 2 | Anomaly detection (sparse coding) | YL | %70 hazır — metrikleri ekle |
| 3 | Robotik chemotaxis | YL/Dr | %50 hazır — motor çıkış + Phase 5 |
| 4 | Neuromorphic deployment (Loihi) | YL | %60 hazır — donanım erişimi lazım |
| 5 | Multi-modal sensör füzyonu | Dr | %40 hazır — Phase 8 gerekli |
| 6 | Swarm/federated learning | Dr | %30 hazır — yeni mimari lazım |
