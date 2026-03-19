# Growing Brain: Incremental Complexity Through Evolution

## Core Principle
Evrim YENİ yapılar icat eder, mevcut çalışan yapıları bozmaz.
Her aşamada beyin büyür — yeni nöronlar, yeni alanlar, yeni bağlantılar.
Eski katmanlar korunur, yenileri üstüne eklenir.

---

## Stage 0: Sinek Mushroom Body (TAMAMLANDI)
```
[64 PN] → [200 KC] → [2 MBON]
              ↕
           [1 APL]

Toplam: 267 nöron
Yetenek: 2 sınıf pattern tanıma (%99)
```

## Stage 1: Multi-Class Expansion
Yeni MBON'lar ekle. KC katmanı aynı kalır.
```
[64 PN] → [200 KC] → [10 MBON]  ← yeni MBON'lar eklendi
              ↕
           [1 APL]

Toplam: 275 nöron (+8)
Yetenek: 10 sınıf pattern tanıma (MNIST rakamları)
```
**Evrimsel değişiklik:** Sadece KC→MBON ağırlıkları ve yeni MBON eşikleri evrimleşir.
Eski PN→KC bağlantıları korunur (zaten iyi çalışıyor).

## Stage 2: Sensory Expansion
Daha büyük input için daha fazla PN.
```
[784 PN] → [2000 KC] → [10 MBON]  ← her şey büyüdü
               ↕
            [1 APL]

Toplam: 2795 nöron (+2520)
Yetenek: 28x28 MNIST tam çözünürlük
```
**Evrimsel değişiklik:** PN sayısı ve KC sayısı artık genome'un bir parçası.
Evrim kaç KC gerektiğine kendisi karar verir.

## Stage 3: Feature Extraction Layer
KC'den önce bir "görsel korteks" katmanı ekle.
```
[784 PN] → [200 Feature] → [2000 KC] → [10 MBON]
            Detectors          ↕
            (evolved)       [5 APL]  ← birden fazla APL

Toplam: ~3000 nöron (+200)
Yetenek: Kenar, köşe, eğri algılama → sparse coding → sınıflandırma
```
**Evrimsel değişiklik:** Feature detector katmanı sıfırdan evrimleşir.
Bunlar biyolojik görsel korteksteki basit hücrelere (simple cells) karşılık gelir.
Her feature detector 3-5 PN'den input alır ve belirli uzamsal desenlere yanıt verir.

## Stage 4: Lateral Inhibition & Competition
KC'ler arası rekabet. Sadece APL değil, KC-KC inhibisyon.
```
[784 PN] → [200 FD] → [2000 KC ←→ KC] → [10 MBON]
                           ↕↕↕
                        [5 APL]

Toplam: ~3000 nöron (aynı, ama bağlantılar çok daha zengin)
Yetenek: Daha keskin sparse coding, pattern ayrımı iyileşir
```
**Evrimsel değişiklik:** KC-KC inhibitör bağlantıları evrimleşir.
Bu, winner-take-all competition'ı yerel hale getirir.

## Stage 5: Recurrent Loops & Temporal Processing
KC→KC excitatory bağlantılar + zaman penceresi.
```
[784 PN] → [200 FD] → [2000 KC ←→ KC] → [10 MBON]
                        ↑ recurrent ↑        |
                        └────────────┘    [Feedback]
                            ↕↕↕               |
                         [5 APL]               ↓
                                          [10 Context]  ← YENİ

Toplam: ~3010 nöron (+10 context)
Yetenek: Ardışık pattern tanıma, kısa süreli bellek
```
**Evrimsel değişiklik:** Recurrent bağlantı yapısı ve context nöronları evrimleşir.
Context nöronları, önceki stimulus'un izini taşır.

## Stage 6: Modular Brain Regions
Birden fazla bağımsız mushroom body, her biri farklı göreve özelleşmiş.
```
                    ┌→ [MB_1: Edges] → [MBON_1]─┐
[784 PN] → [200 FD]─┤→ [MB_2: Curves] → [MBON_2]─┤→ [Decision Layer]
                    └→ [MB_3: Texture] → [MBON_3]─┘    (10 nöron)

Toplam: ~7000 nöron
Yetenek: Çoklu özellik entegrasyonu, daha robust tanıma
```
**Evrimsel değişiklik:** Kaç MB modülü olacağı, her birinin boyutu ve
özelleşmesi evrimleşir. Decision layer tüm modülleri birleştirir.

## Stage 7: Neuromodulatory Control
Dopamin + serotonin + oktopamin modülatör sistemleri.
```
[Sensory] → [FD] → [MB modules] → [Decision]
                         ↑↓                ↑
                    [DAN: Dopamine]    [Reward signal]
                    [SER: Serotonin]   [Novelty signal]
                    [OCT: Octopamine]  [Arousal signal]

Toplam: ~7050 nöron
Yetenek: Bağlama göre davranış değişikliği, motivasyon, dikkat
```
**Evrimsel değişiklik:** Modülatör nöronların bağlantı yapısı
ve salgı kuralları evrimleşir. Bu Phase 2'de başarısız olan dopamin
öğrenmenin DOĞRU implementasyonu — modülatörler yapıya entegre,
dışarıdan ekleme değil.

## Stage 8: Growing Genome
Genome kendisi büyüyebilir — nöron ekleyebilir, bölge oluşturabilir.
```
Genome artık sadece ağırlık matrisi değil:
- Nöron sayısı (değişken)
- Bölge sayısı (değişken)
- Bölgeler arası bağlantı topolojisi
- Her bölgenin iç yapısı
- Modülatör kuralları
- Plastisitesi parametreleri

Mutasyon operatörleri:
- add_neuron: Bir bölgeye yeni nöron ekle
- add_region: Tamamen yeni beyin bölgesi oluştur
- connect_regions: İki bölge arasında yeni pathway
- split_region: Bir bölgeyi ikiye ayır (specialization)
- duplicate_region: Bir bölgeyi kopyala ve farklılaştır
```

---

## Implementation Strategy: Her Stage İçin

1. Önceki stage'in en iyi genome'unu başlangıç noktası al
2. Yeni yapıları rastgele ekle (küçük başla)
3. Evrimle yeni yapıları optimize et
4. Eski yapıların ağırlıklarını koru (freeze veya düşük mutasyon)
5. Test et, dökümante et, bir sonraki stage'e geç

Bu yaklaşım biyolojik evrimin tam karşılığı:
**Eski beyin yapıları korunur, yenileri üstüne eklenir, evrim yenileri optimize eder.**
