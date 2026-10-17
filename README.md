# 👟 E-Ticaret Ayakkabı Danışmanı (Query Intent Routing)

Doğal Dil İşleme (NLP), vektör tabanlı anlamsal arama ve karar yönlendirme (Query Intent Routing) mekanizmalarını bir araya getiren yeni nesil akıllı e-ticaret danışmanı prototipi.

Geleneksel e-ticaret filtreleri soyut insan ihtiyaçlarını (*"Hastanede bütün gün ayaktayım, tabanı yumuşak olsun"*) anlamakta yetersiz kalırken; her sorguyu doğrudan büyük dil modellerine (LLM) göndermek de yüksek maliyet, gecikme ve ürün özelliklerinde halüsinasyon riskini beraberinde getirir.

Bu projede her kullanıcı girdisi **Query Intent Routing** katmanıyla filtrelenir; sadece gereken kaynak (deterministik filtre, vektör araması veya LLM muhakemesi) çalıştırılarak kaynak verimliliği sağlanır.

## 🌟 Öne Çıkan Özellikler

* **Maliyet ve Gecikme Optimizasyonu:** Her sorgu için ağır modeller çalıştırmak yerine kural tabanlı ve semantik katmanlar hibrit olarak koordine edilir.

* **Akıllı Veri Akışı (HTTP Streaming):** 18 GB'lık Amazon ürün veri seti diski ve belleği tüketmeden doğrudan internet üzerinden akıtılarak filtrelenmiştir.

* **Zero-Shot Intent & Slot Extraction:** Beden, fiyat tavanı, kategori ve kullanım amacı gibi kısıtlar sorgudan otomatik çıkarılır.

* **Cold-Start (Netleştirme) Mekanizması:** Kullanıcı *"Merhaba, ayakkabı arıyorum"* gibi belirsiz bir giriş yaptığında rastgele ürün fırlatmak yerine önce ayak numarası, kullanım amacı ve bütçe kriterlerini sorar.

* **Şeffaf Debug & Danışman Paneli (Gradio):** Kullanıcı dostu bir satış sohbeti sunulurken, eş zamanlı olarak arka plandaki yönlendirme kararları (Path A/B/C, slotlar, filtreler) gerçek zamanlı izlenebilir.

## 📐 Sistem Mimarisi

```
               [ Kullanıcı Sorgusu ]
                         │
                         ▼
          ┌─────────────────────────────┐
          │     Query Intent Router     │  ◄── Niyet Analizi & Slot Çıkarımı
          └──────────────┬──────────────┘
                         │
         ┌───────────────┼────────────────────────┐
         │               │                        │
         ▼               ▼                        ▼
    [ PATH A ]      [ PATH B ]               [ PATH C ]
  Deterministik    Semantik Arama           LLM Karşılaştırma
  Metadata Filtre  (ChromaDB + Gemini)      Ergonomi & Kıyaslama
  (Fiyat/Beden)    (Konfor/Deneyim Odaklı)
         │               │                        │
         └───────────────┼────────────────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │      Sales Advisor LLM      │  ◄── Clarification / Soğuk Başlangıç
          └──────────────┬──────────────┘
                         │
                         ▼
        [ Danışman Yanıtı + Şeffaf Karar Paneli ]

```

### 🧭 Yönlendirme Yolları (Routing Paths)

| Yol | Tanım | Örnek Sorgu | 
| ----- | ----- | ----- | 
| **Path A (Deterministik)** | Fiyat, numara ve stok gibi net kısıtları doğrudan metadata filtreleriyle eşleştirir; LLM token israfını önler. | *"2000 TL altı 42 numara koşu ayakkabısı"* | 
| **Path B (Semantik Vektör)** | Soyut ve deneyimsel ihtiyaçları Gemini Embedding ile vektör uzayına taşıyıp ChromaDB'de anlamsal benzerlikle arar. | *"Bütün gün ayaktayım, tabanı yumuşak ve terletmeyen bir model"* | 
| **Path C (Karşılaştırma)** | Seçilen ayakkabılar arasındaki ergonomi, malzeme ve kullanım amacı farklarını analiz eder. | *"Listelenen ilk iki üründen hangisi uzun yürüyüşler için daha uygun?"* | 
| **Clarification (Soğuk Başlangıç)** | Belirsiz girdilerde kullanıcıdan eksik kriterleri toplayan nazik bir satış danışmanı diyaloğu başlatır. | *"Merhaba, bana bir şeyler önerir misin?"* | 

## 📸 Arayüz Önizlemesi

Gradio ile geliştirilen çift panelli arayüz:

* **Sol Panel:** Doğal Türkçe satış danışmanı sohbeti.

* **Sağ Panel:** Seçilen yol (`intent`), çıkarılan slotlar, uygulanan metadata filtreleri ve getirilen ürünlerin benzerlik puanları.

*(Gradio arayüzünden aldığınız ekran görüntüsünü `assets/demo_preview.png` yoluna ekleyebilirsiniz)*

## 📂 Proje Yapısı

```
├── advisor.py            # Ana danışman boru hattı ve yanıt üretim mantığı
├── gemini_client.py      # Google Gemini API istemcisi ve Exponential Backoff
├── vector_store.py       # ChromaDB entegrasyonu, embedding ve metadata arama
├── data_loader.py        # Amazon veri seti ayrıştırma ve zenginleştirme
├── main_colab.ipynb      # Adım adım yürütme ve Gradio arayüz başlatıcısı
├── requirements.txt      # Gerekli Python kütüphaneleri
└── README.md             # Proje dokümantasyonu

```

## 🛠️ Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın

```
git clone https://github.com/KULLANICI_ADINIZ/smart-shoe-advisor.git
cd smart-shoe-advisor

```

### 2. Gerekli Kütüphaneleri Yükleyin

```
pip install -r requirements.txt

```

### 3. API Anahtarını Tanımlayın

```
export GEMINI_API_KEY="your_api_key_here"

```

*(Google Colab üzerinde kullanıyorsanız sol menüdeki 🔑 **Secrets** sekmesinden `GEMINI_API_KEY` adıyla ekleyip Notebook erişimini açmanız yeterlidir).*

### 4. Çalıştırma

`main_colab.ipynb` dosyasını Google Colab veya yerel Jupyter Lab ortamında sırayla çalıştırarak Gradio arayüzünü başlatabilirsiniz.

## 🎓 Teşekkür

Bu proje, 1 haftalık yoğun Doğal Dil İşleme (NLP) eğitim sürecinin bitirme çalışması olarak geliştirilmiştir. Doğal dilin bilgisayarlar tarafından anlamlandırılmasındaki zorlukları aşma ve bir yazılım ürününün kullanıcı ihtiyaçlarına göre sürekli evrilmesi bakış açısını kazandıran değerli hocama teşekkür ederim.
