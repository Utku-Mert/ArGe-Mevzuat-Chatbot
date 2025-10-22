# 🤖 Kurumsal Ar-Ge Mevzuat Danışmanı (Yerel RAG Chatbot)

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, Retrieval Augmented Generation (RAG) mimarisini kullanarak kurumsal Ar-Ge mevzuatları üzerine soruları yanıtlayan yerel bir Streamlit chatbot uygulamasıdır.

## 1. Projenin Amacı 

5746 sayılı Kanun ve ilgili yönetmelikler gibi kurumsal mevzuat dökümanlarını temel alarak, kullanıcılara mevzuat hakkında hızlı, doğru ve kanıta dayalı yanıtlar sunmak. Sistem, yalnızca verilen kaynaklardaki bilgilere odaklanarak halüsinasyon riskini minimize ederken, kaynakta bilgi bulunmadığında konuya dair kısa bir uzman yorumu sunarak kullanıcı deneyimini iyileştirmeyi amaçlar.

## 2. Veri Seti Hakkında Bilgi

Projede kullanılan veri seti, Türkiye Cumhuriyeti'nin **Ar-Ge mevzuatını** içeren iki adet PDF dosyasından oluşmaktadır:
1. **5746_Kanun.pdf**: Araştırma, Geliştirme ve Tasarım Faaliyetlerinin Desteklenmesi Hakkında Kanun.
2. **Arge_Yonetmelik.pdf**: İlgili Kanunun Uygulama Yönetmeliği.

Veri setindeki metinler, hassasiyeti korumak adına RAG zincirine beslenmeden önce 1000 karakterlik parçalara (chunks) ayrılmış ve parçalar arasında %20 oranında örtüşme (overlap) kullanılmıştır.

## 3. Kullanılan Yöntemler ve Çözüm Mimarisi 

Proje, Saf Python, `google-genai` kütüphanesi ve Streamlit kullanılarak geliştirilmiş bir **Yerel RAG Çözüm Mimarisi** sunmaktadır.

| Bileşen | Teknoloji / Model | Açıklama |
| :--- | :--- | :--- |
| **Generative Model (LLM)** | Google Gemini 2.5 Flash API | Kullanıcı sorgularını ve çekilen bağlamı (context) kullanarak nihai cevabı üretir. |
| **Embedding Model** | Google `text-embedding-004` API | Metin parçalarını (chunks) ve kullanıcı sorgusunu yüksek boyutlu vektörlere dönüştürür. |
| **Vektör Veritabanı** | NumPy Array (Yerel Indexleme) | Vektörlerin saklanması, Kosinüs Benzerliği (Cosine Similarity) kullanılarak en alakalı parçaların hızlıca çekilmesi sağlanmıştır. |
| **Arayüz (Web Frontend)** | Streamlit | Chatbot'un kolayca kullanılabilir, etkileşimli ve yerel bir web arayüzü ile sunulmasını sağlar. |
| **Çerçeve (Framework)** | Saf Python / Google GenAI SDK | Projede karmaşık RAG kütüphaneleri (LangChain, Haystack) yerine, çekirdek RAG mantığı Python ve Google GenAI kütüphaneleri ile baştan inşa edilmiştir. |

## 4. Kodunuzun Çalışma Kılavuzu (Virtual Environment & Kurulum) 

Projeyi çalıştırmak için aşağıdaki adımları takip edin:

### Adım 1: Gerekli Dosyalar
Proje dizininizde aşağıdaki dosyaların bulunduğundan emin olun:
- `app.py`
- `run_app.py`
- `requirements.txt`
- `5746_Kanun.pdf`
- `Arge_Yonetmelik.pdf`

### Adım 2: Sanal Ortam Kurulumu
Projeyi yalıtılmış bir ortamda çalıştırmak için bir sanal ortam oluşturun ve etkinleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```
### Adım 3: Bağımlılıkların Kurulumu
`requirements.txt` dosyasındaki gerekli Python paketlerini kurun:

```bash
pip install -r requirements.txt
```

### Adım 4: API Anahtarını Tanımlama
Google Gemini API Anahtarınızı (GEMINI_API_KEY) içeren bir .env dosyası oluşturun ve anahtarı içine yapıştırın:

```bash
# .env dosyası içeriği
GEMINI_API_KEY="...SizinAPIAnahtarınız"
```

### Adım 5: Uygulamayı Çalıştırma
Uygulamayı Streamlit ile çalıştırın. Her zaman run_app.py dosyasını kullanın:
```bash
streamlit run run_app.py
```
