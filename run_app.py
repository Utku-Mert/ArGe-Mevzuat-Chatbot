import os
import subprocess
from dotenv import load_dotenv

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

# API Anahtarını ortam değişkeninden güvenli bir şekilde çek
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("HATA: GEMINI_API_KEY ortam değişkeni yüklenemedi.")
    print("Lütfen proje ana dizinindeki .env dosyasını kontrol edin.")
else:
    os.environ["GEMINI_API_KEY"] = API_KEY

    print("Gemini API Anahtarı yüklendi. Streamlit uygulaması başlatılıyor...")

    try:
        # Streamlit uygulamasını başlatma
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except FileNotFoundError:
        print("HATA: Streamlit çalıştırılamadı. Kütüphanelerin doğru yüklendiğinden emin olun.")
    except Exception as e:
        print(f"Streamlit başlatılırken hata oluştu: {e}")