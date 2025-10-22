import streamlit as st
import os
import sys
import textwrap
from google import genai
from google.genai import types
from pypdf import PdfReader
from numpy.linalg import norm
from numpy import dot
import numpy as np
from dotenv import load_dotenv


def batch_list(iterable, n=100):
    """Bir listeyi N boyutunda parçalara (batch) ayırır."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# --- Gemini RAG Fonksiyonları ---

def chunk_text(text, chunk_size=1000, overlap_ratio=0.2):
    overlap = int(chunk_size * overlap_ratio)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if chunk_size - overlap <= 0:
            break
    return chunks


@st.cache_resource
def setup_rag_pipeline():
    """
    RAG pipeline'ını kurar, PDF'leri yükler ve vektörleştirir.
    """

    load_dotenv()
    api_key_value = os.environ.get('GEMINI_API_KEY')

    if not api_key_value:
        st.error("API Anahtarı bulunamadı. Lütfen run_app.py dosyasındaki API_KEY'i kontrol edin.")
        return None, None, None

    try:
        client = genai.Client(api_key=api_key_value)
        embedding_model = 'text-embedding-004'
        file_paths = ["5746_Kanun.pdf", "Arge_Yonetmelik.pdf"]
        raw_text = ""

        # 1. Dosya Yükleme
        for file_path in file_paths:
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    raw_text += page.extract_text()
            except FileNotFoundError:
                st.error(
                    f"Kritik Hata: '{file_path}' bulunamadı. Lütfen dosyanın proje dizininde olduğundan emin olun.")
                return None, None, None

        # 2. Parçalama
        text_chunks = chunk_text(raw_text, chunk_size=1000, overlap_ratio=0.2)
        if not text_chunks:
            st.error("Parçalama başarısız. PDF dosyalarını kontrol edin.")
            return None, None, None

        # 3. Vektörleme (Embeddings)
        st.info(f"Toplam {len(text_chunks)} parça vektörleştiriliyor... (Batch boyutu: 100)")

        chunk_embeddings = []
        for batch in batch_list(text_chunks, 100):
            embeddings = client.models.embed_content(
                model=embedding_model,
                contents=batch,
            ).embeddings
            chunk_embeddings.extend(embeddings)

        indexed_data = list(zip(text_chunks, [e for e in chunk_embeddings]))
        st.success(f"RAG Veri Seti başarıyla oluşturuldu: {len(indexed_data)} parça.")

        return client, indexed_data, embedding_model
    except Exception as e:
        st.error(f"RAG kurulumunda KRİTİK HATA oluştu: {e}")
        st.warning("Bu genellikle yanlış API anahtarı (400 Invalid Argument) veya bağlantı sorunlarından kaynaklanır.")
        return None, None, None


def get_rag_response(client, query, indexed_data, embedding_model, chat_history, top_k=5):
    """Sorguyu alır, ilgili dokümanları çeker ve Gemini'ye gönderir."""

    # a. Sorguyu Vektörleme (Query Embedding)
    query_embedding_response = client.models.embed_content(
        model=embedding_model,
        contents=[query],
    ).embeddings[0]

    # ContentEmbedding objesini np.array'e dönüştür
    query_embedding = np.array(query_embedding_response)

    # b. En Yakın Komşuları Bulma (Retrieval - Kosinüs Benzerliği)
    scores = []
    for chunk_text, chunk_embed in indexed_data:
        # Vektör veritabanı objesini np.array'e dönüştür
        np_chunk_embed = np.array(chunk_embed)

        # Kosinüs Benzerliği Hesaplaması
        try:
            score = dot(query_embedding, np_chunk_embed) / (norm(query_embedding) * norm(np_chunk_embed))
        except Exception as e:
            # Eğer norm sıfır ise (boş chunk), skoru sıfır yap
            score = 0.0

        scores.append(score)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    context = "\n\n---\n\n".join([indexed_data[i][0] for i in top_indices])

    # c. Gemini Prompter (Generation)
    history_str = "\n".join([f"Kullanıcı: {h['user']}\nAsistan: {h['assistant']}" for h in chat_history])

    # *** GÜNCELLENMİŞ SİSTEM TALİMATI (Kaynağın Adı Belirtiliyor) ***
    system_instruction = f"""Sen bir Ar-Ge mevzuat danışmanısın. Cevap verirken KESİNLİKLE öncelikle aşağıdaki 'VERİ KAYNAĞI' bölümündeki bilgileri kullan.

    KURALLAR:
    1. Yalnızca kaynakta bilgi varsa, cevabı kaynağa göre ver.
    2. Kaynakta bilgi YOKSA, ÖNCE 'Kaynak dökümanlarda (5746 sayılı Kanun / Arge Yönetmelik) bu bilgiye doğrudan ulaşılamamaktadır.' ifadesini kullan.
    3. Hemen ardından, bu konuyla ilgili kısa ve genel bir bilgi/yorum ekle. Cevabın 2-3 cümleyi geçmesin.

    Lütfen her zaman kesin ve mevzuata uygun bir dil kullan.

    SOHBET GEÇMİŞİ:
    {history_str}

    VERİ KAYNAĞI (Mevzuattan Alınan Parçalar):
    {context}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            {"role": "user", "parts": [{"text": query}]}
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    )

    sources = [{"page_content": indexed_data[i][0], "source": "Yüklü Mevzuat"} for i in top_indices]

    return response.text, sources


# --- Streamlit Uygulaması Arayüzü ---
st.set_page_config(page_title="Kurumsal Ar-Ge Mevzuat Chatbot", layout="wide")

# BAŞLIK
st.title("🤖 Ar-Ge Mevzuat Danışmanı")
st.caption("Veri kaynağı: Kurumsal Ar-Ge Mevzuatları (Gemini API ile doğrudan entegrasyon).")

# Session state'i kullanarak RAG pipeline'ını bir kez kurma
if 'client' not in st.session_state:
    client, indexed_data, embedding_model = setup_rag_pipeline()
    st.session_state.client = client
    st.session_state.indexed_data = indexed_data
    st.session_state.embedding_model = embedding_model
    st.session_state.chat_history = []
    st.session_state.current_message = None

client = st.session_state.client
indexed_data = st.session_state.indexed_data
embedding_model = st.session_state.embedding_model

if not client or not indexed_data:
    st.stop()

# --- CHAT GEÇMİŞİNİ GÖSTERME ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Kaynakları geçmiş mesajlarda gösterme
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Kullanılan Kaynaklar (RAG Kanıtı)"):
                for i, doc in enumerate(message["sources"], 1):
                    source_text = doc['page_content']
                    st.markdown(f"**Kaynak {i}:** **Dosya:** {doc.get('source', 'Yüklü Mevzuat')}")
                    st.code(source_text[:500] + ('...' if len(source_text) > 500 else ''), language='markdown')

# Örnek Sorular
example_questions = [
    "5746 sayılı kanunun amacı nedir?",
    "Ar-Ge faaliyetleri kapsamında gelir vergisi stopajı teşviki nasıl uygulanır?",
    "Ar-Ge indirimi hangi gider kalemlerini kapsar?"
]

# Giriş ekranında örnek soruları gösterme
if st.session_state.current_message is None and not st.session_state.chat_history:
    st.subheader("Örnek Sorular (Test Kılavuzu)")
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.current_message = q
            st.rerun()

# Kullanıcı girişi yakalama
if user_query := st.chat_input("Ar-Ge mevzuatı ile ilgili bir soru sorun..."):
    st.session_state.current_message = user_query

# Mesajı İşleme ve Cevaplama
if st.session_state.current_message:
    user_query = st.session_state.current_message

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Cevap aranıyor..."):
            try:
                history_for_rag = [
                    {"user": h["content"], "assistant": h["content"]}
                    for h in st.session_state.chat_history
                    if h["role"] == "user" or h["role"] == "assistant"
                ]

                answer, source_docs = get_rag_response(
                    client, user_query, indexed_data, embedding_model, history_for_rag
                )

                # Geçmişi güncelleme
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": source_docs})
                st.session_state.current_message = None

            except Exception as e:
                error_msg = f"Üzgünüm, RAG zincirinde bir hata oluştu: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.session_state.current_message = None

            st.rerun()