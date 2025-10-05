# Import the necessary libraries
import streamlit as st
import google.generativeai as genai

# --- 1. Page Configuration and Title ---
st.title("🤖 Chatbot Spesialis")
st.caption("Asisten AI yang dilatih dengan instruksi khusus")

# --- 2. Sidebar for Settings ---
with st.sidebar:
    st.subheader("Kontrol")
    # Tombol untuk mereset percakapan
    reset_button = st.button("Reset Percakapan", help="Hapus semua pesan dan mulai dari awal")
    
    st.divider()
    
    # Text area untuk instruksi sistem (persona)
    st.subheader("Atur Persona Chatbot")
    system_prompt = st.text_area(
        "Instruksi Sistem:", 
        "Anda adalah asisten AI yang ahli. Jawab pertanyaan pengguna dengan jelas dan informatif berdasarkan persona yang saya berikan. Jawablah selalu dalam Bahasa Indonesia.",
        height=150
    )

# --- 3. API Key and Model Initialization ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google AI API Key tidak ditemukan. Mohon tambahkan ke Streamlit secrets Anda.", icon="🗝️")
    st.stop()

# Inisialisasi Model Gemini dengan instruksi sistem dari sidebar
# Blok ini hanya berjalan sekali per sesi atau jika persona berubah
if "gemini_model" not in st.session_state or st.session_state.get("last_prompt") != system_prompt:
    st.session_state.last_prompt = system_prompt
    try:
        genai.configure(api_key=google_api_key)
        st.session_state.gemini_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system_prompt
        )
        # Hapus riwayat chat lama saat persona diubah
        st.session_state.pop("chat", None)
        st.session_state.pop("messages", None)
        st.success("Persona chatbot berhasil diperbarui!")
    except Exception as e:
        st.error(f"Gagal menginisialisasi model Gemini: {e}")
        st.stop()

# --- 4. Chat History Management ---
# Inisialisasi sesi chat menggunakan model yang sudah dikonfigurasi
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.gemini_model.start_chat(history=[])

# Inisialisasi riwayat pesan untuk ditampilkan jika belum ada
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Persona saya sudah diatur. Silakan ajukan pertanyaan."}
    ]

# Penanganan tombol reset
if reset_button:
    st.session_state.pop("chat", None)
    st.session_state.pop("messages", None)
    st.rerun()

# --- 5. Display Past Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 6. Handle User Input and API Communication ---
if prompt := st.chat_input("Tulis pesan Anda..."):
    # Tambahkan dan tampilkan pesan pengguna
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dapatkan respons dari asisten
    try:
        with st.spinner("Sedang berpikir..."):
            response = st.session_state.chat.send_message(prompt)
        
        answer = response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        answer = f"Terjadi kesalahan: {e}"
        st.error(answer)

    # Tambahkan dan tampilkan pesan asisten
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

