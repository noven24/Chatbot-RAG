# Import the necessary libraries
import streamlit as st
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, MarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. Page Configuration and Title ---
st.set_page_config(page_title="FitBot RAG", page_icon="🤖")
st.title("🤖 FitBot dengan RAG")
st.caption("Chatbot ini menjawab pertanyaan berdasarkan dokumen yang Anda unggah.")

# --- 2. API Key Configuration ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google AI API Key not found. Please add it to your Streamlit secrets.", icon="🗝️")
    st.stop()

# --- 3. Sidebar for File Upload ---
with st.sidebar:
    st.header("Konfigurasi")
    uploaded_file = st.file_uploader(
        "Unggah Basis Pengetahuan Anda (PDF, TXT, MD)", 
        type=["pdf", "txt", "md"]
    )

# --- Functions for RAG Chain ---

# Fungsi ini di-cache agar tidak perlu membuat ulang vector store setiap saat
@st.cache_resource
def create_vector_store(uploaded_file_bytes, file_name):
    """Membuat vector store dari file yang diunggah."""
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
            tmp_file.write(uploaded_file_bytes)
            tmp_file_path = tmp_file.name

        # Load the document based on its extension
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_file_path)
        elif file_name.endswith(".txt"):
            loader = TextLoader(tmp_file_path)
        elif file_name.endswith(".md"):
            loader = MarkdownLoader(tmp_file_path)
        else:
            st.error("Format file tidak didukung.")
            return None
        
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Clean up the temporary file
        os.remove(tmp_file_path)
        
        st.success(f"Basis pengetahuan dari '{file_name}' berhasil dibuat!")
        return vector_store

    except Exception as e:
        st.error(f"Gagal memproses file: {e}")
        return None

# --- Main Application Logic ---

# Inisialisasi retriever di session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if uploaded_file:
    # Buat vector store saat file baru diunggah
    uploaded_file_bytes = uploaded_file.getvalue()
    vector_store = create_vector_store(uploaded_file_bytes, uploaded_file.name)
    if vector_store:
        st.session_state.retriever = vector_store.as_retriever()
else:
    st.info("Silakan unggah file di sidebar untuk memulai.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- RAG Chain and User Interaction ---
if st.session_state.retriever:
    # Define the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.7)

    # Define the prompt template
    prompt_template = """
    Anda adalah asisten AI yang ahli dalam konten dokumen yang diberikan.
    Jawab pertanyaan pengguna hanya berdasarkan konteks berikut.
    Jika Anda tidak tahu jawabannya dari konteks yang diberikan, katakan "Saya tidak menemukan informasi tersebut di dalam dokumen."
    Jawablah selalu dalam Bahasa Indonesia.

    Konteks:
    {context}

    Pertanyaan:
    {question}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Create the RAG chain
    rag_chain = (
        {"context": st.session_state.retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # React to user input
    if user_prompt := st.chat_input("Tanyakan sesuatu tentang dokumen Anda..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Get assistant response from the RAG chain
        with st.spinner("FitBot sedang berpikir..."):
            response = rag_chain.invoke(user_prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
