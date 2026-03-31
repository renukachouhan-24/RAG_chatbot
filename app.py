import streamlit as st
import os
import hashlib
from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# --- 1. INITIAL SETUP ---
load_dotenv()  # Loads variables from your .env file
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide", page_icon="📚")

# Retrieve API key from Streamlit Cloud secrets first, then local .env
def get_groq_api_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except (StreamlitSecretNotFoundError, KeyError):
        return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()

# --- 2. CORE FUNCTIONS ---

def process_pdf(file_path):
    """
    Handles the RAG Pipeline: Load -> Split -> Embed -> Store
    """
    try:
        # Load the PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Split into chunks (1000 chars with 10% overlap for context)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)
        
        # Initialize FREE Local Embeddings (HuggingFace)
        # This runs on your CPU/GPU, no API key needed.
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create FAISS Vector Store (in-memory)
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        # Initialize Groq LLM (Llama 3.3)
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY, 
            model_name="llama-3.3-70b-versatile",
            temperature=0.2 # Lower temperature = more factual answers
        )
        
        # Create the Retrieval Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vector_db.as_retriever()
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def reset_chat():
    """Clears the chat history from the UI"""
    st.session_state.messages = []


def get_file_id(file_bytes: bytes, file_name: str) -> str:
    """Create a stable ID so we can detect when user uploads a new PDF."""
    content_hash = hashlib.md5(file_bytes).hexdigest()
    return f"{file_name}:{content_hash}"

# --- 3. USER INTERFACE (STREAMLIT) ---

st.title("📚 AI PDF Research Assistant")
st.markdown("Upload a PDF and ask questions. Powered by **Groq + Llama 3**.")

# Sidebar for Configuration & Upload
with st.sidebar:
    st.header("1. Setup")
    
    # Check if API Key exists
    if not GROQ_API_KEY:
        st.warning("⚠️ GROQ_API_KEY not found. Add it in Streamlit Cloud Secrets (for deploy) or .env (for local run).")
    else:
        st.success("✅ API Key Loaded")

    st.divider()
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        current_file_id = get_file_id(file_bytes, uploaded_file.name)

        # Save file to a temporary location
        with open("temp_upload.pdf", "wb") as f:
            f.write(file_bytes)

        # Rebuild vector DB if this is a different PDF than last time
        if st.session_state.get("current_file_id") != current_file_id:
            with st.spinner("Analyzing PDF and building vector database..."):
                st.session_state.qa_bot = process_pdf("temp_upload.pdf")
                st.session_state.current_file_id = current_file_id

            # New PDF => clear old chat history so answers don't mix contexts
            reset_chat()

            if st.session_state.qa_bot:
                st.success("PDF Indexed Successfully!")
        else:
            st.info("This PDF is already indexed.")

    st.divider()
    
    # Reset Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        reset_chat()
        st.rerun()

# --- 4. CHAT LOGIC ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about this document?"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if "qa_bot" in st.session_state and st.session_state.qa_bot:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Call the RAG chain
                response = st.session_state.qa_bot.invoke(prompt)
                full_response = response["result"]
                st.markdown(full_response)
                
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.info("Please upload and process a PDF to start chatting.")