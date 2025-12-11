
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec
import tempfile
import time

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🌿 EcoGuide - Climate RAG Chatbot")

# --- Configuration & Secrets ---
# Try to load secrets from st.secrets, otherwise expect them in environment
# Try to load from environment first, then secrets (handling missing file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except FileNotFoundError:
        pass

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    try:
        PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")
    except FileNotFoundError:
        pass

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ecoguide")
if PINECONE_INDEX_NAME == "ecoguide": # check if default or needs secrets
    try:
        PINECONE_INDEX_NAME = st.secrets.get("PINECONE_INDEX_NAME", "ecoguide")
    except FileNotFoundError:
        pass

# --- Sidebar ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a Climate PDF", type="pdf")
    
    # Optional: Allow user to override keys if not in secrets
    with st.expander("Settings"):
        if not GROQ_API_KEY:
            GROQ_API_KEY = st.text_input("Groq API Key", type="password")
        if not PINECONE_API_KEY:
            PINECONE_API_KEY = st.text_input("Pinecone API Key", type="password")
        if not PINECONE_INDEX_NAME:
            PINECONE_INDEX_NAME = st.text_input("Pinecone Index Name", value="ecoguide")

    process_button = st.button("Process PDF")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Helper Functions ---

def process_pdf(uploaded_file):
    """Reads PDF, splits text, and upserts to Pinecone."""
    if not uploaded_file:
        return None
    
    try:
        # Save uploaded file to a temp file because PyPDFLoader needs a real path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Clean up temp file
        os.remove(tmp_file_path)
        
        return splits
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def get_vectorstore(splits):
    """Create or retrieve Pinecone Vector Store."""
    if not PINECONE_API_KEY:
        st.error("Pinecone API Key is missing.")
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [i.name for i in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            st.warning(f"Index '{PINECONE_INDEX_NAME}' not found. Attempting to create it...")
            try:
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                st.info(f"Index '{PINECONE_INDEX_NAME}' created successfully. Waiting for initialization...")
                time.sleep(10) # Wait for index to be ready
            except Exception as create_error:
                st.error(f"Failed to create index: {create_error}")
                return None
        
        vectorstore = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=PINECONE_INDEX_NAME,
            pinecone_api_key=PINECONE_API_KEY
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        return None

def get_llm_chain(vector_store):
    """Create the RetrievalQA chain."""
    if not GROQ_API_KEY:
        st.error("Groq API Key is missing.")
        return None
        
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant"
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. 
    If the answer is not in the context, say "I cannot find the answer in the document."
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- Main Application Logic ---

if process_button and uploaded_file:
    if not GROQ_API_KEY or not PINECONE_API_KEY:
        st.error("Please provide valid API keys in settings or secrets.toml.")
    else:
        with st.spinner("Processing PDF... This may take a moment."):
            splits = process_pdf(uploaded_file)
            if splits:
                st.session_state.vector_store = get_vectorstore(splits)
                if st.session_state.vector_store:
                    st.success("PDF Processed and stored in Pinecone!")
                else:
                    st.error("Failed to create vector store.")

# --- Chat Interface ---

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question about the climate document...")

if user_input:
    if not st.session_state.vector_store:
        st.warning("Please upload and process a PDF document first.")
    else:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_llm_chain(st.session_state.vector_store)
                if chain:
                    response_dict = chain.invoke({"input": user_input})
                    response_text = response_dict['answer']
                    st.write(response_text)
                    
                    # Update History
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
