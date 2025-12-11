
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
# Load defaults from environment/secrets if available, but do not enforce them here.
env_groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
env_pinecone_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
env_pinecone_index = os.getenv("PINECONE_INDEX_NAME") or st.secrets.get("PINECONE_INDEX_NAME", "ecoguide")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    
    st.markdown("### API Configuration")
    st.info("Enter your own API keys to use this app. Keys are not stored persistently.")

    # Input fields for API keys (use env values as defaults if they exist)
    groq_api_key = st.text_input("Groq API Key", type="password", value=env_groq_key)
    pinecone_api_key = st.text_input("Pinecone API Key", type="password", value=env_pinecone_key)
    pinecone_index_name = st.text_input("Pinecone Index Name", value=env_pinecone_index)

    st.divider()
    
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a Climate PDF", type="pdf")
    process_button = st.button("Process PDF")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Helper Functions ---

def process_pdf(uploaded_file):
    """Reads PDF, splits text."""
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

def get_vectorstore(splits, api_key, index_name):
    """Create or retrieve Pinecone Vector Store."""
    if not api_key:
        st.error("Pinecone API Key is missing.")
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = [i.name for i in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            st.warning(f"Index '{index_name}' not found. Attempting to create it...")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                st.info(f"Index '{index_name}' created successfully. Waiting for initialization...")
                time.sleep(10) # Wait for index to be ready
            except Exception as create_error:
                st.error(f"Failed to create index: {create_error}")
                return None
        
        vectorstore = PineconeVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=index_name,
            pinecone_api_key=api_key
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {e}")
        return None

def get_llm_chain(vector_store, api_key):
    """Create the RetrievalQA chain."""
    if not api_key:
        st.error("Groq API Key is missing.")
        return None
        
    llm = ChatGroq(
        groq_api_key=api_key,
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
    if not groq_api_key or not pinecone_api_key:
        st.error("Please provide valid API keys in the sidebar.")
    else:
        with st.spinner("Processing PDF... This may take a moment."):
            splits = process_pdf(uploaded_file)
            if splits:
                st.session_state.vector_store = get_vectorstore(splits, pinecone_api_key, pinecone_index_name)
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
            if not groq_api_key:
                st.error("Groq API Key is missing. Please add it in the sidebar.")
            else:
                with st.spinner("Thinking..."):
                    chain = get_llm_chain(st.session_state.vector_store, groq_api_key)
                    if chain:
                        try:
                            response_dict = chain.invoke({"input": user_input})
                            response_text = response_dict['answer']
                            st.write(response_text)
                            
                            # Update History
                            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
