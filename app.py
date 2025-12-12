import streamlit as st
import os
import time
import uuid
import tempfile
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader

# --- Configuration & Setup ---
load_dotenv()

st.set_page_config(
    page_title="EcoGuide - Climate RAG",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🌿 EcoGuide - Climate RAG Chatbot")

# --- API Clients Initialization (Lazy Loading) ---

@st.cache_resource
def get_embedding_model():
    # Load a small, fast local model for embeddings
    return SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def get_pinecone_client(api_key):
    return Pinecone(api_key=api_key)

def get_groq_client(api_key):
    return Groq(api_key=api_key)

# --- Helper Functions ---

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def recursive_text_split(text, chunk_size=1000, chunk_overlap=200):
    """
    Simple recursive-like splitter that respects sentence boundaries roughly
    by just splitting on generic length for now, or using a simple strategy.
    To match 'RecursiveCharacterTextSplitter' behavior without the library,
    we can just split by characters with overlap.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward by chunk_size - overlap
        start += chunk_size - chunk_overlap
        
    return chunks

def upsert_to_pinecone(index, chunks, embeddings, namespace=""):
    """
    Upserts vectors to Pinecone.
    """
    vectors = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        # Create a unique ID for each chunk
        vector_id = str(uuid.uuid4())
        # Metadata allows us to retrieve the text later
        metadata = {"text": chunk}
        vectors.append({
            "id": vector_id, 
            "values": vector.tolist(), 
            "metadata": metadata
        })
    
    # Batch upsert (Pinecone recommends batches of 100 or so)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch, namespace=namespace)

def get_relevant_context(index, query_vector, k=3, namespace=""):
    results = index.query(
        vector=query_vector.tolist(),
        top_k=k,
        include_metadata=True,
        namespace=namespace
    )
    
    contexts = []
    for match in results['matches']:
        if 'metadata' in match and 'text' in match['metadata']:
            contexts.append(match['metadata']['text'])
    return "\n\n---\n\n".join(contexts)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Settings")
    st.info("Enter your API keys. No data is stored persistently.")
    
    env_groq = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    env_pine = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
    env_idx = os.getenv("PINECONE_INDEX_NAME") or st.secrets.get("PINECONE_INDEX_NAME", "ecoguide")

    groq_api_key = st.text_input("Groq API Key", type="password", value=env_groq)
    pinecone_api_key = st.text_input("Pinecone API Key", type="password", value=env_pine)
    pinecone_index_name = st.text_input("Pinecone Index Name", value=env_idx)
    
    st.divider()
    
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    process_button = st.button("Process PDF")
    
    st.divider()
    
    # Status Area
    status_container = st.empty()

# --- Main Logic ---

# Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF
if process_button and uploaded_file and pinecone_api_key and pinecone_index_name:
    try:
        with st.spinner("Initializing Embedding Model..."):
            model = get_embedding_model()
            
        with st.spinner("Reading PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
                
            raw_text = read_pdf(tmp_path)
            os.remove(tmp_path)
            
        with st.spinner("Splitting Text..."):
            chunks = recursive_text_split(raw_text)
            st.sidebar.write(f"Created {len(chunks)} text chunks.")
            
        with st.spinner("Generating Embeddings..."):
            embeddings = model.encode(chunks)
            
        with st.spinner("Connecting to Pinecone..."):
            pc = get_pinecone_client(pinecone_api_key)
            
            # Check/Create Index
            existing_indexes = [i.name for i in pc.list_indexes()]
            if pinecone_index_name not in existing_indexes:
                st.sidebar.warning(f"Creating index '{pinecone_index_name}'...")
                pc.create_index(
                    name=pinecone_index_name,
                    dimension=384, # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                time.sleep(10) # Wait for init
                
            index = pc.Index(pinecone_index_name)
            
        with st.spinner("Upserting to Knowledge Base..."):
            upsert_to_pinecone(index, chunks, embeddings)
            st.toast("Document processed and added to Knowledge Base!", icon="✅")
            
    except Exception as e:
        st.error(f"Error processing document: {e}")

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a climate question..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Assistant Response
    with st.chat_message("assistant"):
        if not groq_api_key or not pinecone_api_key:
            st.error("Please configure API keys in the sidebar.")
            response_text = "I need API keys to function."
        else:
            try:
                # A. Embed Query
                model = get_embedding_model()
                query_vector = model.encode([prompt])[0]
                
                # B. Retrieve form Pinecone
                pc = get_pinecone_client(pinecone_api_key)
                index = pc.Index(pinecone_index_name)
                context_text = get_relevant_context(index, query_vector, k=4)
                
                # C. Call Groq LLM
                groq_client = get_groq_client(groq_api_key)
                
                system_prompt = f"""You are a helpful Climate Assistant. Use the provided context to answer the user's question.
If the answer is not in the context, say you don't know based on the document.

Context:
{context_text}
"""
                
                stream = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.1,
                    stream=True
                )
                
                # Stream response
                def stream_parser(stream):
                    for chunk in stream:
                        if chunk.choices:
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content

                response_text = st.write_stream(stream_parser(stream))
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                response_text = "Sorry, I encountered an error."
                
    st.session_state.messages.append({"role": "assistant", "content": response_text})
