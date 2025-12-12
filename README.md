# 🌿 EcoGuide - Climate RAG Chatbot

An AI-powered document assistant that retrieves information from uploaded climate PDFs using Groq (Llama3) and Pinecone.

## 📖 Introduction
EcoGuide is a Retrieval-Augmented Generation (RAG) application designed to make exploring climate reports and documents interactive and easy. Instead of scrolling through hundreds of pages of PDFs, users can simply upload a document and ask natural language questions. The app uses advanced vector search to find relevant sections of the text and a powerful LLM to synthesize concise answers.

## 🚀 Features
- **RAG Architecture**: Retrieves relevant chunks from your PDF to answer accurate queries effectively.
- **DeepSpeed/Groq**: Uses the lightning-fast `llama-3.1-8b-instant` model via Groq API.
- **Vector Search**: Serverless Pinecone index for scalable and fast embedding retrieval.
- **Local Embeddings**: Uses HuggingFace's `all-MiniLM-L6-v2` locally to generate embeddings, keeping costs low.
- **Chat Interface**: Simple Streamlit chat interface with history.

## ⚙️ How It Works (Architecture)
The application follows a standard RAG pipeline:
1.  **Ingestion**: The user uploads a PDF.
2.  **Processing**: Custom logic splits the PDF into smaller, manageable text chunks (1000 characters) with overlap.
3.  **Embedding**: Each text chunk is converted into a 384-dimensional vector using a local HuggingFace model.
4.  **Storage**: These vectors are uploaded to a Pinecone Serverless Index (`ecoguide`).
5.  **Retrieval**: When a user asks a question, it is also embedded. The system searches Pinecone for the most similar text chunks.
6.  **Generation**: The retrieved chunks + the user's question are sent to the Groq LLM to generate a final answer.

## 🛠️ Tech Stack
- **Languages**: Python 3.9+
- **Frontend**: Streamlit
- **Orchestration**: LangChain
- **LLM**: Groq (Llama-3.1-8b-instant)
- **Vector Database**: Pinecone (Serverless)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)

---

## 📦 Installation & Local Setup

### Prerequisites
- Python 3.9 or higher installed.
- API Keys for **Groq** and **Pinecone**.

### Steps
1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd vibe_rag_aqi_conversation_bot
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    
    pip install -r requirements.txt
    ```

3.  **Setup Configuration (Optional but Recommended)**
    Create a `.streamlit/secrets.toml` file in the root directory to store your keys safely.
    
    **Path:** `.streamlit/secrets.toml`
    ```toml
    GROQ_API_KEY = "gsk_..."
    PINECONE_API_KEY = "pcsk_..."
    PINECONE_INDEX_NAME = "ecoguide"
    ```
    *If you skip this, you can enter keys manually in the app sidebar.*

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

---

## 🕹️ Usage Guide

### 1. Configure Settings
On the sidebar, ensure your API keys are valid. If you added them to `secrets.toml`, they will auto-load. If not, paste them into the "Groq API Key" and "Pinecone API Key" fields.

### 2. Upload Document
- Locate the **"Upload Document"** section in the sidebar.
- Click "Browse files" and select a PDF (e.g., a climate report).
- Click the **"Process PDF"** button.
- *Wait for the success message: "PDF Processed and stored in Pinecone!"*

### 3. Connect to Existing Knowledge Base
- If you have already processed a document previously and simply want to chat, the app will auto-connect to your existing Pinecone index `ecoguide` on launch if keys are present.

### 4. Chat
- Go to the main chat interface.
- Type your question in the input box (e.g., *"What are the key risks mentioned in this report?"*).
- The bot will "Think..." and respond with an answer derived solely from the document context.

---

## 🔧 Configuration Reference

| Environment Variable | Description | Default |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Your API key from [Groq Console](https://console.groq.com/). | None |
| `PINECONE_API_KEY` | Your API key from [Pinecone Console](https://app.pinecone.io/). | None |
| `PINECONE_INDEX_NAME` | Name of the index to store vectors. | `ecoguide` |

---

## ☁️ Deployment (Streamlit Cloud)
1.  Push code to GitHub.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Deploy the repo.
4.  **Crucial**: In Streamlit Cloud "Advanced Settings", copy-paste the contents of your `secrets.toml` into the "Secrets" field.

## ❓ Troubleshooting

**Q: "Pinecone API Key is missing" error?**
A: Ensure you have pasted the key in the sidebar or strictly followed the `secrets.toml` format.

**Q: Index creation fails?**
A: Free tier Pinecone allows only **1 serverless project**. Check if you already have an index in your Pinecone console. You may need to delete an old one or rename the `PINECONE_INDEX_NAME`.

**Q: "NotImplementedError" during deployment?**
A: This often happens if the environment tries to use CUDA for embeddings. The code explicitly forces `device='cpu'` locally to avoid this, but ensure you aren't overriding torch settings in a custom environment.

**Q: "Rate Limit Exceeded" from Groq?**
A: The free tier of Groq has limits. Wait a minute and try again.
