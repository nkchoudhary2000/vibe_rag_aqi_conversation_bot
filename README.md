# 🌿 EcoGuide - Climate RAG Chatbot

An AI-powered document assistant that retrieves information from uploaded climate PDFs using Groq (Llama3) and Pinecone.

## 🚀 Features
- **RAG Architecture**: Retrieves relevant chunks from your PDF to answer accurate queries.
- **DeepSpeed/Groq**: Uses the lightning-fast `llama-3.1-8b-instant` model via Groq.
- **Vector Search**: Serverless Pinecone index for scalable embedding retrieval.
- **Local Embeddings**: Uses `all-MiniLM-L6-v2` locally to keep costs zero.

## 🛠️ Tech Stack
- **Python 3.9+**
- **Streamlit** (UI)
- **LangChain** (Orchestration)
- **Groq API** (LLM)
- **Pinecone** (Vector DB)
- **HuggingFace** (Embeddings)

## 📦 Installation & Local Setup

1.  **Clone the repository** (or navigate to your project folder).

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Secrets**:
    Create a file named `.streamlit/secrets.toml` in your project root:
    ```toml
    GROQ_API_KEY = "gsk_..."
    PINECONE_API_KEY = "pcsk_..."
    PINECONE_INDEX_NAME = "ecoguide"
    ```
    *Note: The app will automatically attempt to create the Pinecone index `ecoguide` if it doesn't exist.*

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## ☁️ Deploy to Streamlit Community Cloud

1.  Push your code to a GitHub repository.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Connect your GitHub account and select your repository.
4.  **Advanced Settings**:
    - Go to "Secrets" in the deploy dashboard.
    - Paste the content of your `secrets.toml` into the secrets area.
5.  Click **Deploy**.

## ⚠️ Important Notes
- **Free Tier Only**: This project is designed to run completely on free tiers.
- **Pinecone Index**: The app includes auto-creation logic for the index, but you can also manually create it:
    - **Name**: `ecoguide` (or whatever you set in secrets)
    - **Dimensions**: `384`
    - **Metric**: `cosine`
