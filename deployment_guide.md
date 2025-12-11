# ☁️ Deployment Guide: Streamlit Community Cloud

This guide will walk you through hosting your EcoGuide Chatbot for free using Streamlit Community Cloud.

## Prerequisites

1.  **GitHub Account**: You need a [GitHub account](https://github.com/).
2.  **API Keys**: Ensure you have your `GROQ_API_KEY` and `PINECONE_API_KEY` ready.
3.  **Code pushed to GitHub**: Your project must be in a public (or private) GitHub repository.

## Step 1: Push Code to GitHub

Since you have initialized this project locally, you need to push it to a new GitHub repository.

1.  Create a **new repository** on GitHub (e.g., `ecoguide-rag-bot`).
2.  Follow the instructions to push an existing repository:
    ```bash
    git remote add origin https://github.com/YOUR_USERNAME/ecoguide-rag-bot.git
    git branch -M main
    git push -u origin main
    ```

## Step 2: Deploy on Streamlit Cloud

1.  Go to [share.streamlit.io](https://share.streamlit.io/) and log in.
2.  Click **"New app"**.
3.  Select **"Use existing repo"**.
4.  **Repository**: Select your `ecoguide-rag-bot` repo.
5.  **Branch**: `main`.
6.  **Main file path**: `app.py`.

## Step 3: Configure Secrets (CRITICAL)

Before clicking "Deploy", you must set your API keys as secrets. Streamlit Cloud does **not** read your local `.env` file.

1.  Click **"Advanced settings..."** (or "Manage app" -> "Settings" -> "Secrets" after deploying).
2.  In the "Secrets" text area, paste the following (replace with your actual keys):

    ```toml
    GROQ_API_KEY = "gsk_..."
    PINECONE_API_KEY = "pcsk_..."
    PINECONE_INDEX_NAME = "ecoguide"
    ```

3.  Click **"Save"**.

## Step 4: Launch

1.  Click **"Deploy"**.
2.  Streamlit will install the dependencies from `requirements.txt` and start your app.
3.  **Troubleshooting**:
    - If you see `ModuleNotFoundError`, check your `requirements.txt`.
    - If you see "No active indexes", ensure your `PINECONE_INDEX_NAME` matches the secret and give the app a moment to auto-create it on the first run.

## Zero Cost Confirmation
- **Streamlit Cloud**: Free for community apps.
- **Groq**: Currently offers free beta access (check their pricing page for updates).
- **Pinecone**: Free tier (Serverless Starter) allows 1 index with up to 2GB of storage.
