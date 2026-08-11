# 🐉 LoreKeeper: Campaign Memory

LoreKeeper is a lightweight Retrieval-Augmented Generation (RAG) AI assistant built specifically for Dungeon Masters. DMs often have scattered notes about NPCs, locations, and past sessions. LoreKeeper allows you to index all your Markdown or Text notes and chat with them mid-game, instantly recalling specific details without breaking the flow of your session.

## Features
- **Local Document Indexing**: Uses ChromaDB to locally vectorize and store your campaign notes.
- **AI-Powered Retrieval**: Uses Google's latest Gemini models to read your notes and answer conversational questions.
- **Beautiful UI**: A clean Streamlit interface with drag-and-drop file uploading.
- **Source Citations**: The AI tells you exactly which notes it pulled the information from!

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vivekdevaa124/lorekeeper.git
   cd lorekeeper
   ```

2. **Install the dependencies:**
   Make sure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API Key:**
   - Get a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
   - Rename the `.env.example` file to `.env`.
   - Paste your API key into the `.env` file: `GEMINI_API_KEY=your_api_key_here`

## Usage

1. **Start the server:**
   ```bash
   streamlit run app.py
   ```
2. **Upload Notes**: Once the browser window opens, drag and drop your `.md` or `.txt` campaign notes into the sidebar.
3. **Sync Data**: Click the **Sync Notes (Re-Index Data)** button to embed your notes into the vector database.
4. **Chat**: Ask LoreKeeper anything about your campaign!

## Project Structure
- `app.py`: The Streamlit web interface and routing.
- `src/rag_engine.py`: The core RAG logic (Document parsing, ChromaDB setup, and Gemini API integration).
- `data/`: Where your Markdown session notes are stored.
- `.env`: (Ignored by git) Where your API key lives.
