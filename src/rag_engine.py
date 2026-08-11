import os
import glob
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

class DocumentLoader:
    """
    A simple loader to read markdown and text files from a directory 
    and split them into manageable chunks (paragraphs).
    """
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_and_chunk(self):
        """Reads all .md and .txt files and splits them by double newlines."""
        documents = []
        metadatas = []
        ids = []
        
        doc_id = 0
        for ext in ["*.md", "*.txt"]:
            for file_path in self.data_dir.rglob(ext):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                        # Very simple chunking strategy: split by double newlines (paragraphs)
                        chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 50]
                        
                        for chunk in chunks:
                            documents.append(chunk)
                            metadatas.append({"source": file_path.name})
                            ids.append(f"doc_{doc_id}")
                            doc_id += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
        return documents, metadatas, ids


class LoreDatabase:
    """
    Manages the vector store using TF-IDF and Cosine Similarity.
    (Replaces ChromaDB to maintain zero-dependency local embeddings on Python 3.14).
    """
    def __init__(self, persist_directory=None, collection_name=None):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.metadatas = []
        self.tfidf_matrix = None

    def add_documents(self, documents, metadatas, ids):
        """Adds documents to the vector store."""
        if not documents:
            return 0
            
        self.documents = documents
        self.metadatas = metadatas
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        return len(documents)

    def search(self, query, top_k=3):
        """Searches the vector store for the most relevant chunks."""
        if not self.documents or self.tfidf_matrix is None:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        if len(similarities) == 0:
            return []
            
        k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        formatted_results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                formatted_results.append({
                    "text": self.documents[idx],
                    "source": self.metadatas[idx]["source"]
                })
        return formatted_results


class AnswerGenerator:
    """
    Connects to the Gemini API to generate an answer based on the retrieved context.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env")
            
        genai.configure(api_key=api_key)
        # Using the standard Gemini 1.5 Flash model for fast, capable generation
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_answer(self, query, context_docs):
        """Constructs a prompt with the context and asks Gemini to answer."""
        
        if not context_docs:
            return "I don't have any information about that in my notes.", []

        # Build context string
        context_str = ""
        sources_used = set()
        
        for doc in context_docs:
            context_str += f"- {doc['text']}\n"
            sources_used.add(doc['source'])

        prompt = f"""
You are "LoreKeeper", an assistant for a Dungeon Master running a D&D campaign.
Use the following notes from past sessions to answer the question.
If the answer is not contained in the notes, say that you don't know based on the provided lore.
Keep your answer concise and helpful for a DM mid-game.

CAMPAIGN NOTES CONTEXT:
{context_str}

DM'S QUESTION:
{query}

YOUR ANSWER:
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text, list(sources_used)
        except Exception as e:
            return f"Error generating answer: {e}", []

