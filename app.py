import streamlit as st
from src.rag_engine import DocumentLoader, LoreDatabase, AnswerGenerator

# --- UI Configuration ---
st.set_page_config(page_title="LoreKeeper AI", page_icon="🐉", layout="centered")

st.title("🐉 LoreKeeper: Campaign Memory")
st.markdown("*A Retrieval-Augmented Generation assistant for Dungeon Masters.*")

# --- Initialize RAG Components ---
@st.cache_resource
def get_database():
    return LoreDatabase()

@st.cache_resource
def get_generator():
    try:
        return AnswerGenerator()
    except ValueError as e:
        st.error(f"Configuration Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize AI Model: {e}")
        st.stop()

db = get_database()
generator = get_generator()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    if st.button("Sync Notes (Re-Index Data)"):
        with st.spinner("Reading markdown notes and generating embeddings..."):
            loader = DocumentLoader()
            docs, metadatas, ids = loader.load_and_chunk()
            
            if len(docs) == 0:
                st.warning("No notes found in the `data/` directory.")
            else:
                num_indexed = db.add_documents(docs, metadatas, ids)
                st.success(f"Successfully indexed {num_indexed} lore chunks!")

    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Drop your `.md` or `.txt` session notes into the `data/` folder.
    2. Click **Sync Notes** above.
    3. Ask questions about your campaign in the chat!
    """)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"Sources: {', '.join(msg['sources'])}")

# Chat input
if query := st.chat_input("Ask a lore question (e.g., 'Who is the barkeep at The Prancing Pony?'):"):
    # Show user message
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Consulting the ancient tomes..."):
            # 1. Retrieve Context
            context = db.search(query, top_k=3)
            
            # 2. Generate Answer
            answer, sources = generator.generate_answer(query, context)
            
            # Display answer
            st.markdown(answer)
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
