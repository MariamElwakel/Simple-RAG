import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Page Config
st.set_page_config(
    page_title="Arabic RAG App",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Arabic RAG Question Answering")
st.write("Ask questions based on your Arabic document using RAG.")


#ENV
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Fixed Chunking 
def fixed_chunking(text, chunk_size=800, chunk_overlap=100):

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap

    return chunks


# Create Documents
def create_documents(chunks):

    documents = []

    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i
                }
            )
        )

    return documents


# Build Vector Store
@st.cache_resource
def build_vector_store(text):

    chunks = fixed_chunking(text)
    documents = create_documents(chunks)

    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )

    persist_dir = "vector_store"

    if os.path.exists(persist_dir):
        # Load existing DB instead of deleting
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function
        )
    else:
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=persist_dir
        )
        vector_store.persist()

    return vector_store, len(chunks)


# Load LLM
@st.cache_resource
def load_llm():

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

    return llm


# Sidebar — File Upload
st.sidebar.header("📂 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload Arabic TXT file",
    type=["txt"]
)

text_data = None

if uploaded_file:
    text_data = uploaded_file.read().decode("utf-8")
    st.sidebar.success("File uploaded successfully ✅")

elif os.path.exists("arabic.txt"):
    with open("arabic.txt", "r", encoding="utf-8") as f:
        text_data = f.read()
    st.sidebar.info("Using local arabic.txt file")

else:
    st.warning("Upload a file or place arabic.txt in the project folder")
    st.stop()


# Build or Load Vector Store
if "vector_store" not in st.session_state:

    with st.spinner("Building vector database..."):

        vector_store, num_chunks = build_vector_store(text_data)

        st.session_state.vector_store = vector_store
        st.session_state.num_chunks = num_chunks

else:
    vector_store = st.session_state.vector_store
    num_chunks = st.session_state.num_chunks


st.success(f"Vector DB ready — {num_chunks} chunks created")


# Question Input
query = st.text_input("💬 Ask a question from the document:")

k = st.slider("Number of retrieved chunks", 1, 5, 3)


# Ask Button
if st.button("Get Answer") and query:

    llm = load_llm()

    results = vector_store.similarity_search_with_score(query, k=k)

    docs = [doc for doc, score in results]

    context = "\n\n".join(
        f"[Chunk ID: {d.metadata['chunk_id']}]\n{d.page_content}"
        for d in docs
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say:
"Not specified in the context."

Context:
{context}

Question:
{query}

Answer clearly.
"""

    with st.spinner("Generating answer..."):
        response = llm.invoke(prompt)

    st.subheader("📌 Answer")
    st.write(response.content)


    # Show retrieved chunks
    with st.expander("🔎 Retrieved Chunks"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Rank {i} — Chunk ID {d.metadata['chunk_id']}**")
            st.write(d.page_content)
            st.divider()