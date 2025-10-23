import time
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from src.config import Settings
from src.loaders import load_documents_from_paths
from src.splitter import make_splitter
from src.embeddings import get_embedder
from src.vectorstore import VectorStore
from src.qa import build_qa_chain
from src.utils import ensure_dirs

# Load environment variables
load_dotenv()

st.set_page_config(page_title="ModiFace DocQA", layout="wide")
st.title("ModiFace Document QA Agent")
st.markdown("Upload PDFs or TXT files, build an index, and ask grounded questions.")

# Initialize configuration
settings = Settings()
ensure_dirs(["data/uploads", "data/index"])
index_dir = Path("data/index")
upload_dir = Path("data/uploads")
vs = None

# ==============================
# Upload and Index Documents
# ==============================
st.subheader("1. Upload & Index Documents")

files = st.file_uploader(
    "Upload PDFs or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

col1, _ = st.columns([1, 1])
with col1:
    if st.button("Clear index"):
        VectorStore.clear(index_dir)
        st.success("Index cleared.")

# Automatically index as soon as files are uploaded
if files:
    saved_paths = []
    for f in files:
        dest = upload_dir / f.name
        with open(dest, "wb") as out:
            out.write(f.read())
        saved_paths.append(dest)

    st.info(f"Loaded {len(saved_paths)} documents.")
    docs = load_documents_from_paths(saved_paths)

    splitter = make_splitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    st.write(f"Split into {len(chunks)} chunks.")

    embedder = get_embedder(settings)
    vs = VectorStore.persist_or_create(index_dir, embedder)

    start = time.time()
    VectorStore.add_documents(vs, chunks, index_dir)
    end = time.time()
    st.success(f"Indexed {len(chunks)} chunks in {end - start:.2f}s.")

# Always load existing vector store if available
if vs is None:
    embedder = get_embedder(settings)
    vs = VectorStore.persist_or_create(index_dir, embedder)

# ==============================
# Question Answering
# ==============================
st.subheader("Ask a Question")
query = st.text_input("Your question:")

if st.button("Get Answer") and query:
    embedder = get_embedder(settings)
    vs = VectorStore.persist_or_create(index_dir, embedder)
    retriever = vs.as_retriever(search_kwargs={"k": settings.TOP_K})
    qa = build_qa_chain(settings, retriever)

    with st.spinner("Thinking..."):
        result = qa(query)

    st.markdown("### Answer")
    st.write(result.answer)

    st.markdown("### Sources")
    for i, src in enumerate(result.sources, 1):
        with st.expander(f"Source {i}"):
            st.write(src.metadata.get("source", "unknown"))
            st.code(src.page_content[:1000])

st.caption("Built with Streamlit, LangChain, and FAISS")
