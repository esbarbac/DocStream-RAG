from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
import faiss

class VectorStore:
    @staticmethod
    def persist_or_create(index_dir: Path, embeddings) -> FAISS:
        """
        Load an existing FAISS index if it exists and matches the embedding dimension,
        otherwise recreate it safely.
        """
        index_dir.mkdir(parents=True, exist_ok=True)
        index_file = index_dir / "index.faiss"
        store_file = index_dir / "index.pkl"

        # Try to load an existing index
        if index_file.exists() and store_file.exists():
            try:
                store = FAISS.load_local(
                    str(index_dir),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                # Check dimension compatibility
                test_vec = embeddings.embed_query("dimension check")
                if store.index.d != len(test_vec):
                    print("FAISS dimension mismatch detected. Rebuilding index.")
                    for file in index_dir.glob("*"):
                        file.unlink()
                    raise ValueError("FAISS index dimension mismatch.")
                return store
            except Exception as e:
                print(f"Recreating FAISS index due to error: {e}")
                for file in index_dir.glob("*"):
                    file.unlink()

        # Create new FAISS index
        test_vector = embeddings.embed_query("test")
        dim = len(test_vector)
        index = faiss.IndexFlatL2(dim)

        vs = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )

        # Helper methods
        def _is_empty(self):
            return len(self.index_to_docstore_id) == 0

        def _count(self):
            return len(self.index_to_docstore_id)

        FAISS.is_empty = _is_empty
        FAISS.count = _count

        return vs

    @staticmethod
    def clear(index_dir: Path):
        """Remove all index files."""
        for file in index_dir.glob("*"):
            file.unlink()

    @staticmethod
    def add_documents(vs: FAISS, docs: List[Document], index_dir: Path):
        """Add and persist new documents."""
        if not docs:
            print("No documents to add to vector store.")
            return
        vs.add_documents(docs)
        vs.save_local(str(index_dir))
