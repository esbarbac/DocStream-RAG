from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

def load_documents_from_paths(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
        else:
            loader = TextLoader(str(p), autodetect_encoding=True)
        docs.extend(loader.load())
        
    for d in docs:
        d.metadata["source"] = str(d.metadata.get("source", p.name))
    return docs
