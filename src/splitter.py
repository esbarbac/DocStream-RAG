from langchain_text_splitters import RecursiveCharacterTextSplitter

def make_splitter(chunk_size: int = 300, chunk_overlap: int = 50):
    """Splitter tuned for short letters and essays."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", "?", "!"]
    )
