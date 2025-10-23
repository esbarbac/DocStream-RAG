from langchain_openai import OpenAIEmbeddings
from .config import Settings

def get_embedder(settings: Settings):
    """Return OpenAI embedder using model and API key from settings."""
    if not settings.OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY.")
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY
    )
