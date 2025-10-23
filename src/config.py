import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class Settings:
    EMBEDDING_BACKEND: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "openai"))
    OPENAI_EMBEDDING_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    LLM_PROVIDER: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    OPENAI_LLM_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"))
    CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", 500)))
    CHUNK_OVERLAP: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", 100)))
    TOP_K: int = field(default_factory=lambda: int(os.getenv("TOP_K", 4)))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
