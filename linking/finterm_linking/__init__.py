"""Original FINTERM pipeline; original text + list[str], no extractor dependency."""
from .config import LinkerConfig
from .orchestrator import Linker, LinkResult

__all__ = ["Linker", "LinkerConfig", "LinkResult", "link"]
__version__ = "0.1.0"


def link(text, terms, *, config=None, mode=None, top_k=5, document_id=None):
    """Convenience facade. Configure local resources explicitly for full/H100."""
    return Linker(config or LinkerConfig()).link(text, terms, mode=mode, top_k=top_k, document_id=document_id)
