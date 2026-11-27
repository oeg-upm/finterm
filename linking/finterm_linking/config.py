"""Execution policy around frozen algorithms, never overrides their recipes."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys


@dataclass(frozen=True)
class LinkerConfig:
    work_dir: Path = field(default_factory=lambda: Path.cwd() / "finterm_runs")
    code_dir: Path = field(default_factory=lambda: Path(__file__).parent / "_vendor/terminlink_en")
    mode: str = "quick"
    similarity_mode: str = "prepare"
    offline: bool = True
    search_policy: str = "historical"
    translation_policy: str = "historical"
    preselection_policy: str = "conservative"
    lemma_map: Path | None = None
    wikidata_seed_cache: Path | None = None
    archived_similarity_dir: Path | None = None
    qwen14_hub_cache: Path | None = None
    embedding_resource_dir: Path | None = None
    translation_cache: Path | None = None
    defterm_cache: Path | None = None
    embedding_cache: Path | None = None
    gpu_lock: Path | None = None
    max_translations: int = 32
    max_gate_calls: int = 16
    max_definition_calls: int = 8
    max_new_embeddings: int = 256
    translation_token_limit: int | None = None
    gate_token_limit: int | None = None
    definition_token_limit: int | None = None
    http_contact: str | None = None
    language: str = "en"
    term_origin: str = "user"

    def __post_init__(self):
        if self.mode not in {"quick", "full"}:
            raise ValueError("mode must be quick or full")
        if self.similarity_mode not in {"prepare", "archived_replay", "h100"}:
            raise ValueError("execution must be prepare, archived_replay or h100")
        if self.mode == "full" and self.similarity_mode != "h100":
            raise ValueError("full requires h100 execution; preparation/replay is not a full inference run")
        if self.similarity_mode == "h100" and sys.platform == "win32":
            raise ValueError("Real inference is disabled on Windows; use the validated H100 environment")
        if self.similarity_mode == "h100" and self.embedding_resource_dir is None:
            raise ValueError("h100 requires an explicit embedding_resource_dir")
        if (self.similarity_mode == "archived_replay") != (self.archived_similarity_dir is not None):
            raise ValueError("archived_similarity_dir is exclusive to archived_replay")
        if self.search_policy not in {"historical", "coverage"}:
            raise ValueError("Use an existing retrieval policy: historical or coverage")
        if self.translation_policy not in {"historical", "multilingual"}:
            raise ValueError("Use an existing Sense-2 policy: historical or multilingual")
        if self.preselection_policy not in {"conservative", "direct_only"}:
            raise ValueError("Use an existing DefTerm preselection policy")
        if type(self.offline) is not bool or self.language != "en":
            raise ValueError("This distribution accepts English text; offline must be bool")
        if not self.term_origin or any(char in self.term_origin for char in "\r\n"):
            raise ValueError("term_origin must be an explicit single-line label")
        for name in ("max_translations", "max_gate_calls", "max_definition_calls", "max_new_embeddings"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be a finite nonnegative integer")
        for name in ("translation_token_limit", "gate_token_limit", "definition_token_limit"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(name + " must be a positive original-contract value")
        for name in ("work_dir", "code_dir", "lemma_map", "wikidata_seed_cache", "archived_similarity_dir",
                     "qwen14_hub_cache", "embedding_resource_dir", "translation_cache", "defterm_cache",
                     "embedding_cache", "gpu_lock"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())

    def budgets(self):
        return {"translations": self.max_translations if self.mode == "full" else 0,
                "gate": self.max_gate_calls if self.mode == "full" else 0,
                "definitions": self.max_definition_calls if self.mode == "full" else 0,
                "embeddings": self.max_new_embeddings if self.similarity_mode == "h100" else 0}

    def validate_recipes(self, runtime):
        from finterm_defterm import recipe
        from finterm_sense2_qwen import generation_contract
        actual = {"translation_token_limit": generation_contract()["generation"]["max_new_tokens"],
                  "gate_token_limit": recipe("gate")["generation"]["max_new_tokens"],
                  "definition_token_limit": recipe("definition")["generation"]["max_new_tokens"]}
        for name, expected in actual.items():
            if getattr(self, name) not in (None, expected):
                raise ValueError(name + " differs from the original fixed contract; use inference budgets instead")
        return actual

    def to_dict(self):
        return {name: str(value) if isinstance(value, Path) else value for name, value in asdict(self).items()}
