"""Pinned Qwen3-Embedding-4B, official Sentence Transformers pipeline and cosine.

Provisioning is explicit and separate from offline inference. Importing this
module imports no ML runtime, installs nothing and downloads no resources.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

from finterm_sense2 import BASE, canonical, file_hash, object_hash, require


MODEL_ID = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
ST_VERSION = "5.2.0"
DIMENSION = 2560
MAX_SEQUENCE_LENGTH = 32768
RESOURCE_SCHEMA = "finterm-qwen3-embedding-resources-v1"
BACKEND_KIND = "QWEN3_EMBEDDING4B_SENTENCE_TRANSFORMERS"
TEST_BACKEND = "CONTROLLED_EMBEDDINGS_NO_MODEL"
# Official pinned Hub file metadata, not model weights. SHA-256 is supplied for
# LFS files; small files have the provider's blob SHA-1 (computed without Git).
FILES = {
    "1_Pooling/config.json": (313, "blob_sha1", "81de5602eacbce382009c5af7a23085871801d8f"),
    "config.json": (727, "blob_sha1", "8b4b87fc69023e7a224eb6563753aaf3223d8b98"),
    "config_sentence_transformers.json": (215, "blob_sha1", "76aef3ade63553ebb698fe3c2a3264040ed093f8"),
    "generation_config.json": (117, "blob_sha1", "d46f1983345269c582611bbedb3ca0a13f8e5f7b"),
    "merges.txt": (1671853, "blob_sha1", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
    "model-00001-of-00002.safetensors": (4965826464, "sha256", "e70bfe3c970523fb7ef4eddffed2254ce3f1e7150c3de2af4342de129dd756f8"),
    "model-00002-of-00002.safetensors": (3077765624, "sha256", "ed1b87c8e9eb7e535a1a155e4fd00d9f4dba80e58a6db48a4c9f82cede7079c1"),
    "model.safetensors.index.json": (30431, "blob_sha1", "3d736ef26714eee0abde3e05104ee1b3ec26c974"),
    "modules.json": (349, "blob_sha1", "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
    "tokenizer.json": (11422947, "sha256", "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d"),
    "tokenizer_config.json": (7256, "blob_sha1", "df3a9d96759529ca1006eb6db024bbb099a97578"),
    "vocab.json": (2776833, "blob_sha1", "4783fe10ac3adce15ac8f358ef5462739852c569"),
}


def check_environment(*, require_python311=True):
    """Metadata-only preflight; does not import Torch/ST or initialize CUDA."""
    expected = {"torch": "2.10.0+cu126", "transformers": "4.57.6", "accelerate": "1.15.0",
                "bitsandbytes": "0.50.2", "sentence-transformers": ST_VERSION}
    installed = {}
    for name in (*expected, "huggingface-hub", "safetensors"):
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    issues = ["DEPENDENCY_VERSION:" + name for name, wanted in expected.items() if installed[name] != wanted]
    if require_python311 and sys.version_info[:2] != (3, 11):
        issues.append("PYTHON_3_11_REQUIRED_FOR_VALIDATED_H100_PROFILE")
    return {"python": sys.version.split()[0], "packages": installed, "expected": expected,
            "compatible": not issues, "issues": issues, "ml_imported_by_preflight": False}


def contract():
    return {"backend_kind": BACKEND_KIND, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "sentence_transformers_version": ST_VERSION, "transformers_version": "4.57.6", "torch_version": "2.10.0+cu126",
            "embedding_dimension": DIMENSION, "dtype": "bfloat16", "padding_side": "left", "attention": "sdpa",
            "pooling": "OFFICIAL_SENTENCE_TRANSFORMERS_LAST_TOKEN", "normalize_embeddings": True,
            "prompt_policy": "SYMMETRIC_STS_EMPTY_PROMPT_V1", "prompt": "", "similarity": "cosine",
            "max_sequence_length": MAX_SEQUENCE_LENGTH, "truncation_policy": "REJECT_NEVER_SILENTLY_TRUNCATE",
            "resource_index_sha256": object_hash(FILES), "adapter_sha256": file_hash(Path(__file__))}


@contextmanager
def exclusive_lock(path):
    """An OS lock is released on crashes; no stale-owner deletion is needed."""
    stream = Path(path).open("a+b")
    locked = False
    try:
        if stream.seek(0, 2) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _verify_files(snapshot):
    records = {}
    for name, (size, algorithm, expected) in FILES.items():
        path = snapshot / name
        require(path.is_file() and path.stat().st_size == size, "Embedding resource missing/wrong size: " + name)
        sha256 = hashlib.sha256()
        blob = hashlib.sha1(("blob " + str(size) + "\0").encode("ascii"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(block)
                blob.update(block)
        actual = sha256.hexdigest()
        require((actual if algorithm == "sha256" else blob.hexdigest()) == expected, "Pinned embedding resource digest differs: " + name)
        records[name] = {"size": size, "sha256": actual}
    modules = json.loads((snapshot / "modules.json").read_text(encoding="utf-8"))
    require([row["type"] for row in modules] == ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling",
                                                "sentence_transformers.models.Normalize"], "Official embedding pipeline differs")
    pooling = json.loads((snapshot / "1_Pooling/config.json").read_text(encoding="utf-8"))
    require(pooling["pooling_mode_lasttoken"] is True and pooling["word_embedding_dimension"] == DIMENSION, "Official last-token pooling differs")
    settings = json.loads((snapshot / "config_sentence_transformers.json").read_text(encoding="utf-8"))
    require(settings["similarity_fn_name"] == "cosine" and settings["default_prompt_name"] is None, "Official ST similarity/default prompt differs")
    return records


def _snapshot(root, manifest):
    relative = manifest.get("snapshot_relative_path")
    if relative is not None:
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Unsafe resource snapshot location")
        snapshot = (root / path).resolve()
    else:
        snapshot = Path(manifest["external_snapshot_path"]).resolve()
    require(snapshot.name == MODEL_REVISION and snapshot.parent.name == "snapshots", "Resource directory does not identify pinned revision")
    return snapshot


def verify_resources(resource_dir):
    root = Path(resource_dir).resolve()
    manifest = json.loads((root / "embedding_resources.json").read_text(encoding="utf-8"))
    require(manifest["schema_version"] == RESOURCE_SCHEMA and manifest["model_id"] == MODEL_ID
            and manifest["model_revision"] == MODEL_REVISION and manifest["resource_index_sha256"] == object_hash(FILES), "Embedding resource manifest differs")
    snapshot = _snapshot(root, manifest)
    records = _verify_files(snapshot)
    require(records == manifest["files"] and manifest["content_sha256"] == object_hash(records), "Embedding local resource manifest hash differs")
    return snapshot, manifest


def provision_resources(resource_dir, *, hub_cache_dir=None):
    """Explicit one-time setup. Reuses a verified snapshot without network calls."""
    environment = check_environment()
    require(environment["compatible"], "Embedding environment is incompatible; inspect check_environment() before setup")
    root = Path(resource_dir).resolve()
    require(root.parent.is_dir() and root != BASE and BASE not in root.parents, "Keep model resources outside the source project in an existing persistent parent")
    root.mkdir(exist_ok=True)
    with exclusive_lock(root / ".provision.lock"):
        if (root / "embedding_resources.json").exists():
            _, manifest = verify_resources(root)
            return manifest
        from huggingface_hub import snapshot_download, try_to_load_from_cache

        hub = Path(hub_cache_dir).resolve() if hub_cache_dir is not None else root / "hub"
        require(hub != BASE and BASE not in hub.parents, "Do not download weights into the source project")
        hub.mkdir(parents=True, exist_ok=True)
        config = try_to_load_from_cache(MODEL_ID, "config.json", revision=MODEL_REVISION, cache_dir=str(hub))
        snapshot = Path(config).parent if isinstance(config, str) else None
        complete = snapshot is not None and all((snapshot / name).is_file() for name in FILES)
        if not complete:
            snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, cache_dir=str(hub),
                                             allow_patterns=list(FILES), max_workers=2))
        require(snapshot.name == MODEL_REVISION, "Downloaded snapshot revision differs")
        records = _verify_files(snapshot)
        local = root in snapshot.resolve().parents
        manifest = {"schema_version": RESOURCE_SCHEMA, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                    "snapshot_relative_path": snapshot.resolve().relative_to(root).as_posix() if local else None,
                    "external_snapshot_path": None if local else str(snapshot.resolve()), "resource_index_sha256": object_hash(FILES),
                    "content_sha256": object_hash(records), "files": records}
        descriptor, name = tempfile.mkstemp(prefix=".embedding-resources-", suffix=".tmp", dir=root)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(canonical(manifest) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, root / "embedding_resources.json")
        finally:
            if temporary.exists():
                temporary.unlink()
        return manifest


class InputTooLong(ValueError):
    def __init__(self, indices):
        super().__init__("Text exceeds the pinned model context; not silently truncated")
        self.indices = tuple(indices)


def official_cosine(context_vector, representation_vectors):
    require(check_environment()["compatible"], "Embedding scoring environment incompatible")
    import torch
    from sentence_transformers.util import cos_sim
    contexts = torch.tensor([context_vector], dtype=torch.float32, device="cpu")
    documents = torch.tensor(representation_vectors, dtype=torch.float32, device="cpu")
    return cos_sim(contexts, documents)[0].tolist()


class QwenEmbedding4B:
    """Official ST encoder; no automatic provisioning or GPU model coexistence."""
    def __init__(self, resource_dir, *, batch_size=1, gpu_reserve_gib=8.0, allow_shared_gpu=False):
        require(type(batch_size) is int and batch_size > 0, "Embedding batch size must be positive")
        require(isinstance(gpu_reserve_gib, (int, float)) and gpu_reserve_gib >= 0, "Invalid GPU memory reserve")
        self.resource_dir = Path(resource_dir)
        self.batch_size, self.gpu_reserve_gib, self.allow_shared_gpu = batch_size, float(gpu_reserve_gib), allow_shared_gpu
        self.identity = contract()
        self.model = None
        self.runtime = None

    def prepare(self):
        if self.model is not None:
            return
        require(check_environment()["compatible"], "Embedding environment incompatible; no automatic package changes")
        snapshot, resources = verify_resources(self.resource_dir)
        import torch
        from sentence_transformers import SentenceTransformer

        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and "H100" in torch.cuda.get_device_name(0).upper(), "H100/bfloat16 is required")
        require(str(torch.__version__) == "2.10.0+cu126" and torch.version.cuda == "12.6", "Validated CUDA PyTorch build differs")
        free, total = torch.cuda.mem_get_info(0)
        allocated, reserved = torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0)
        weights = sum(size for name, (size, _, _) in FILES.items() if name.endswith(".safetensors"))
        required = weights + int(self.gpu_reserve_gib * 1024 ** 3)
        require(self.allow_shared_gpu or allocated == 0 and reserved == 0, "GPU already used in this process; release 14B models or use a clean kernel")
        require(free >= required, "Insufficient free GPU memory for embedding weights and configured activation reserve")
        model = SentenceTransformer(str(snapshot), device="cuda:0", local_files_only=True, trust_remote_code=False,
                    similarity_fn_name="cosine", model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
                    tokenizer_kwargs={"padding_side": "left"})
        model.max_seq_length = MAX_SEQUENCE_LENGTH
        model.eval()
        require(model.get_sentence_embedding_dimension() == DIMENSION and model.similarity_fn_name == "cosine", "Loaded ST model shape/similarity differs")
        self.model = model
        self.runtime = {"backend_kind": BACKEND_KIND, "python": sys.version.split()[0], "packages": check_environment()["packages"],
                        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "resource_content_sha256": resources["content_sha256"],
                        "device_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda, "dtype": "bfloat16",
                        "batch_size": self.batch_size, "allow_shared_gpu": self.allow_shared_gpu,
                        "gpu_memory_before_load": {"free_bytes": free, "total_bytes": total, "allocated_bytes": allocated,
                                                   "reserved_bytes": reserved, "required_bytes": required}, "local_files_only": True}

    def encode(self, texts):
        self.prepare()
        oversized = []
        for index, text in enumerate(texts):
            ids = self.model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            if len(ids) > MAX_SEQUENCE_LENGTH:
                oversized.append(index)
        if oversized:
            raise InputTooLong(oversized)
        values = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_tensor=True, normalize_embeddings=True,
                                   show_progress_bar=False, prompt="", prompt_name=None)
        from sentence_transformers.util import normalize_embeddings
        # Retain unit length after BF16 -> portable float32 cache conversion.
        return normalize_embeddings(values.float()).cpu().tolist()

    def similarity(self, context_vector, representation_vectors):
        # Same official function selected by SentenceTransformer.similarity when
        # similarity_fn_name='cosine'; cache-only scoring need not load weights.
        return official_cosine(context_vector, representation_vectors)

    def close(self):
        if self.model is not None:
            self.model = None
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
