"""Tiny simulated HF snapshots. No real weights, downloads or inference.

Uses native symlinks where permitted. On Windows without symlink privilege,
only filesystem link metadata is simulated; targets/data remain real fixtures.
"""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from llm_adapters import AdapterProfile
from extractor import FinancialExtractor
from extractor.financial import ADAPTERS
from experimental_evaluation.provenance import model_inventory


class Links:
    native_total = 0
    simulated_total = 0

    def __init__(self):
        self.virtual = {}

    @staticmethod
    def absolute(path):
        return Path(os.path.abspath(path))

    def link(self, path, target, directory=False):
        path, target = Path(path), Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(target, target_is_directory=directory)
            Links.native_total += 1
        except (OSError, NotImplementedError) as exc:
            if os.name != "nt":
                raise
            # No privilege escalation or production bypass. A placeholder is
            # needed only for from_local's preliminary config.is_file check.
            if directory:
                path.mkdir()
            else:
                path.write_bytes(b"synthetic link placeholder; not model data")
            self.virtual[self.absolute(path)] = target
            Links.simulated_total += 1

    def remove(self, path):
        path = Path(path)
        if self.virtual.pop(self.absolute(path), None) is not None and path.is_dir():
            path.rmdir()
        else:
            path.unlink()

    @contextmanager
    def active(self):
        cls = type(Path.cwd())
        is_link, read_link, resolve = cls.is_symlink, cls.readlink, cls.resolve
        def linked(path):
            return self.absolute(path) in self.virtual or is_link(path)
        def target(path):
            value = self.virtual.get(self.absolute(path))
            return value if value is not None else read_link(path)
        def resolved(path, *args, **kwargs):
            seen = set()
            current = self.absolute(path)
            while current in self.virtual:
                if current in seen:
                    raise RuntimeError("synthetic symlink loop")
                seen.add(current)
                value = self.virtual[current]
                current = self.absolute(value if value.is_absolute() else current.parent / value)
            return resolve(current, *args, **kwargs)
        with patch.object(cls, "is_symlink", linked), patch.object(cls, "readlink", target), patch.object(cls, "resolve", resolved):
            yield


class HfCacheTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        print(f"HF cache symlink fixtures: native={Links.native_total}, Windows-simulated={Links.simulated_total}")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("FINTERM_TEST_TMP"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.links = Links()
        self.active = self.links.active(); self.active.__enter__()
        self.addCleanup(self.active.__exit__, None, None, None)

    def profile(self, model, source):
        base = ADAPTERS[model]()
        return AdapterProfile(model, base.model_id,
                              replace(base.load_config, source=str(source), revision="b" * 40, local_files_only=True),
                              base.generation_options)

    def cache(self, model="qwen3", *, metadata_links=True, shard_links=True, sharded=True, repository=None):
        base = ADAPTERS[model]()
        repo = self.root / (repository or "models--" + base.model_id.replace("/", "--"))
        snapshot, blobs = repo / "snapshots" / ("b" * 40), repo / "blobs"
        snapshot.mkdir(parents=True); blobs.mkdir()
        count = {"qwen3": 8, "gemma2": 4, "dister": 7}[model] if sharded else 1
        shard_names = [f"model-{n:05d}-of-{count:05d}.safetensors" for n in range(1, count + 1)] if sharded else ["model.safetensors"]
        targets = {}
        def asset(name, content, linked):
            if linked:
                blob = blobs / hashlib.sha256(name.encode() + content).hexdigest()
                blob.write_bytes(content)
                self.links.link(snapshot / name, Path("../../blobs") / blob.name)
                targets[name] = blob
            else:
                (snapshot / name).write_bytes(content)
                targets[name] = snapshot / name
        asset("config.json", json.dumps({"model_type": {"qwen3": "qwen3", "gemma2": "gemma2", "dister": "llama"}[model]}).encode(), metadata_links)
        asset("tokenizer_config.json", b"{}", metadata_links)
        asset("tokenizer.json", b"{}", metadata_links)
        if sharded:
            asset("model.safetensors.index.json", json.dumps({"weight_map": {f"weight_{n}": name for n, name in enumerate(shard_names)}}).encode(), metadata_links)
        for n, name in enumerate(shard_names):
            asset(name, b"synthetic-not-safetensors-weights-" + str(n).encode(), shard_links)
        return snapshot, blobs, self.profile(model, snapshot), shard_names, targets

    def replace_link(self, path, target, directory=False):
        self.links.remove(path)
        self.links.link(path, target, directory)

    def update_index(self, snapshot, mapping):
        path = snapshot / "model.safetensors.index.json"
        self.links.remove(path)
        path.write_text(json.dumps({"weight_map": mapping}), encoding="utf-8")

    def test_official_three_model_snapshots_with_metadata_and_shards_in_same_blobs(self):
        for model, expected in (("qwen3", 8), ("gemma2", 4), ("dister", 7)):
            with self.subTest(model=model):
                snapshot, blobs, profile, names, _ = self.cache(model)
                inventory = model_inventory(profile, allow_hf_cache_blobs=True)
                self.assertEqual(len(inventory["weight_shards"]), expected)
                self.assertEqual(inventory["local_file_policy"]["authorized_blobs"], str(blobs))
                self.assertFalse(inventory["weight_content_hashes_verified"])
                with patch("llm_adapters.adapters.load_huggingface_runtime", side_effect=AssertionError("No model may load")) as loader:
                    engine = FinancialExtractor.from_local(model, snapshot)
                self.assertFalse(engine.adapter.is_loaded)
                self.assertEqual(engine.adapter.generation_options, profile.generation)
                self.assertEqual(len(engine.model_files["weight_shards"]), expected)
                loader.assert_not_called()

    def test_historical_default_does_not_authorize_external_blobs(self):
        _, _, profile, _, _ = self.cache()
        with self.assertRaises(ValueError):
            model_inventory(profile)
        with self.assertRaises(TypeError):
            model_inventory(profile, allow_hf_cache_blobs="yes")

    def test_copied_directory_result_stays_compatible(self):
        snapshot, _, profile, _, _ = self.cache(metadata_links=False, shard_links=False)
        strict = model_inventory(profile)
        opted = model_inventory(profile, allow_hf_cache_blobs=True)
        opted.pop("local_file_policy")
        self.assertEqual(strict, opted)

    def test_single_safetensors_file_link_is_supported(self):
        _, _, profile, _, _ = self.cache("dister", sharded=False)
        result = model_inventory(profile, allow_hf_cache_blobs=True, hash_weights=True)
        self.assertEqual(set(result["weight_shards"]), {"model.safetensors"})
        self.assertEqual(len(result["weight_shards"]["model.safetensors"]["sha256"]), 64)

    def test_a_link_to_the_snapshot_directory_is_supported(self):
        snapshot, _, profile, _, _ = self.cache()
        alias = self.root / "local-model-alias"
        self.links.link(alias, snapshot, directory=True)
        engine = FinancialExtractor.from_local("qwen3", alias)
        self.assertEqual(engine.model_files["local_file_policy"]["snapshot_root"], str(snapshot))

    def test_optional_full_weight_hash_is_of_the_real_blob(self):
        _, _, profile, names, targets = self.cache()
        result = model_inventory(profile, allow_hf_cache_blobs=True, hash_weights=True)
        self.assertEqual(result["weight_shards"][names[0]]["sha256"], hashlib.sha256(targets[names[0]].read_bytes()).hexdigest())

    def test_broken_missing_or_empty_shards_are_rejected(self):
        snapshot, blobs, profile, names, targets = self.cache()
        for kind in ("broken", "missing", "empty"):
            with self.subTest(kind=kind):
                path = snapshot / names[0]
                self.links.remove(path)
                if kind == "broken":
                    self.links.link(path, Path("../../blobs/does-not-exist"))
                elif kind == "empty":
                    path.write_bytes(b"")
                with self.assertRaises(ValueError):
                    model_inventory(profile, allow_hf_cache_blobs=True)
                if kind != "missing":
                    self.links.remove(path)
                self.links.link(path, Path("../../blobs") / targets[names[0]].name)

    def test_external_and_other_repository_targets_are_rejected(self):
        snapshot, _, profile, names, targets = self.cache()
        external = self.root / "outside"; external.mkdir()
        other = self.root / "models--other--model/blobs"; other.mkdir(parents=True)
        for destination in (external / "weight", other / "weight"):
            destination.write_bytes(b"untrusted synthetic weights")
            self.replace_link(snapshot / names[0], destination)
            with self.assertRaisesRegex(ValueError, "External"):
                model_inventory(profile, allow_hf_cache_blobs=True)

    def test_intermediate_external_link_is_rejected_even_if_it_points_back(self):
        snapshot, _, profile, names, targets = self.cache()
        relay = self.root / "external-relay"
        self.links.link(relay, targets[names[0]])
        self.replace_link(snapshot / names[0], relay)
        with self.assertRaisesRegex(ValueError, "External"):
            model_inventory(profile, allow_hf_cache_blobs=True)

    def test_unsafe_index_names_rejected_before_resolution(self):
        snapshot, _, profile, _, _ = self.cache()
        names = ["../weights.safetensors", "sub/../weights.safetensors", "../../blobs/weight",
                 "/outside.safetensors", r"..\weight.safetensors", r"C:\weight.safetensors", "C:weight.safetensors",
                 "a:stream", "", "sub//weight.safetensors", "./weight.safetensors", "weight.safetensors ", None, []]
        for name in names:
            with self.subTest(name=name):
                self.update_index(snapshot, {"weight": name})
                with self.assertRaises(ValueError):
                    model_inventory(profile, allow_hf_cache_blobs=True)

    def test_parent_directory_links_are_not_accepted(self):
        snapshot, blobs, profile, names, targets = self.cache()
        parent = snapshot / "weights"
        self.links.link(parent, blobs, directory=True)
        self.update_index(snapshot, {"weight": "weights/" + targets[names[0]].name})
        with self.assertRaisesRegex(ValueError, "parent directory"):
            model_inventory(profile, allow_hf_cache_blobs=True)

    def test_blobs_directory_cannot_be_redirected(self):
        snapshot, blobs, profile, _, _ = self.cache()
        relocated = blobs.with_name("saved-blobs")
        blobs.rename(relocated)
        self.links.link(blobs, relocated, directory=True)
        with self.assertRaisesRegex(ValueError, "directory cannot itself be a link"):
            model_inventory(profile, allow_hf_cache_blobs=True)

    def test_cycles_fail_closed(self):
        snapshot, blobs, profile, names, _ = self.cache()
        self.links.link(blobs / "cycle1", Path("cycle2"))
        self.links.link(blobs / "cycle2", Path("cycle1"))
        self.replace_link(snapshot / names[0], Path("../../blobs/cycle1"))
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            model_inventory(profile, allow_hf_cache_blobs=True)

    def test_external_or_broken_metadata_links_are_rejected(self):
        snapshot, _, profile, _, _ = self.cache()
        outside = self.root / "outside-config.json"
        outside.write_text('{"model_type":"qwen3"}', encoding="utf-8")
        self.replace_link(snapshot / "config.json", outside)
        with self.assertRaisesRegex(ValueError, "External"):
            model_inventory(profile, allow_hf_cache_blobs=True)
        self.replace_link(snapshot / "config.json", Path("../../blobs/missing-config"))
        with self.assertRaises(ValueError):
            model_inventory(profile, allow_hf_cache_blobs=True)

    def test_unknown_cache_layout_does_not_authorize_sibling_blobs(self):
        _, _, profile, _, _ = self.cache(repository="models--unrelated--repo")
        with self.assertRaises(ValueError):
            model_inventory(profile, allow_hf_cache_blobs=True)

    def test_empty_blob_is_rejected(self):
        _, _, profile, names, targets = self.cache()
        targets[names[0]].write_bytes(b"")
        with self.assertRaises(ValueError):
            model_inventory(profile, allow_hf_cache_blobs=True)

    def test_broken_optional_metadata_is_not_silently_ignored(self):
        snapshot, _, profile, _, _ = self.cache()
        self.links.link(snapshot / "generation_config.json", Path("../../blobs/missing-generation-config"))
        with self.assertRaises(ValueError):
            model_inventory(profile, allow_hf_cache_blobs=True)

    def test_source_traversal_is_rejected_by_functional_api(self):
        snapshot, _, profile, _, _ = self.cache()
        path = snapshot / ".." / snapshot.name
        with self.assertRaisesRegex(ValueError, "components"):
            FinancialExtractor.from_local("qwen3", path)
        with self.assertRaises(ValueError):
            model_inventory(replace(profile, load=replace(profile.load, source=str(path))), allow_hf_cache_blobs=True)


if __name__ == "__main__":
    unittest.main()
