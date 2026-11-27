"""Adapted old protocol helpers; synthetic data and embeddings, never real models.

Origins and adaptations are recorded in RECOVERED_TESTS.md. No old application
module is imported. Only the hash-verified H100 runtime supplies algorithms.
"""
import json
import math
import re


class ControlledEncoder:
    """Recovered from tests/test_similarity.py::ControlledEncoder."""
    def __init__(self, mapping=None, *, action=None, batch_size=2):
        self.identity = {"backend_kind": "CONTROLLED_EMBEDDINGS_NO_MODEL",
                         "model_id": "CONTROLLED_NOT_A_REAL_MODEL", "model_revision": "CONTROLLED",
                         "embedding_dimension": 2, "normalize_embeddings": True,
                         "similarity": "cosine", "prompt_policy": "SYMMETRIC_EMPTY_TEST"}
        self.mapping, self.action = mapping or {}, action
        self.batch_size = batch_size
        self.prepare_calls, self.calls = 0, []
        self.runtime = {"backend_kind": self.identity["backend_kind"], "test_only": True}

    def prepare(self):
        self.prepare_calls += 1

    def encode(self, texts):
        self.calls.extend(texts)
        return self.action(texts) if self.action else [self.mapping.get(text, [1.0, 0.0]) for text in texts]

    def similarity(self, context, representations):
        return [sum(a * b for a, b in zip(context, rep)) /
                math.sqrt(sum(a * a for a in context) * sum(b * b for b in rep)) for rep in representations]


Q_ROVER, Q_SAMPLES, Q_EXACT, Q_CONTEXT, Q_DISAMBIG = (
    "Q990000001", "Q990000002", "Q990000003", "Q990000004", "Q990000099")
Q_TYPE, Q_PARENT, Q_SOURCE = "Q990000010", "Q990000011", "Q990000012"


class FixtureRecorder:
    """Recovered stage02_fixture_support.py; runtime is injected explicitly."""
    def __init__(self, api, terms, contexts, lemmas):
        self.api, self.responses, self.requests, self.owners = api, {}, [], {}
        queries = api.generate_context_literal_queries(terms, contexts, lemma_resolver=lemmas.__getitem__)
        for term, query, _ in api.search_specs(terms, queries, lemma_resolver=lemmas.__getitem__):
            self.owners.setdefault(query.casefold(), set()).add(term)

    def ids_for_query(self, query, sparql=False):
        owners = self.owners.get(query.casefold(), set())
        if "basalt" in owners:
            return []
        found = []
        if "lunar rover" in owners:
            found.extend([Q_ROVER, Q_DISAMBIG])
            if query.casefold() != "lunar rover":
                found.append(Q_CONTEXT)
            if sparql:
                found.append(Q_EXACT)
        if "samples" in owners:
            found.append(Q_SAMPLES)
        return found

    def entity(self, qid):
        labels = {Q_ROVER: "SYNTHETIC TEST ONLY rover", Q_SAMPLES: "SYNTHETIC TEST ONLY samples",
                  Q_EXACT: "SYNTHETIC TEST ONLY exact-label candidate",
                  Q_CONTEXT: "SYNTHETIC TEST ONLY context candidate",
                  Q_DISAMBIG: "SYNTHETIC TEST ONLY disambiguation", Q_TYPE: "SYNTHETIC TEST ONLY type",
                  Q_PARENT: "SYNTHETIC TEST ONLY parent", Q_SOURCE: "SYNTHETIC TEST ONLY source",
                  "Q4167410": "Wikimedia disambiguation page"}
        p31 = "Q4167410" if qid == Q_DISAMBIG else Q_TYPE
        claims = {pid: [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": value}}}}]
                  for pid, value in (("P31", p31), ("P279", Q_PARENT), ("P1343", Q_SOURCE))}
        return {"id": qid, "labels": {"en": {"value": labels.get(qid, "SYNTHETIC TEST ONLY unknown")}},
                "aliases": {"en": [{"value": "synthetic alias"}, {"value": "synthetic alias"}]},
                "descriptions": {"en": {"value": "Wikimedia disambiguation page" if qid == Q_DISAMBIG
                                        else "SYNTHETIC FIXTURE ONLY semantic description"},
                                 "es": {"value": "Descripción sintética de prueba"}}, "claims": claims}

    def get_json(self, endpoint, params):
        api = self.api
        if params.get("action") == "wbsearchentities":
            payload = {"searchinfo": {"search": params["search"]}, "search": [
                {"id": qid, "description": "Wikimedia disambiguation page" if qid == Q_DISAMBIG else "SYNTHETIC FIXTURE ONLY",
                 "match": {"type": "label", "language": "en", "text": params["search"]}}
                for qid in self.ids_for_query(params["search"])]}
        elif params.get("action") == "wbgetentities":
            payload = {"entities": {qid: self.entity(qid) for qid in params["ids"].split("|")}}
        elif endpoint == api.WIKIDATA_SPARQL_URL and "?spec_index" in params["query"]:
            bindings = []
            for index, literal in re.findall(r'\("(\d+)" "((?:\\.|[^"\\])*)"@en\)', params["query"]):
                for qid in self.ids_for_query(json.loads('"' + literal + '"'), sparql=True):
                    bindings.append({"spec_index": {"value": index},
                                     "item": {"value": "http://www.wikidata.org/entity/" + qid}})
            payload = {"results": {"bindings": bindings}}
        elif endpoint == api.WIKIDATA_SPARQL_URL:
            payload = {"results": {"bindings": [
                {"item": {"value": "http://www.wikidata.org/entity/" + qid},
                 "property": {"value": "http://www.wikidata.org/prop/direct/" + pid},
                 "value": {"value": "http://www.wikidata.org/entity/" + value}}
                for qid in re.findall(r"wd:(Q[1-9][0-9]*)", params["query"])
                for pid, value in (("P31", "Q4167410" if qid == Q_DISAMBIG else Q_TYPE),
                                   ("P279", Q_PARENT), ("P1343", Q_SOURCE))]}}
        else:
            raise AssertionError("Unexpected synthetic request")
        self.responses[api.JsonCache.request_key(endpoint, params)] = payload
        self.requests.append({"endpoint": endpoint, "params": dict(params)})
        return payload


def controlled_cache(runtime, terms, contexts, lemmas):
    recorder = FixtureRecorder(runtime.api, terms, contexts, lemmas)
    runtime.stage02.collect(runtime.api, terms, contexts, lemmas.__getitem__, recorder)
    return {"version": 1, "data_provenance": "CONTROLLED_FIXTURE_NOT_LIVE_WIKIDATA",
            "responses": recorder.responses}
