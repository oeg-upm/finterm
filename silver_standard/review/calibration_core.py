#!/usr/bin/env python3
"""Evaluate archived extraction predictions restricted to the same 120 reviewed pairs.
Requires evaluation_core.py in the same directory and the existing V2 JSONL archives.
No API calls, no GPU, no modification of original data. Human labels are distributed without names or notes; publication requires approval.
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv, json
import evaluation_core as original

HUMAN = {'A-001': 'SPECIFIC', 'A-002': 'SPECIFIC', 'A-003': 'SPECIFIC', 'A-004': 'PARTIAL', 'A-005': 'NOT_TERM', 'A-006': 'NOT_TERM', 'A-007': 'NOT_TERM', 'A-008': 'NOT_TERM', 'A-009': 'SPECIFIC', 'A-010': 'SPECIFIC', 'A-011': 'SPECIFIC', 'A-012': 'SPECIFIC', 'A-013': 'GENERIC', 'A-014': 'NOT_TERM', 'A-015': 'GENERIC', 'A-016': 'PARTIAL', 'A-017': 'SPECIFIC', 'A-018': 'PARTIAL', 'A-019': 'NOT_TERM', 'A-020': 'NOT_TERM', 'A-021': 'SPECIFIC', 'A-022': 'GENERIC', 'A-023': 'SPECIFIC', 'A-024': 'GENERIC', 'A-025': 'PARTIAL', 'A-026': 'NOT_TERM', 'A-027': 'NOT_TERM', 'A-028': 'SPECIFIC', 'A-029': 'GENERIC', 'A-030': 'NOT_TERM', 'A-031': 'GENERIC', 'A-032': 'GENERIC', 'A-033': 'NOT_TERM', 'A-034': 'SPECIFIC', 'A-035': 'GENERIC', 'A-036': 'NOT_TERM', 'A-037': 'SPECIFIC', 'A-038': 'NOT_TERM', 'A-039': 'SPECIFIC', 'A-040': 'NOT_TERM', 'A-041': 'NOT_TERM', 'A-042': 'PARTIAL', 'A-043': 'GENERIC', 'A-044': 'SPECIFIC', 'A-045': 'GENERIC', 'A-046': 'NOT_TERM', 'A-047': 'SPECIFIC', 'A-048': 'PARTIAL', 'A-049': 'SPECIFIC', 'A-050': 'PARTIAL', 'A-051': 'NOT_TERM', 'A-052': 'SPECIFIC', 'A-053': 'PARTIAL', 'A-054': 'NOT_TERM', 'A-055': 'GENERIC', 'A-056': 'PARTIAL', 'A-057': 'PARTIAL', 'A-058': 'PARTIAL', 'A-059': 'PARTIAL', 'A-060': 'PARTIAL', 'A-061': 'GENERIC', 'A-062': 'GENERIC', 'A-063': 'NOT_TERM', 'A-064': 'SPECIFIC', 'A-065': 'NOT_TERM', 'A-066': 'NOT_TERM', 'A-067': 'PARTIAL', 'A-068': 'SPECIFIC', 'A-069': 'PARTIAL', 'A-070': 'NOT_TERM', 'A-071': 'PARTIAL', 'A-072': 'GENERAL_USE', 'A-073': 'NOT_TERM', 'A-074': 'NOT_TERM', 'A-075': 'NOT_TERM', 'A-076': 'GENERAL_USE', 'A-077': 'NOT_TERM', 'A-078': 'SPECIFIC', 'A-079': 'NOT_TERM', 'A-080': 'PARTIAL', 'A-081': 'NOT_TERM', 'A-082': 'GENERAL_USE', 'A-083': 'NOT_TERM', 'A-084': 'NOT_TERM', 'A-085': 'NOT_TERM', 'A-086': 'NOT_TERM', 'A-087': 'SPECIFIC', 'A-088': 'SPECIFIC', 'A-089': 'SPECIFIC', 'A-090': 'PARTIAL', 'A-091': 'PARTIAL', 'A-092': 'GENERIC', 'A-093': 'PARTIAL', 'A-094': 'SPECIFIC', 'A-095': 'NOT_TERM', 'A-096': 'NOT_TERM', 'A-097': 'GENERIC', 'A-098': 'NOT_TERM', 'A-099': 'SPECIFIC', 'A-100': 'GENERIC', 'A-101': 'PARTIAL', 'A-102': 'NOT_TERM', 'A-103': 'SPECIFIC', 'A-104': 'NOT_TERM', 'A-105': 'SPECIFIC', 'A-106': 'NOT_TERM', 'A-107': 'GENERIC', 'A-108': 'PARTIAL', 'A-109': 'NOT_TERM', 'A-110': 'PARTIAL', 'A-111': 'NOT_TERM', 'A-112': 'NOT_TERM', 'A-113': 'NOT_TERM', 'A-114': 'NOT_TERM', 'A-115': 'NOT_TERM', 'A-116': 'GENERIC', 'A-117': 'PARTIAL', 'A-118': 'SPECIFIC', 'A-119': 'SPECIFIC', 'A-120': 'SPECIFIC'}

def compute(d, v2):
    ids = {r['case_id']: r['pair_id'] for r in d['inputs']}
    assert set(ids) == set(HUMAN) and len(set(ids.values())) == 120
    assert d['source_sha256'] and len(d['pairs']) == 990
    assert all(set(v2[m]) == set(ids.values()) for m in original.JUDGES)
    old = {cid: original.majority([d['v1'][m][pid] for m in original.JUDGES]) for cid,pid in ids.items()}
    new = {cid: original.majority([v2[m][pid] for m in original.JUDGES]) for cid,pid in ids.items()}
    existing = {cid: d['pairs'][pid]['term'] in d['refs'][d['pairs'][pid]['segment_id']] for cid,pid in ids.items()}
    assert sum(existing.values()) == 100
    assert all(existing[cid] == (d['pairs'][pid]['source']=='reference') for cid,pid in ids.items())
    predictions = {m: {cid: d['pairs'][pid]['term'] in d['pred'][m].get(d['pairs'][pid]['segment_id'],[]) for cid,pid in ids.items()} for m in original.MODELS}
    metrics=[]; audit=[]; summaries=[]
    for criterion, accepted in original.ACCEPT.items():
        # Same policy as the global comparison: when judges disagree,
        # retain existing annotations and do not add absent candidates.
        binary = {'original':existing, 'V1':{c: existing[c] if old[c]=='DISAGREE' else old[c] in accepted for c in ids}, 'V2':{c: existing[c] if new[c]=='DISAGREE' else new[c] in accepted for c in ids}, 'human':{c:HUMAN[c] in accepted for c in ids}}
        changed = sorted(c for c in ids if binary['V1'][c]!=binary['V2'][c])
        summaries.append({'criterion':criterion,'positives':{k:sum(v.values()) for k,v in binary.items()},'changed_V1_to_V2':len(changed),'human_matches_on_changed':{k:sum(binary[k][c]==binary['human'][c] for c in changed) for k in ('V1','V2')}})
        for c in sorted(ids):
            audit.append({'criterion':criterion,'case_id':c,'pair_id':ids[c],'human_label':HUMAN[c],'V1_label':old[c],'V2_label':new[c],**{k+'_accepted':int(v[c]) for k,v in binary.items()},'changed_V1_to_V2':int(c in changed)})
        for scope, cases in [('all_120',sorted(ids)),('changed_only_diagnostic',changed)]:
            if not cases: continue
            for model, pred in predictions.items():
                for reference, ref in binary.items():
                    tp=sum(pred[c] and ref[c] for c in cases); fp=sum(pred[c] and not ref[c] for c in cases); fn=sum(not pred[c] and ref[c] for c in cases); tn=len(cases)-tp-fp-fn
                    metrics.append(dict(criterion=criterion,scope=scope,n=len(cases),model=model,reference=reference,positives=tp+fn,TP=tp,FP=fp,FN=fn,TN=tn,P=tp/(tp+fp) if tp+fp else 0,R=tp/(tp+fn) if tp+fn else 0,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))
    return metrics,audit,summaries


with (Path(__file__).resolve().parent / "data/human_calibration_120.csv").open(encoding="utf-8-sig",newline="") as f:
    HUMAN={r["case_id"]:r["label"] for r in csv.DictReader(f)}
