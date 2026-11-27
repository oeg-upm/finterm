#!/usr/bin/env python3
"""Recalculate the original extraction evaluation. Python standard library only.
Run from any directory: python extraction/evaluate_archived.py
All eleven comparative configurations are included in archived_predictions.zip.
No API calls, model execution or modification of references.
"""
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import csv, json, hashlib, sys
ROOT=Path(__file__).resolve().parent.parent
norm=lambda s:' '.join(s.casefold().split())
def lines(b):return [norm(t) for t in b.decode('utf-8-sig').splitlines() if t.strip()]
with ZipFile(ROOT/'data/evaluation_sample.zip') as z:
    m=json.loads(z.read('manifest.json'))
    assert len(m)==len({r['id'] for r in m})==60
    for r in m:assert hashlib.sha256(z.read(r['file'])).hexdigest()==r['sha256']
    ids={r['id'] for r in m if r['evaluable']}
    assert len(ids)==58
    refs={s:set(lines(z.read('references/'+s+'.key'))) for s in ids}
assert sum(map(len,refs.values()))==3433
with ZipFile(ROOT/'silver_standard/enriched.zip') as z:
    full={Path(n).stem:n for n in z.namelist() if n.endswith('.key')}
    assert len(full)==2238 and all(refs[s]==set(lines(z.read(full[s]))) for s in ids)
with ZipFile(ROOT/'silver_standard/core.zip') as z:
    full={Path(n).stem:n for n in z.namelist() if n.endswith('.key')}
    core={s:set(lines(z.read(full[s]))) for s in ids}
    assert len(full)==2555 and sum(map(len,core.values()))==2028
pred=defaultdict(dict)
archives=[ROOT/'extraction/archived_predictions.zip']
for p in archives:
    with ZipFile(p) as z:
        for n in z.namelist():
            if not n.endswith('.key'):continue
            model,s=n.split('/'); s=Path(s).stem
            assert s in ids and s not in pred[model]
            data=lines(z.read(n))
            if model.startswith('AttentionRank_') or model in ('KeyBERT','PromptRank'):
                data=data[:15]
                assert len(data)==len(set(data))==15,(model,s)
            pred[model][s]=set(data)
assert set(pred)=={'AttentionRank_BERT','AttentionRank_FinBERT','AttentionRank_RoBERTa','KeyBERT','PromptRank','qwen_zero','qwen_few','gemma_zero','gemma_few','dister_zero','dister_few'}
w=csv.DictWriter(sys.stdout,fieldnames=['model','reference','segments','available_predictions','reference_pairs','TP','FP','FN','P','R','F1'])
w.writeheader()
for model,pp in sorted(pred.items()):
    assert len(pp)=={'dister_zero':57,'dister_few':56}.get(model,58),(model,len(pp))
    for refname,rr in [('core',core),('enriched_original',refs)]:
        tp=sum(len(pp.get(s,set())&rr[s]) for s in ids)
        fp=sum(len(pp.get(s,set())-rr[s]) for s in ids)
        fn=sum(len(rr[s]-pp.get(s,set())) for s in ids)
        w.writerow(dict(model=model,reference=refname,segments=len(ids),available_predictions=len(pp),reference_pairs=tp+fn,TP=tp,FP=fp,FN=fn,P=tp/(tp+fp) if tp+fp else 0,R=tp/(tp+fn) if tp+fn else 0,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))
