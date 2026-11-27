#!/usr/bin/env python3
"""FINTERM: evaluate archived extractions after a partial judge-based reference review.
Self-contained snapshot of the verified original reference/prediction archives and V1 votes.
Reads the 360 archived V2 calibration responses. Standard library only. No API/GPU.
Never modifies the original silver, saved votes, or repository. Output is a separate ZIP.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import base64, zlib, hashlib, json, csv, io

HOME = Path(__file__).resolve().parent
LABELS = {'SPECIFIC','GENERIC','GENERAL_USE','PARTIAL','NOT_TERM'}
MODELS = ['AttentionRank_BERT','AttentionRank_FinBERT','AttentionRank_RoBERTa','KeyBERT','PromptRank','qwen_zero','qwen_few','gemma_zero','gemma_few','dister_zero','dister_few']
JUDGES = ['Cohere','Mistral','Nemotron']
ACCEPT = {'strict': {'SPECIFIC'}, 'broad': {'SPECIFIC','GENERIC'}}

def sha(b): return hashlib.sha256(b).hexdigest()
def load_data():
    d=json.loads((HOME/'data/evaluation_snapshot.json').read_text(encoding='utf-8'))
    for m in JUDGES:
        records=[json.loads(line) for line in (HOME/'votes/v1'/f'{m}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records)==990 and {r['pair_id']:r['label'] for r in records}==d['v1'][m]
    return d

def majority(labels):
    assert len(labels)==3 and set(labels)<=LABELS
    lab,n=Counter(labels).most_common(1)[0]
    return lab if n>=2 else 'DISAGREE'

def load_v2(d):
    prompt=d['prompt']; prefix=sha(prompt.encode())[:12]
    root=HOME/'votes/v2_A'
    assert (HOME/'prompts/V2.txt').read_text(encoding='utf-8')==prompt
    votes={m:{} for m in JUDGES}; raw_files={}
    records={}
    expected_ids={r['case_id'] for r in d['inputs']}
    for m in JUDGES:
        lines=(root/f'{m}.jsonl').read_bytes().splitlines()
        assert len(lines)==120
        records[m]={json.loads(line)['case_id']:line for line in lines}
        assert set(records[m])==expected_ids
    for r in d['inputs']:
        cid=r['case_id']; pid=r['pair_id']; h=sha(prompt.format(term=r['term'],context=r['context']).encode())
        for m in JUDGES:
            p=f'{root}/{m}.jsonl:{cid}'
            raw=records[m][cid]; v=json.loads(raw)
            assert all(v[k]==r[k] for k in ('case_id','pair_id','term','context')), f'Caso/contexto diferente: {p}'
            assert v['model']==d['models'][m] and v['prompt_sha256']==h, f'Modelo/prompt diferente: {p}'
            cfg=(d['n_cfg'] if m=='Nemotron' else {'temperature':0,'max_tokens':d['m_limits'][cid],'reasoning_effort':'low','stream':True} if m=='Mistral' else {'temperature':0,'max_tokens':250,'stream':False})
            assert v['config']==cfg, f'Configuracion diferente: {p}'
            assert v['finish_reason']==('COMPLETE' if m=='Cohere' else 'stop'), f'Respuesta incompleta: {p}'
            obj=json.loads(v['raw_response'])
            assert obj['label']==v['label'] and v['label'] in LABELS
            assert isinstance(obj['justification'],str) and obj['justification'].strip() and obj['justification']==v['justification']
            assert pid not in votes[m]
            votes[m][pid]=v['label']; raw_files[f'votos_V2/{m}/{cid}.json']=raw
    expected={
        'Cohere':{'GENERIC':39,'SPECIFIC':60,'NOT_TERM':20,'PARTIAL':1},
        'Mistral':{'GENERIC':17,'SPECIFIC':56,'NOT_TERM':37,'PARTIAL':8,'GENERAL_USE':2},
        'Nemotron':{'SPECIFIC':55,'PARTIAL':7,'GENERIC':35,'NOT_TERM':15,'GENERAL_USE':8}}
    for m in JUDGES:
        assert len(votes[m])==120
        assert Counter(votes[m].values())==expected[m], f'Los recuentos de {m} difieren de la ultima salida de Jupyter.'
        print(f'OK: {m} V2, 120/120 respuestas verificadas')
    p2={k:majority([votes[m][k] for m in JUDGES]) for k in votes['Cohere']}
    assert Counter(p2.values())=={'GENERIC':30,'SPECIFIC':58,'DISAGREE':10,'NOT_TERM':18,'PARTIAL':4}
    print('OK: panel V2, 110 decisiones mayoritarias y 10 desacuerdos')
    return votes,raw_files

def evaluate(d,v2):
    orig={s:set(v) for s,v in d['refs'].items()}; ids=sorted(orig)
    pred={m:{s:set(t) for s,t in v.items()} for m,v in d['pred'].items()}
    assert len(ids)==58 and sum(map(len,orig.values()))==3433
    assert set(pred)==set(MODELS)
    assert all(len(pred[m])=={'dister_zero':57,'dister_few':56}.get(m,58) for m in MODELS)
    pairs=d['pairs']; assert len(pairs)==990
    a={r['pair_id'] for r in d['inputs']}; assert len(a)==120
    assert all(set(v2[m])==a for m in JUDGES)
    assert Counter(pairs[k]['source'] for k in a)=={'reference':100,'pool':20}
    p1={k:majority([d['v1'][m][k] for m in JUDGES]) for k in pairs}
    p2={k:majority([v2[m][k] for m in JUDGES]) for k in a}
    assert Counter(p1.values())=={'SPECIFIC':624,'GENERIC':155,'GENERAL_USE':57,'PARTIAL':41,'NOT_TERM':33,'DISAGREE':80}
    groups={'A_V1':{k:p1[k] for k in a}, 'A_V2':p2, 'V1_990':p1, 'MIXTA_990':{**p1,**p2}}
    refs={'original':orig}; changes={}; decisions=[]
    for group, pp in groups.items():
        for policy, accept in ACCEPT.items():
            name=f'{group}_{policy}'; rr={s:set(v) for s,v in orig.items()}; counts=Counter()
            for pid,lab in sorted(pp.items()):
                row=pairs[pid]; s,t=row['segment_id'],row['term']; existing=t in orig[s]
                assert existing==(row['source']=='reference')
                if lab=='DISAGREE': action='KEEP_DISAGREE' if existing else 'KEEP_OUT_DISAGREE'
                elif lab in accept:
                    action='KEEP' if existing else 'ADD'
                    rr[s].add(t)
                else:
                    action='REMOVE' if existing else 'DO_NOT_ADD'
                    rr[s].discard(t)
                counts[action]+=1
                decisions.append(dict(scenario=name,pair_id=pid,source=row['source'],label=lab,action=action))
            refs[name]=rr
            n=sum(map(len,rr.values()))
            assert n==3433-counts['REMOVE']+counts['ADD']
            changes[name]={'reviewed_pairs':len(pp),'adds':counts['ADD'],'removals':counts['REMOVE'],'reference_pairs':n,'actions':dict(counts)}
    assert changes['V1_990_strict']['reference_pairs']==3290
    assert changes['V1_990_broad']['reference_pairs']==3445
    metrics=[]
    for m in MODELS:
        pp=pred[m]
        for name,rr in refs.items():
            tp=sum(len(pp.get(s,set())&rr[s]) for s in ids)
            fp=sum(len(pp.get(s,set())-rr[s]) for s in ids)
            fn=sum(len(rr[s]-pp.get(s,set())) for s in ids)
            metrics.append(dict(model=m,reference=name,segments=58,available_predictions=len(pp),reference_pairs=tp+fn,TP=tp,FP=fp,FN=fn,P=tp/(tp+fp) if tp+fp else 0,R=tp/(tp+fn) if tp+fn else 0,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0))
    lookup={(r['model'],r['reference']):r for r in metrics}
    expected_tp=[333,314,324,34,70,943,1263,584,808,820,1001]
    assert [lookup[m,'original']['TP'] for m in MODELS]==expected_tp
    return refs,changes,decisions,metrics

def csv_bytes(rs):
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs); return f.getvalue().encode('utf-8-sig')

def show(metrics,changes):
    lookup={(r['model'],r['reference']):r for r in metrics}
    print('\n=== COMPARACION PRINCIPAL: SOLO SE CAMBIAN LOS MISMOS 120 PARES ===')
    print('Todos los resultados: 58 segmentos. F1 expresado en porcentaje.')
    cols=['original','A_V1_strict','A_V2_strict','A_V1_broad','A_V2_broad']
    print(f'{"Metodo":24s}'+''.join(f'{c:>14}' for c in cols))
    for m in MODELS: print(f'{m:24s}'+''.join(f'{100*lookup[m,c]["F1"]:14.2f}' for c in cols))
    for c in cols[1:]:
        s=changes[c]; print(c,':',s['removals'],'eliminados,',s['adds'],'anadidos,',s['reference_pairs'],'referencias')
    print('\n=== COMPARACION ADICIONAL: 990 V1 frente a 870 V1 + 120 V2 ===')
    print('MIXTA_990 NO significa que los 990 casos se hayan evaluado con V2.')
    cols=['V1_990_strict','MIXTA_990_strict','V1_990_broad','MIXTA_990_broad']
    print(f'{"Metodo":24s}'+''.join(f'{c:>19}' for c in cols))
    for m in MODELS: print(f'{m:24s}'+''.join(f'{100*lookup[m,c]["F1"]:19.2f}' for c in cols))
    for c in cols:
        s=changes[c]; print(c,':',s['removals'],'eliminados,',s['adds'],'anadidos,',s['reference_pairs'],'referencias')

