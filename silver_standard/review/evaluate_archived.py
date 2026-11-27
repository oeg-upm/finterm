#!/usr/bin/env python3
"""Recompute the archived reference-sensitivity and calibration results. No API/GPU."""
from pathlib import Path
from collections import Counter
import csv, json, math
import evaluation_core as e
import calibration_core as c
ROOT=Path(__file__).resolve().parent

def compare(name, actual):
    with (ROOT/'results'/name).open(encoding='utf-8-sig',newline='') as f:
        archived=list(csv.DictReader(f))
    assert len(archived)==len(actual),name
    for a,b in zip(archived,actual):
        assert set(a)==set(b),name
        for k,v in b.items():
            if isinstance(v,float): assert math.isclose(float(a[k]),v,rel_tol=1e-12,abs_tol=1e-12),(name,k,a[k],v)
            else: assert a[k]==str(v),(name,k,a[k],v)
    print(f'OK: {name}: {len(actual)} rows')

def agreement(x,y):
    n=len(x); o=sum(a==b for a,b in zip(x,y))/n
    cx,cy=Counter(x),Counter(y); p=sum(cx[k]*cy[k] for k in cx)/(n*n)
    return o,(o-p)/(1-p) if p!=1 else None

def main():
    d=e.load_data(); v2,_=e.load_v2(d)
    refs,changes,decisions,metrics=e.evaluate(d,v2)
    compare('metrics_58.csv',metrics); compare('decisions_58.csv',decisions)
    assert changes==json.loads((ROOT/'results/reference_changes.json').read_text())
    mm,dd,ss=c.compute(d,v2); compare('metrics_120.csv',mm);compare('decisions_120.csv',dd)
    assert ss==json.loads((ROOT/'results/calibration_summary.json').read_text())
    with (ROOT/'data/calibration_votes.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    archived=json.loads((ROOT/'results/panel_agreement.json').read_text())
    for r in rows:
        assert c.HUMAN[r['case_id']]==r['human']
        for m in e.JUDGES:
            assert r[m+'_V1']==d['v1'][m][r['pair_id']] and r[m+'_V2']==v2[m][r['pair_id']]
        for v in ('V1','V2'): assert r['panel_'+v]==e.majority([r[m+'_'+v] for m in e.JUDGES])
    for m in e.JUDGES+['Panel']:
        key=m if m!='Panel' else 'panel'
        for v in ('V1','V2'):
            col=key+'_'+v; rr=[r for r in rows if r[col]!='DISAGREE']; a=archived[m][v]
            assert len(rr)==a['n_consenso']; assert Counter(r[col] for r in rows)==a['etiquetas']
            for k,acc in [('estricto',e.ACCEPT['strict']),('amplio',e.ACCEPT['broad'])]:
                o,kap=agreement([r['human'] in acc for r in rr],[r[col] in acc for r in rr]); assert math.isclose(o,a['acuerdo_'+k]);assert math.isclose(kap,a['kappa_'+k])
    common=[r for r in rows if r['panel_V1']!='DISAGREE' and r['panel_V2']!='DISAGREE']
    assert len(common)==101
    for v in ('V1','V2'):
        for k,acc in [('estricto',e.ACCEPT['strict']),('amplio',e.ACCEPT['broad'])]:
            _,kap=agreement([r['human'] in acc for r in common],[r['panel_'+v] in acc for r in common]);assert math.isclose(kap,archived['Panel']['comparacion_mismos_casos_con_consenso'][v]['kappa_'+k])
    print('OK: calibration agreement and common-consensus kappa')
    print('All archived metrics verified; no files modified.')
if __name__=='__main__': main()
