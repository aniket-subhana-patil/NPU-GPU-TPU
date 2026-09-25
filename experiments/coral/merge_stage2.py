#!/usr/bin/env python3
"""
merge_stage2.py -- combine correct VERDICTS (from reparse_logs.py output) with
correct MEMORY columns (from the original stage-2 run) into one final CSV.

Why: the .log files have the operator table (verdict) but not the memory lines;
the original stage2_results.csv had memory parsed correctly but verdicts wrong.
This takes the right half of each.

Usage:
    python3 merge_stage2.py --verdicts stage2_verdicts.csv --memory stage2_old.csv --out stage2_final.csv
"""
import argparse, csv

def load(p):
    with open(p, newline='') as f:
        return {r['hash']: r for r in csv.DictReader(f)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--verdicts',required=True,help='CSV from reparse_logs.py (correct mapped/ops)')
    ap.add_argument('--memory',required=True,help='original stage2 CSV (correct memory cols)')
    ap.add_argument('--out',default='stage2_final.csv')
    a=ap.parse_args()

    verd=load(a.verdicts)
    mem=load(a.memory)

    FIELDS=['hash','params','total_ops','ops_on_tpu','ops_on_cpu',
            'onchip_used_bytes','offchip_streamed_bytes','mapped','nonmapped_ops','status']
    out={}
    for h,v in verd.items():
        row={k:v.get(k,'') for k in FIELDS}
        m=mem.get(h,{})
        # take memory from the old CSV where the reparse left it blank
        if not row.get('onchip_used_bytes') and m.get('onchip_used_bytes'):
            row['onchip_used_bytes']=m['onchip_used_bytes']
        if not row.get('offchip_streamed_bytes') and m.get('offchip_streamed_bytes'):
            row['offchip_streamed_bytes']=m['offchip_streamed_bytes']
        # params: prefer whichever is present
        if not row.get('params') and m.get('params'):
            row['params']=m['params']
        out[h]=row

    with open(a.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        for h in out: w.writerow(out[h])

    t=sum(1 for r in out.values() if r['mapped']=='TRUE')
    p=sum(1 for r in out.values() if r['mapped']=='PARTIAL')
    over=sum(1 for r in out.values() if str(r.get('offchip_streamed_bytes')) not in ('','0'))
    print(f'Merged {len(out)} models -> {a.out}')
    print(f'  TRUE(clean)={t}  PARTIAL={p}')
    print(f'  cache-overflow (offchip>0)={over}')
    # cross-tab: how many clean models also overflow cache?
    clean_over=sum(1 for r in out.values() if r['mapped']=='TRUE' and str(r.get('offchip_streamed_bytes')) not in ('','0'))
    print(f'  of TRUE models, {clean_over} overflow cache (these show the latency cliff)')

if __name__=='__main__': main()
