#!/usr/bin/env python3
"""
reparse_logs.py -- read existing edgetpu_compiler .log files and produce the
correct stage2_results.csv. NO recompiling, NO compiler needed. Runs anywhere
(WSL2 or native Windows with python).

The v16 compiler already wrote a <hash>_edgetpu.log per model containing the
Operator/Count/Status table. This just reads them.

Usage:
    python3 reparse_logs.py --edgetpu-dir "/mnt/c/.../nasbench_study/edgetpu"

    # if your stage-1 results.csv (with params) is elsewhere:
    python3 reparse_logs.py --edgetpu-dir ... --stage1-csv "/mnt/c/.../nasbench_study/results.csv"
"""
import argparse, csv, glob, os, re, sys

RE_TOTAL   = re.compile(r'Total number of operations:\s*(\d+)')
RE_ONCHIP  = re.compile(r'On-chip memory used for caching model parameters:\s*([\d.]+)([KMG]?i?B)')
RE_OFFCHIP = re.compile(r'Off-chip memory used for streaming uncached model parameters:\s*([\d.]+)([KMG]?i?B)')
RE_OP_ROW  = re.compile(r'^([A-Z_0-9]+)\s+(\d+)\s+(.+?)\s*$', re.MULTILINE)

_UNIT={'B':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3,'KB':1000,'MB':1000**2,'GB':1000**3}
def to_bytes(n,u): return int(float(n)*_UNIT.get(u,1))

FIELDS=['hash','params','total_ops','ops_on_tpu','ops_on_cpu',
        'onchip_used_bytes','offchip_streamed_bytes','mapped','nonmapped_ops','status']

def parse_log(log):
    total=RE_TOTAL.search(log); total=int(total.group(1)) if total else None
    tpu=cpu=0; nonmapped=[]
    for name,count,status in RE_OP_ROW.findall(log):
        c=int(count)
        if 'Mapped to Edge TPU' in status: tpu+=c
        else: cpu+=c; nonmapped.append(f'{name}x{c}')
    on=RE_ONCHIP.search(log); off=RE_OFFCHIP.search(log)
    return dict(total_ops=total, ops_on_tpu=tpu, ops_on_cpu=cpu,
                onchip_used_bytes=to_bytes(*on.groups()) if on else '',
                offchip_streamed_bytes=to_bytes(*off.groups()) if off else '',
                nonmapped_ops=';'.join(nonmapped),
                parsed_ok=(total is not None and (tpu+cpu)==total and total>0))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--edgetpu-dir',required=True)
    ap.add_argument('--stage1-csv',default=None)
    ap.add_argument('--out',default=None)
    a=ap.parse_args()
    if not os.path.isdir(a.edgetpu_dir): sys.exit(f'not found: {a.edgetpu_dir}')
    study=os.path.dirname(a.edgetpu_dir.rstrip('/\\'))
    outp=a.out or os.path.join(study,'stage2_results.csv')

    params={}
    s1=a.stage1_csv or os.path.join(study,'results.csv')
    if os.path.exists(s1):
        with open(s1,newline='') as f:
            for r in csv.DictReader(f): params[r['hash']]=r.get('params','')
        print(f'params merged from {s1}')
    else:
        print(f'(no stage-1 csv at {s1}; params left blank)')

    logs=sorted(glob.glob(os.path.join(a.edgetpu_dir,'*_edgetpu.log')))
    print(f'{len(logs)} .log files found.')
    if not logs: sys.exit('No *_edgetpu.log files -- wrong dir?')

    rows={}
    for lg in logs:
        h=os.path.basename(lg).replace('_edgetpu.log','')
        try:
            with open(lg,errors='replace') as f: text=f.read()
        except Exception as e:
            rows[h]={**{k:'' for k in FIELDS},'hash':h,'mapped':'FALSE','status':f'read_error:{type(e).__name__}'}
            continue
        info=parse_log(text)
        row={k:'' for k in FIELDS}; row['hash']=h; row['params']=params.get(h,'')
        for k in ('total_ops','ops_on_tpu','ops_on_cpu','onchip_used_bytes','offchip_streamed_bytes','nonmapped_ops'):
            row[k]=info[k]
        if not info['parsed_ok']:
            row['mapped']='PARSE_FAILED'; row['status']='parse_failed_see_log'
        elif info['ops_on_cpu']==0:
            row['mapped']='TRUE'; row['status']='mapped'
        else:
            row['mapped']='PARTIAL'; row['status']=f"partial_{info['ops_on_cpu']}_cpu_ops"
        rows[h]=row

    with open(outp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        for h in rows: w.writerow(rows[h])

    t=sum(1 for r in rows.values() if r['mapped']=='TRUE')
    p=sum(1 for r in rows.values() if r['mapped']=='PARTIAL')
    pf=sum(1 for r in rows.values() if r['mapped']=='PARSE_FAILED')
    over=sum(1 for r in rows.values() if str(r['offchip_streamed_bytes']) not in ('','0'))
    print(f'\n=== RESULTS ===')
    print(f'  TRUE  (clean, fully on TPU): {t}')
    print(f'  PARTIAL (some CPU fallback): {p}')
    if pf: print(f'  PARSE_FAILED (check logs)  : {pf}')
    print(f'  models overflowing on-chip cache (offchip>0): {over}')
    print(f'\nWritten: {outp}')
    if p:
        print('\nDistinct CPU-fallback op sets among PARTIAL models:')
        seen={}
        for r in rows.values():
            if r['mapped']=='PARTIAL' and r['nonmapped_ops']:
                seen[r['nonmapped_ops']]=seen.get(r['nonmapped_ops'],0)+1
        for ops,cnt in sorted(seen.items(),key=lambda x:-x[1])[:10]:
            print(f'   {cnt:4d}x  {ops}')

if __name__=='__main__': main()
