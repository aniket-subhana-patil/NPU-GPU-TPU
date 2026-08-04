#!/usr/bin/env python3
"""
reparse_logs_v2.py -- read existing edgetpu_compiler .log files -> stage2 CSV.
NO recompiling. Runs anywhere (WSL2 or native Windows).

v2 fix: the .log FILE format omits the "Total number of operations" line that
appears on stdout, and has no memory lines. So we:
  - do NOT require a total-ops line (verdict comes purely from the operator table)
  - consider a parse successful if we found >=1 operator row
  - leave memory columns blank (they get merged from the original stage-2 CSV)

Robust to tabs or multiple spaces between columns, and to CRLF line endings.

Usage:
    python3 reparse_logs_v2.py --edgetpu-dir "/mnt/c/.../nasbench_study/edgetpu" --out stage2_verdicts.csv
"""
import argparse, csv, glob, os, re, sys

# operator row: NAME  COUNT  STATUS   (any run of whitespace incl. tabs between)
RE_OP_ROW = re.compile(r'^([A-Z][A-Z_0-9]*)[ \t]+(\d+)[ \t]+(.+?)[ \t]*$', re.MULTILINE)
RE_TOTAL  = re.compile(r'Total number of operations:\s*(\d+)')  # optional

FIELDS=['hash','params','total_ops','ops_on_tpu','ops_on_cpu',
        'onchip_used_bytes','offchip_streamed_bytes','mapped','nonmapped_ops','status']

def parse_log(log):
    rows = RE_OP_ROW.findall(log)
    tpu=cpu=0; nonmapped=[]
    n_rows=0
    for name,count,status in rows:
        # guard: skip a header row if 'Count' somehow matched (it won't: \d+ required)
        c=int(count); n_rows+=1
        if 'Mapped to Edge TPU' in status: tpu+=c
        else: cpu+=c; nonmapped.append(f'{name}x{c}')
    total_m=RE_TOTAL.search(log)
    total=int(total_m.group(1)) if total_m else (tpu+cpu)
    return dict(total_ops=total, ops_on_tpu=tpu, ops_on_cpu=cpu,
                nonmapped_ops=';'.join(nonmapped),
                parsed_ok=(n_rows>=1))   # <-- only need operator rows, not a total line

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--edgetpu-dir',required=True)
    ap.add_argument('--stage1-csv',default=None)
    ap.add_argument('--out',default=None)
    a=ap.parse_args()
    if not os.path.isdir(a.edgetpu_dir): sys.exit(f'not found: {a.edgetpu_dir}')
    study=os.path.dirname(a.edgetpu_dir.rstrip('/\\'))
    outp=a.out or os.path.join(study,'stage2_verdicts.csv')

    params={}
    s1=a.stage1_csv or os.path.join(study,'results.csv')
    if os.path.exists(s1):
        with open(s1,newline='') as f:
            for r in csv.DictReader(f): params[r['hash']]=r.get('params','')

    logs=sorted(glob.glob(os.path.join(a.edgetpu_dir,'*_edgetpu.log')))
    print(f'{len(logs)} .log files found.')
    if not logs: sys.exit('no *_edgetpu.log files')

    rows={}; examples_shown=0
    for lg in logs:
        h=os.path.basename(lg).replace('_edgetpu.log','')
        with open(lg,errors='replace') as f: text=f.read()
        info=parse_log(text)
        row={k:'' for k in FIELDS}; row['hash']=h; row['params']=params.get(h,'')
        row['total_ops']=info['total_ops']; row['ops_on_tpu']=info['ops_on_tpu']
        row['ops_on_cpu']=info['ops_on_cpu']; row['nonmapped_ops']=info['nonmapped_ops']
        if not info['parsed_ok']:
            row['mapped']='PARSE_FAILED'; row['status']='parse_failed_see_log'
            if examples_shown<1:  # dump one failing file so we can see why
                print('\n--- FIRST PARSE_FAILED FILE (raw, first 500 chars) ---')
                print(repr(text[:500])); print('--- end ---\n'); examples_shown+=1
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
    print(f'=== RESULTS ===')
    print(f'  TRUE  (clean): {t}')
    print(f'  PARTIAL      : {p}')
    print(f'  PARSE_FAILED : {pf}')
    print(f'Written: {outp}')
    if p:
        seen={}
        for r in rows.values():
            if r['mapped']=='PARTIAL' and r['nonmapped_ops']:
                seen[r['nonmapped_ops']]=seen.get(r['nonmapped_ops'],0)+1
        print('\nCPU-fallback op sets among PARTIAL:')
        for ops,cnt in sorted(seen.items(),key=lambda x:-x[1])[:10]:
            print(f'   {cnt:4d}x {ops}')

if __name__=='__main__': main()
