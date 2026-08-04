#!/usr/bin/env python3
"""
stage2_v16.py  --  run INSIDE WSL2 / Linux (needs edgetpu_compiler)

Corrected stage 2 for edgetpu_compiler v16.x, which reports an OPERATOR TABLE
(Operator / Count / Status) rather than "run on Edge TPU / on CPU" summary lines.

For each stage-1 .tflite: compile, parse the operator table, and record
  - ops_on_tpu / ops_on_cpu   (summed from the table by Status)
  - total_ops                 (header line, as a cross-check)
  - onchip_used_bytes / offchip_streamed_bytes
  - mapped: TRUE (0 CPU ops) / PARTIAL (some CPU ops) / FALSE (no output)
  - nonmapped_ops             (which operators fell back, if any)

Also saves every raw compiler log to <out>/<hash>.compilerlog.txt so nothing is
ever a silent parse failure again.

Compiling is fast (~0.5s/model), so a full re-run over ~1000 models is minutes.

Usage:
    python3 stage2_v16.py --tflite-dir "/mnt/c/.../nasbench_study/tflite"
"""
import argparse, csv, glob, os, re, subprocess, sys

RE_TOTAL   = re.compile(r'Total number of operations:\s*(\d+)')
RE_ONCHIP  = re.compile(r'On-chip memory used for caching model parameters:\s*([\d.]+)([KMG]?i?B)')
RE_OFFCHIP = re.compile(r'Off-chip memory used for streaming uncached model parameters:\s*([\d.]+)([KMG]?i?B)')
RE_OP_ROW  = re.compile(r'^([A-Z_0-9]+)\s+(\d+)\s+(.+?)\s*$', re.MULTILINE)
RE_SUCCESS = re.compile(r'Compilation succeeded')

_UNIT = {'B':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3,'KB':1000,'MB':1000**2,'GB':1000**3}
def to_bytes(n,u): return int(float(n)*_UNIT.get(u,1))

FIELDS = ['hash','params','total_ops','ops_on_tpu','ops_on_cpu',
          'onchip_used_bytes','offchip_streamed_bytes','mapped',
          'nonmapped_ops','status']

def parse_log(log):
    total = RE_TOTAL.search(log)
    total = int(total.group(1)) if total else None
    tpu = cpu = 0
    nonmapped = []
    for name, count, status in RE_OP_ROW.findall(log):
        c = int(count)
        if 'Mapped to Edge TPU' in status:
            tpu += c
        else:
            cpu += c
            nonmapped.append(f'{name}x{c}')
    on = RE_ONCHIP.search(log); off = RE_OFFCHIP.search(log)
    return {
        'total_ops': total, 'ops_on_tpu': tpu, 'ops_on_cpu': cpu,
        'onchip_used_bytes': to_bytes(*on.groups()) if on else '',
        'offchip_streamed_bytes': to_bytes(*off.groups()) if off else '',
        'nonmapped_ops': ';'.join(nonmapped),
        'parsed_ok': (total is not None and (tpu+cpu) == total and total > 0),
    }

def load_csv(p):
    if not os.path.exists(p): return {}
    with open(p, newline='') as f: return {r['hash']: r for r in csv.DictReader(f)}
def save_csv(p, rows):
    tmp=p+'.tmp'
    with open(tmp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader()
        for h in rows: w.writerow({k:rows[h].get(k,'') for k in FIELDS})
    os.replace(tmp,p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tflite-dir',required=True)
    ap.add_argument('--out-dir',default=None)
    ap.add_argument('--compiler',default='edgetpu_compiler')
    a=ap.parse_args()
    if not os.path.isdir(a.tflite_dir): sys.exit(f'not found: {a.tflite_dir}')
    study=os.path.dirname(a.tflite_dir.rstrip('/'))
    out=a.out_dir or os.path.join(study,'edgetpu')
    csvp=os.path.join(study,'stage2_results.csv')
    os.makedirs(out,exist_ok=True)
    try:
        v=subprocess.run([a.compiler,'--version'],capture_output=True,text=True)
        print('Compiler:',(v.stdout or v.stderr).strip().splitlines()[0])
    except FileNotFoundError:
        sys.exit(f"'{a.compiler}' not found -- run inside WSL2 with edgetpu-compiler.")

    files=sorted(f for f in glob.glob(os.path.join(a.tflite_dir,'*.tflite'))
                 if not f.endswith('_edgetpu.tflite'))
    print(f'{len(files)} .tflite files.')

    # merge stage-1 params
    params={}
    s1=os.path.join(study,'results.csv')
    if os.path.exists(s1):
        with open(s1,newline='') as f:
            for r in csv.DictReader(f): params[r['hash']]=r.get('params','')

    rows=load_csv(csvp)
    todo=[f for f in files if os.path.splitext(os.path.basename(f))[0] not in rows]
    print(f'{len(files)-len(todo)} done; {len(todo)} to compile.\n')

    for i,tfl in enumerate(todo,1):
        h=os.path.splitext(os.path.basename(tfl))[0]
        row={k:'' for k in FIELDS}; row['hash']=h; row['params']=params.get(h,'')
        try:
            proc=subprocess.run([a.compiler,'-o',out,'-s',tfl],
                                capture_output=True,text=True,timeout=300)
            log=proc.stdout+proc.stderr
            with open(os.path.join(out,f'{h}.compilerlog.txt'),'w') as lf: lf.write(log)
            compiled=os.path.join(out,os.path.basename(tfl).replace('.tflite','_edgetpu.tflite'))
            info=parse_log(log)
            for k in ('total_ops','ops_on_tpu','ops_on_cpu','onchip_used_bytes',
                      'offchip_streamed_bytes','nonmapped_ops'):
                row[k]=info[k]
            if not os.path.exists(compiled) and not RE_SUCCESS.search(log):
                row['mapped']='FALSE'; row['status']='no_output'
            elif not info['parsed_ok']:
                row['mapped']='PARSE_FAILED'; row['status']='parse_failed_see_log'
            elif info['ops_on_cpu']==0:
                row['mapped']='TRUE'; row['status']='mapped'
            else:
                row['mapped']='PARTIAL'; row['status']=f"partial_{info['ops_on_cpu']}_cpu_ops"
        except subprocess.TimeoutExpired:
            row['mapped']='FALSE'; row['status']='compile_timeout'
        except Exception as e:
            row['mapped']='FALSE'; row['status']=f'error:{type(e).__name__}'
        rows[h]=row
        if i%25==0 or i==len(todo):
            save_csv(csvp,rows)
            t=sum(1 for r in rows.values() if r.get('mapped')=='TRUE')
            p=sum(1 for r in rows.values() if r.get('mapped')=='PARTIAL')
            fa=sum(1 for r in rows.values() if r.get('mapped')=='FALSE')
            pf=sum(1 for r in rows.values() if r.get('mapped')=='PARSE_FAILED')
            print(f'  [{i}/{len(todo)}] TRUE={t} PARTIAL={p} FALSE={fa} PARSE_FAILED={pf}')
    save_csv(csvp,rows)
    t=sum(1 for r in rows.values() if r.get('mapped')=='TRUE')
    p=sum(1 for r in rows.values() if r.get('mapped')=='PARTIAL')
    fa=sum(1 for r in rows.values() if r.get('mapped')=='FALSE')
    print(f'\nDone. TRUE(clean)={t}  PARTIAL={p}  FALSE={fa}')
    print(f'Results: {csvp}')
    if p:
        print('\nOps causing CPU fallback in PARTIAL models (sample):')
        seen=set()
        for r in rows.values():
            if r.get('mapped')=='PARTIAL' and r.get('nonmapped_ops') and r['nonmapped_ops'] not in seen:
                seen.add(r['nonmapped_ops']); print('  ',r['nonmapped_ops'])
                if len(seen)>=8: break

if __name__=='__main__': main()
