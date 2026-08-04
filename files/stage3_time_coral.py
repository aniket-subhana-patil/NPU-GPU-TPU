#!/usr/bin/env python3
"""
stage3_time_coral.py -- run on NATIVE WINDOWS with the Coral USB accelerator
plugged in and PyCoral installed. (NOT WSL2 -- the USB device needs native access.)

Times each compiled _edgetpu.tflite model on the Edge TPU:
  - 5 warmup invokes (discarded -- first invoke includes one-time model load)
  - 100 timed invokes -> records median / min / p95 / mean latency (ms)

Merges with stage2_final.csv so the output has params + memory + latency all
together, ready to plot latency vs. params colored by cache overflow.

Usage (native Windows terminal):
    # quick test on 50 models first:
    python stage3_time_coral.py --edgetpu-dir "C:\\...\\nasbench_study\\edgetpu" --stage2-csv "C:\\...\\stage2_final.csv" --limit 50

    # then the full run (drop --limit):
    python stage3_time_coral.py --edgetpu-dir "C:\\...\\nasbench_study\\edgetpu" --stage2-csv "C:\\...\\stage2_final.csv"

Resumes: models already timed in the output CSV are skipped, so you can stop
and rerun. Order is randomized (seed-fixed) so a partial run still samples the
whole param range, not just alphabetically-early hashes.
"""
import argparse, csv, glob, os, sys, time, random

WARMUP = 5
TIMED = 100

OUT_FIELDS = ['hash','params','onchip_used_bytes','offchip_streamed_bytes',
              'total_ops','mapped',
              'latency_ms_median','latency_ms_min','latency_ms_p95',
              'latency_ms_mean','n_timed','status']

def load_csv(p):
    if not os.path.exists(p): return {}
    with open(p, newline='') as f: return {r['hash']: r for r in csv.DictReader(f)}

def save_csv(p, rows):
    tmp=p+'.tmp'
    with open(tmp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=OUT_FIELDS); w.writeheader()
        for h in rows: w.writerow({k:rows[h].get(k,'') for k in OUT_FIELDS})
    os.replace(tmp,p)

def percentile(sorted_vals, q):
    # simple linear-interp percentile on an already-sorted list
    if not sorted_vals: return ''
    k=(len(sorted_vals)-1)*q
    lo=int(k); hi=min(lo+1,len(sorted_vals)-1)
    return sorted_vals[lo] + (sorted_vals[hi]-sorted_vals[lo])*(k-lo)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--edgetpu-dir',required=True,help='folder with *_edgetpu.tflite')
    ap.add_argument('--stage2-csv',required=True,help='stage2_final.csv (params+memory)')
    ap.add_argument('--out',default=None)
    ap.add_argument('--limit',type=int,default=None,help='time only N models (for a quick test)')
    ap.add_argument('--warmup',type=int,default=WARMUP)
    ap.add_argument('--runs',type=int,default=TIMED)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    try:
        from pycoral.utils.edgetpu import make_interpreter
    except ImportError:
        sys.exit('PyCoral not found. Stage 3 needs pycoral + Coral USB on native Windows.\n'
                 'Install: pip install pycoral  (and the Edge TPU runtime from coral.ai).')
    import numpy as np

    if not os.path.isdir(a.edgetpu_dir): sys.exit(f'not found: {a.edgetpu_dir}')
    outp=a.out or os.path.join(os.path.dirname(a.stage2_csv), 'stage3_latency.csv')

    stage2=load_csv(a.stage2_csv)
    models=sorted(glob.glob(os.path.join(a.edgetpu_dir,'*_edgetpu.tflite')))
    print(f'{len(models)} compiled models found.')

    # randomize order (fixed seed) so partial runs span the param range
    random.Random(a.seed).shuffle(models)
    if a.limit:
        models=models[:a.limit]
        print(f'--limit {a.limit}: timing a {a.limit}-model subset.')

    rows=load_csv(outp)
    todo=[m for m in models
          if os.path.basename(m).replace('_edgetpu.tflite','') not in rows]
    print(f'{len(models)-len(todo)} already timed; {len(todo)} to go.\n')

    t_start=time.time()
    for i,mp in enumerate(todo,1):
        h=os.path.basename(mp).replace('_edgetpu.tflite','')
        s2=stage2.get(h,{})
        row={k:'' for k in OUT_FIELDS}
        row['hash']=h
        for k in ('params','onchip_used_bytes','offchip_streamed_bytes','total_ops','mapped'):
            row[k]=s2.get(k,'')
        try:
            interp=make_interpreter(mp)
            interp.allocate_tensors()
            inp=interp.get_input_details()[0]
            x=np.random.randint(-128,128,size=inp['shape'],dtype=inp['dtype'])
            interp.set_tensor(inp['index'],x)

            for _ in range(a.warmup):
                interp.invoke()

            ts=[]
            for _ in range(a.runs):
                t0=time.perf_counter()
                interp.invoke()
                ts.append((time.perf_counter()-t0)*1000.0)

            ts_sorted=sorted(ts)
            row['latency_ms_median']=f'{percentile(ts_sorted,0.5):.4f}'
            row['latency_ms_min']=f'{ts_sorted[0]:.4f}'
            row['latency_ms_p95']=f'{percentile(ts_sorted,0.95):.4f}'
            row['latency_ms_mean']=f'{sum(ts)/len(ts):.4f}'
            row['n_timed']=len(ts)
            row['status']='timed'
        except Exception as e:
            row['status']=f'error:{type(e).__name__}:{str(e)[:60]}'
            print(f'  {h[:10]} ERROR {row["status"]}')

        rows[h]=row
        if i%10==0 or i==len(todo):
            save_csv(outp,rows)
            done=sum(1 for r in rows.values() if r.get('status')=='timed')
            rate=i/(time.time()-t_start)
            eta=(len(todo)-i)/rate/60 if rate else 0
            # show a sample median so you can sanity-check live
            last_med=row.get('latency_ms_median','?')
            print(f'  [{i}/{len(todo)}] timed | {rate:.1f}/s | ETA {eta:.1f} min | last median={last_med}ms')

    save_csv(outp,rows)
    timed=sum(1 for r in rows.values() if r.get('status')=='timed')
    dt=(time.time()-t_start)/60
    print(f'\nDone in {dt:.1f} min. {timed} models timed -> {outp}')
    print('Columns: params, memory, and latency_ms_median -- ready to plot.')
    # quick descriptive: median latency for cache-fit vs overflow
    fit=[float(r['latency_ms_median']) for r in rows.values()
         if r.get('status')=='timed' and str(r.get('offchip_streamed_bytes')) in ('','0')]
    over=[float(r['latency_ms_median']) for r in rows.values()
          if r.get('status')=='timed' and str(r.get('offchip_streamed_bytes')) not in ('','0')]
    if fit and over:
        fit.sort(); over.sort()
        print(f'\n  cache-fit models   (n={len(fit)}): median latency {fit[len(fit)//2]:.3f} ms')
        print(f'  cache-overflow     (n={len(over)}): median latency {over[len(over)//2]:.3f} ms')
        print('  ^ if overflow is much larger than fit, that is your memory cliff, quantified.')

if __name__=='__main__': main()
