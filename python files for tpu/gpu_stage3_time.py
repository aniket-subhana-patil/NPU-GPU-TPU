#!/usr/bin/env python3
"""
gpu_stage3_time.py -- Time int8 NAS-Bench models on the GPU via ONNX Runtime.

Provider-selectable so the SAME script handles:
  --provider cuda      -> CUDAExecutionProvider (works immediately; may run int8
                          QDQ by dequantizing internally)
  --provider tensorrt  -> TensorrtExecutionProvider with int8 enabled + engine
                          cache (true int8 on tensor cores; slow first build)
  --provider dml       -> DmlExecutionProvider (for the Radeon 780M later)

Same methodology as the Coral/NPU stage 3: warmup runs discarded, then N timed
runs, recording median/min/p95/mean latency (ms). Merges Coral latency by hash.

Run in the gpu-onnx conda env (with CUDA/cuDNN installed).

    conda activate gpu-onnx

    # ALWAYS test small first -- TensorRT engine builds are slow:
    python gpu_stage3_time.py --int8-dir npu_study\\onnx_int8 \
        --coral-csv "C:\\...\\stage3_latency.csv" \
        --provider tensorrt --limit 20

    # then full:
    python gpu_stage3_time.py --int8-dir npu_study\\onnx_int8 \
        --coral-csv "C:\\...\\stage3_latency.csv" --provider tensorrt

Resumable: skips models already timed.
"""
import argparse, csv, glob, os, sys, time
import numpy as np


def make_providers(provider, engine_cache_dir):
    if provider == 'cuda':
        return ['CUDAExecutionProvider'], None
    if provider == 'dml':
        return ['DmlExecutionProvider'], None
    if provider == 'tensorrt':
        os.makedirs(engine_cache_dir, exist_ok=True)
        # TensorRT EP options: enable int8, cache engines so each model builds once
        trt_opts = {
            'trt_int8_enable': True,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': engine_cache_dir,
            'trt_timing_cache_enable': True,
        }
        # fall back to CUDA then CPU if a subgraph won't build on TRT
        return (['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'],
                [trt_opts, {}, {}])
    sys.exit(f'unknown provider: {provider}')


OUT_FIELDS = ['hash', 'params', 'coral_latency_ms', 'coral_offchip_bytes',
              'gpu_latency_ms_median', 'gpu_latency_ms_min',
              'gpu_latency_ms_p95', 'gpu_latency_ms_mean', 'gpu_n_timed',
              'provider', 'status']


def load_csv(p):
    if not os.path.exists(p):
        return {}
    with open(p, newline='') as f:
        return {r['hash']: r for r in csv.DictReader(f)}


def save_csv(p, rows):
    tmp = p + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for h in rows:
            w.writerow({k: rows[h].get(k, '') for k in OUT_FIELDS})
    os.replace(tmp, p)


def pct(sv, q):
    if not sv:
        return ''
    k = (len(sv) - 1) * q
    lo = int(k); hi = min(lo + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--int8-dir', required=True)
    ap.add_argument('--coral-csv', default=None)
    ap.add_argument('--provider', default='cuda', choices=['cuda', 'tensorrt', 'dml'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--engine-cache', default='trt_engine_cache')
    a = ap.parse_args()

    # Make TensorRT DLLs (in the pip tensorrt_libs folder) findable before ORT loads
    # its TensorRT provider. Without this, ORT can't find nvinfer_10.dll and falls
    # back to CUDA. add_dll_directory is the clean Python 3.8+ way (no PATH edits).
    if a.provider == 'tensorrt':
        try:
            import tensorrt_libs, os as _os
            trt_lib_dir = _os.path.dirname(tensorrt_libs.__file__)
            _os.add_dll_directory(trt_lib_dir)
            print(f'Registered TensorRT DLL dir: {trt_lib_dir}')
        except Exception as _e:
            print(f'WARNING: could not register tensorrt_libs dir: {_e}')

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit('onnxruntime not found -- activate the gpu-onnx env.')

    providers, provider_options = make_providers(a.provider, a.engine_cache)
    print(f'Provider: {a.provider}  ->  {providers[0]}')
    if a.provider == 'tensorrt':
        print('TensorRT: int8 enabled, engine cache ->', a.engine_cache)
        print('NOTE: first run builds an engine per model (slow). Cached after.')

    out_path = a.out or os.path.join(
        os.path.dirname(a.int8_dir.rstrip('/\\')), f'gpu_latency_{a.provider}.csv')

    coral = load_csv(a.coral_csv) if a.coral_csv else {}

    models = sorted(glob.glob(os.path.join(a.int8_dir, '*.int8.onnx')))
    if a.limit:
        models = models[:a.limit]
    print(f'{len(models)} int8 models to time on GPU.\n')

    rows = load_csv(out_path)
    t0 = time.time()
    done = 0
    for i, mpth in enumerate(models, 1):
        h = os.path.basename(mpth).replace('.int8.onnx', '')
        if h in rows and rows[h].get('status') == 'timed':
            continue

        c = coral.get(h, {})
        row = {k: '' for k in OUT_FIELDS}
        row['hash'] = h
        row['params'] = c.get('params', '')
        row['coral_latency_ms'] = c.get('latency_ms_median', '')
        row['coral_offchip_bytes'] = c.get('offchip_streamed_bytes', '')
        row['provider'] = a.provider

        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(mpth, sess_options=so,
                                        providers=providers,
                                        provider_options=provider_options)
            actual = sess.get_providers()[0]  # which EP actually got used
            inp = sess.get_inputs()[0]
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
            x = np.random.rand(*shape).astype(np.float32)
            feed = {inp.name: x}

            for _ in range(a.warmup):
                sess.run(None, feed)

            ts = []
            for _ in range(a.runs):
                t = time.perf_counter()
                sess.run(None, feed)
                ts.append((time.perf_counter() - t) * 1000.0)

            ts.sort()
            row['gpu_latency_ms_median'] = f'{pct(ts, 0.5):.4f}'
            row['gpu_latency_ms_min'] = f'{ts[0]:.4f}'
            row['gpu_latency_ms_p95'] = f'{pct(ts, 0.95):.4f}'
            row['gpu_latency_ms_mean'] = f'{sum(ts)/len(ts):.4f}'
            row['gpu_n_timed'] = len(ts)
            row['provider'] = actual  # record the EP that really ran
            row['status'] = 'timed'
            done += 1
        except Exception as e:
            row['status'] = f'error:{type(e).__name__}:{str(e)[:70]}'
            print(f'  {h[:10]} ERROR {row["status"]}')

        rows[h] = row
        if i % 5 == 0 or i == len(models):
            save_csv(out_path, rows)
            rate = i / (time.time() - t0)
            eta = (len(models) - i) / rate / 60 if rate else 0
            last = row.get('gpu_latency_ms_median', '?')
            print(f'  [{i}/{len(models)}] timed={done} | {rate:.2f}/s | '
                  f'ETA {eta:.1f} min | last={last}ms')

    save_csv(out_path, rows)
    timed = sum(1 for r in rows.values() if r.get('status') == 'timed')
    dt = (time.time() - t0) / 60
    print(f'\nDone in {dt:.1f} min. {timed} models timed on GPU ({a.provider}) -> {out_path}')
    # note which EP actually ran (important for tensorrt: did it use TRT or fall back?)
    eps = {}
    for r in rows.values():
        if r.get('status') == 'timed':
            eps[r['provider']] = eps.get(r['provider'], 0) + 1
    print('Execution providers actually used:', eps)


if __name__ == '__main__':
    main()
