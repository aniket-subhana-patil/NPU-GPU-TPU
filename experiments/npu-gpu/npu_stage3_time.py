#!/usr/bin/env python3
"""
npu_stage3_time.py -- Time int8 NAS-Bench models on the Ryzen AI NPU (Phoenix).

Uses the EXACT provider config from AMD's working quicktest.py (4x4.xclbin,
target X1, provider-options method) -- NOT the firmware-env-var approach.
Times each model: warmup runs discarded, then N timed runs, recording
median / min / p95 / mean latency (ms). Same methodology as the Coral stage 3.

Run INSIDE the activated ryzen-ai-1.7.1 conda env, on native Windows.

    conda activate ryzen-ai-1.7.1

    # quick test on 20 models first:
    python npu_stage3_time.py --int8-dir npu_study\\onnx_int8 --coral-csv "C:\\...\\stage3_latency.csv" --limit 20

    # then all of them:
    python npu_stage3_time.py --int8-dir npu_study\\onnx_int8 --coral-csv "C:\\...\\stage3_latency.csv"

Merges with the Coral stage3_latency.csv (by hash) so the output has BOTH the
Coral latency and the NPU latency side by side, ready to compare/plot.
Resumable: skips models already timed.
"""
import argparse, csv, glob, os, sys, time
import numpy as np


def build_provider_options(install_dir):
    """EXACT config from AMD's working quicktest for PHX/HPT."""
    xclbin = os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '4x4.xclbin')
    return [{
        'target': 'X1',
        'xlnx_enable_py3_round': 0,
        'xclbin': xclbin,
    }]


OUT_FIELDS = ['hash', 'params',
              'coral_latency_ms', 'coral_offchip_bytes',
              'npu_latency_ms_median', 'npu_latency_ms_min',
              'npu_latency_ms_p95', 'npu_latency_ms_mean', 'npu_n_timed', 'status']


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


def percentile(sorted_vals, q):
    if not sorted_vals:
        return ''
    k = (len(sorted_vals) - 1) * q
    lo = int(k); hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--int8-dir', required=True, help='folder of *.int8.onnx models')
    ap.add_argument('--coral-csv', default=None, help='Coral stage3_latency.csv to merge')
    ap.add_argument('--install-dir', default=None, help='Ryzen AI install dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--runs', type=int, default=100)
    a = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit('onnxruntime not found -- run inside the ryzen-ai conda env.')

    install_dir = a.install_dir or os.environ.get('RYZEN_AI_INSTALLATION_PATH') \
                  or r'C:\Program Files\RyzenAI\1.7.1'
    if not os.path.isdir(install_dir):
        sys.exit(f'Ryzen AI install dir not found: {install_dir} (pass --install-dir)')

    provider_options = build_provider_options(install_dir)
    xclbin = provider_options[0]['xclbin']
    if not os.path.isfile(xclbin):
        sys.exit(f'xclbin not found: {xclbin}')
    print(f'Using xclbin: {xclbin}')

    out_path = a.out or os.path.join(os.path.dirname(a.int8_dir.rstrip('/\\')),
                                     'npu_latency.csv')

    # merge Coral results if provided
    coral = load_csv(a.coral_csv) if a.coral_csv else {}
    if coral:
        print(f'Merging Coral latencies from {a.coral_csv} ({len(coral)} rows)')

    models = sorted(glob.glob(os.path.join(a.int8_dir, '*.int8.onnx')))
    if a.limit:
        models = models[:a.limit]
    print(f'{len(models)} int8 models to time.\n')

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

        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3  # quiet
            sess = ort.InferenceSession(mpth, sess_options=so,
                                        providers=['VitisAIExecutionProvider'],
                                        provider_options=provider_options)
            inp = sess.get_inputs()[0]
            # int8 quantized model still takes float32 input (QDQ handles it)
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
            row['npu_latency_ms_median'] = f'{percentile(ts, 0.5):.4f}'
            row['npu_latency_ms_min'] = f'{ts[0]:.4f}'
            row['npu_latency_ms_p95'] = f'{percentile(ts, 0.95):.4f}'
            row['npu_latency_ms_mean'] = f'{sum(ts)/len(ts):.4f}'
            row['npu_n_timed'] = len(ts)
            row['status'] = 'timed'
            done += 1
        except Exception as e:
            row['status'] = f'error:{type(e).__name__}:{str(e)[:60]}'
            print(f'  {h[:10]} ERROR {row["status"]}')

        rows[h] = row
        if i % 10 == 0 or i == len(models):
            save_csv(out_path, rows)
            rate = i / (time.time() - t0)
            eta = (len(models) - i) / rate / 60 if rate else 0
            last = row.get('npu_latency_ms_median', '?')
            print(f'  [{i}/{len(models)}] timed={done} | {rate:.1f}/s | '
                  f'ETA {eta:.1f} min | last={last}ms')

    save_csv(out_path, rows)
    timed = sum(1 for r in rows.values() if r.get('status') == 'timed')
    dt = (time.time() - t0) / 60
    print(f'\nDone in {dt:.1f} min. {timed} models timed on NPU -> {out_path}')

    # quick Coral-vs-NPU comparison if both present
    both = [(float(r['coral_latency_ms']), float(r['npu_latency_ms_median']))
            for r in rows.values()
            if r.get('status') == 'timed' and r.get('coral_latency_ms')
            and r.get('npu_latency_ms_median')]
    if both:
        c_med = sorted(x[0] for x in both)[len(both)//2]
        n_med = sorted(x[1] for x in both)[len(both)//2]
        print(f'\n  Median latency (n={len(both)} matched models):')
        print(f'    Coral Edge TPU: {c_med:.3f} ms')
        print(f'    Ryzen AI NPU:   {n_med:.3f} ms')


if __name__ == '__main__':
    main()
