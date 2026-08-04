#!/usr/bin/env python3
"""
local_stage1_parallel.py

STAGE 1, run LOCALLY across all CPU cores. Reads NAS-Bench-101 architectures
directly from the .tfrecord (raw digit-string parser -- no `nasbench` package,
no TF1/protobuf issues), builds each Keras model, converts to int8 TFLite, and
writes the tflite/ + results.csv layout that stages 2 and 3 expect.

Uses multiprocessing.Pool so N models convert simultaneously. On an 8-core CPU
this is ~6-8x faster than single-threaded / Colab.

Usage:
    python local_stage1_parallel.py --tfrecord path\\to\\nasbench_only108.tfrecord --n 1000

    # control worker count (default = CPU count):
    python local_stage1_parallel.py --tfrecord ... --n 1000 --workers 8

Resumes automatically: rows already marked 'converted' in results.csv are
skipped, so you can Ctrl+C and rerun freely.

Requires: tensorflow, numpy, and nasbench_keras.py in the same folder.
"""

import argparse
import base64  # noqa: F401  (kept for compatibility; raw adj is digit-string)
import csv
import json
import multiprocessing as mp
import os
import time

import numpy as np


# ----------------------------------------------------------------------------
# Raw reader -- verified against the real file. Field [2] is a flat digit
# string like '0101...' of length dim*dim (NOT base64).
# ----------------------------------------------------------------------------
def read_all_specs(path):
    """Return list of (hash, matrix(np.int8), ops(list[str])), deduplicated."""
    import tensorflow as tf
    seen = set()
    specs = []
    for serialized_row in tf.compat.v1.io.tf_record_iterator(path):
        module_hash, epochs, raw_adj, raw_ops, _metrics = \
            json.loads(serialized_row.decode('utf-8'))
        if module_hash in seen:
            continue
        seen.add(module_hash)
        ops = raw_ops.split(',')
        dim = len(ops)
        if len(raw_adj) != dim * dim:
            continue  # malformed row; skip
        adj = np.array([int(c) for c in raw_adj], dtype=np.int8).reshape(dim, dim)
        specs.append((module_hash, adj, ops))
    return specs


# ----------------------------------------------------------------------------
# Worker: build + convert ONE model. Runs in a separate process.
# ----------------------------------------------------------------------------
# Each worker imports TF lazily and silences its chatter, so 8 processes don't
# each dump startup logs. The output dir is passed so the worker writes its own
# .tflite and returns a result row for the parent to checkpoint.
def _worker_init():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # quiet the per-process logging
    import warnings
    warnings.filterwarnings('ignore')


def _convert_one(args):
    """args = (hash, matrix, ops, tflite_dir, rep_samples). Returns a result dict."""
    module_hash, matrix, ops, tflite_dir, rep_samples = args
    out_path = os.path.join(tflite_dir, f'{module_hash}.tflite')
    if os.path.exists(out_path):
        # already done in a prior run; report size + params unknown-from-here
        return {'hash': module_hash, 'status': 'converted',
                'tflite_file': f'{module_hash}.tflite',
                'tflite_bytes': os.path.getsize(out_path), 'params': ''}
    try:
        import tensorflow as tf  # noqa: F401
        from nasbench_keras import build_keras_model, convert_int8_tflite
        model = build_keras_model(matrix, ops)
        params = model.count_params()
        tfl = convert_int8_tflite(model, rep_samples=rep_samples)
        with open(out_path, 'wb') as f:
            f.write(tfl)
        return {'hash': module_hash, 'status': 'converted',
                'tflite_file': f'{module_hash}.tflite',
                'tflite_bytes': len(tfl), 'params': params}
    except Exception as e:
        return {'hash': module_hash, 'status': f'error:{type(e).__name__}',
                'tflite_file': '', 'tflite_bytes': '', 'params': ''}


# ----------------------------------------------------------------------------
CSV_FIELDS = ['hash', 'params', 'tflite_file', 'tflite_bytes', 'status']


def load_done(csv_path):
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline='') as f:
        return {r['hash']: r for r in csv.DictReader(f)}


def save_rows(csv_path, rows):
    tmp = csv_path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for h in rows:
            w.writerow({k: rows[h].get(k, '') for k in CSV_FIELDS})
    os.replace(tmp, csv_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrecord', required=True, help='path to nasbench .tfrecord')
    ap.add_argument('--n', type=int, default=1000, help='number of models to sample')
    ap.add_argument('--workers', type=int, default=None,
                    help='parallel worker processes (default: CPU count)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bins', type=int, default=20, help='param stratification bins')
    ap.add_argument('--rep-samples', type=int, default=100)
    ap.add_argument('--out', default='nasbench_study', help='output directory')
    args = ap.parse_args()

    workers = args.workers or mp.cpu_count()
    tflite_dir = os.path.join(args.out, 'tflite')
    csv_path = os.path.join(args.out, 'results.csv')
    os.makedirs(tflite_dir, exist_ok=True)

    print(f'Reading architectures from {args.tfrecord} ...')
    specs = read_all_specs(args.tfrecord)
    print(f'  {len(specs)} unique architectures.')

    # Stratify by edge-count proxy so the sample spans small->large modules.
    # (True params are recorded per model as they convert.)
    edge_counts = np.array([int(s[1].sum()) for s in specs])
    bins = np.linspace(edge_counts.min(), edge_counts.max() + 1, args.bins + 1)
    which = np.digitize(edge_counts, bins)
    per_bin = max(1, args.n // args.bins)
    rng = np.random.default_rng(args.seed)

    chosen = []
    for b in range(1, args.bins + 1):
        idx = np.where(which == b)[0]
        if len(idx):
            chosen.extend(rng.choice(idx, size=min(per_bin, len(idx)),
                                     replace=False).tolist())
    if len(chosen) < args.n:
        remaining = list(set(range(len(specs))) - set(chosen))
        chosen.extend(rng.choice(remaining,
                      size=min(args.n - len(chosen), len(remaining)),
                      replace=False).tolist())
    chosen = chosen[:args.n]
    print(f'Sampled {len(chosen)} architectures (seed={args.seed}).')

    rows = load_done(csv_path)
    todo = [specs[i] for i in chosen
            if not (rows.get(specs[i][0], {}).get('status') == 'converted'
                    and os.path.exists(os.path.join(tflite_dir, f'{specs[i][0]}.tflite')))]
    print(f'{len(chosen) - len(todo)} already done; {len(todo)} to convert '
          f'across {workers} workers.\n')

    if not todo:
        print('Nothing to do -- all sampled models already converted.')
        return

    tasks = [(h, m, o, tflite_dir, args.rep_samples) for (h, m, o) in todo]

    t0 = time.time()
    completed = 0
    # imap_unordered streams results back as each worker finishes, so we can
    # checkpoint progressively and show a live rate/ETA.
    with mp.Pool(workers, initializer=_worker_init) as pool:
        for result in pool.imap_unordered(_convert_one, tasks):
            rows[result['hash']] = {'hash': result['hash'],
                                    'params': result['params'],
                                    'tflite_file': result['tflite_file'],
                                    'tflite_bytes': result['tflite_bytes'],
                                    'status': result['status']}
            completed += 1
            if result['status'] != 'converted':
                print(f"  {result['hash'][:10]} SKIP {result['status']}")
            if completed % 20 == 0:
                save_rows(csv_path, rows)
                rate = completed / (time.time() - t0)
                eta = (len(todo) - completed) / rate / 60 if rate else 0
                ok = sum(1 for r in rows.values() if r.get('status') == 'converted')
                print(f'  [{completed}/{len(todo)}] {ok} ok total | '
                      f'{rate*60:.1f}/min | ETA {eta:.0f} min')

    save_rows(csv_path, rows)
    ok = sum(1 for r in rows.values() if r.get('status') == 'converted')
    dt = (time.time() - t0) / 60
    print(f'\nDone in {dt:.1f} min. {ok} models converted to int8 TFLite in {tflite_dir}')
    print('Next: zip nothing needed -- run stages 2 (WSL2 compiler) and 3 (Coral) '
          'on this folder directly.')


if __name__ == '__main__':
    # Windows requires the guard + spawn; this is already the default on Win.
    main()
