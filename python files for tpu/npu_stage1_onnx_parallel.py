#!/usr/bin/env python3
"""
npu_stage1_onnx_parallel.py -- Build the SAME 1000 NAS-Bench architectures from
the Coral run and export each to ONNX, PARALLELIZED across all CPU cores.

Same output as npu_stage1_onnx.py but uses multiprocessing.Pool so N models
export simultaneously (~6-8x faster on 8 cores).

Reads architecture hashes from the Coral run's results.csv, looks each up in the
.tfrecord, rebuilds with nasbench_keras.py, exports to ONNX opset 17.

Requires: tensorflow, tf2onnx, onnx, numpy, nasbench_keras.py (same folder).

Usage:
    python npu_stage1_onnx_parallel.py --tfrecord nasbench_only108.tfrecord \
        --results-csv "C:\\...\\nasbench_study\\results.csv" \
        --out-dir npu_study\\onnx

    # control workers (default = CPU count):
    python npu_stage1_onnx_parallel.py ... --workers 8

Resumable: skips architectures whose .onnx already exists.
"""
import argparse, csv, json, os, sys, time
import multiprocessing as mp
import numpy as np


def load_hashes(results_csv):
    wanted = {}
    with open(results_csv, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('status') == 'converted' or r.get('tflite_file'):
                wanted[r['hash']] = r.get('params', '')
    return wanted


def build_spec_index(tfrecord, wanted_hashes):
    import tensorflow as tf
    found = {}
    for serialized in tf.compat.v1.io.tf_record_iterator(tfrecord):
        h, epochs, raw_adj, raw_ops, _ = json.loads(serialized.decode('utf-8'))
        if h not in wanted_hashes or h in found:
            continue
        ops = raw_ops.split(',')
        dim = len(ops)
        if len(raw_adj) != dim * dim:
            continue
        adj = np.array([int(c) for c in raw_adj], dtype=np.int8).reshape(dim, dim)
        found[h] = (adj.tolist(), ops)   # lists so they pickle cleanly to workers
        if len(found) == len(wanted_hashes):
            break
    return found


# ---- worker: export ONE model to ONNX (runs in its own process) ----
def _worker_init():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.filterwarnings('ignore')


def _export_one(args):
    h, matrix, ops, out_dir, opset = args
    out_path = os.path.join(out_dir, f'{h}.onnx')
    if os.path.exists(out_path):
        return (h, 'skip')
    try:
        import numpy as _np
        import tensorflow as tf
        import tf2onnx, onnx
        from nasbench_keras import build_keras_model

        model = build_keras_model(_np.array(matrix, dtype=_np.int8), ops)
        spec = (tf.TensorSpec((1, 32, 32, 3), tf.float32, name='input'),)
        model.output_names = ['output']
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=opset)
        onnx.save(onnx_model, out_path)
        return (h, 'ok')
    except Exception as e:
        return (h, f'error:{type(e).__name__}:{str(e)[:50]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrecord', required=True)
    ap.add_argument('--results-csv', required=True)
    ap.add_argument('--out-dir', default='npu_study/onnx')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()

    workers = a.workers or mp.cpu_count()
    os.makedirs(a.out_dir, exist_ok=True)

    # dependency check in the parent before spawning workers
    try:
        import tensorflow, tf2onnx, onnx
        from nasbench_keras import build_keras_model  # noqa
    except ImportError as e:
        sys.exit(f'Missing dependency: {e}\nInstall: pip install tensorflow tf2onnx onnx')

    print(f'Reading hashes from {a.results_csv} ...')
    wanted = load_hashes(a.results_csv)
    print(f'  {len(wanted)} architectures.')

    print(f'Indexing specs from {a.tfrecord} (one pass) ...')
    specs = build_spec_index(a.tfrecord, wanted)
    print(f'  found {len(specs)} / {len(wanted)} specs.')

    tasks = [(h, m, o, a.out_dir, a.opset) for h, (m, o) in specs.items()]
    # skip already-done up front for a clean count
    todo = [t for t in tasks if not os.path.exists(os.path.join(a.out_dir, f'{t[0]}.onnx'))]
    print(f'{len(tasks) - len(todo)} already exported; {len(todo)} to go '
          f'across {workers} workers.\n')

    if not todo:
        print('All done already.')
        return

    t0 = time.time()
    ok = err = 0
    with mp.Pool(workers, initializer=_worker_init) as pool:
        for i, (h, status) in enumerate(pool.imap_unordered(_export_one, todo), 1):
            if status == 'ok' or status == 'skip':
                ok += 1
            else:
                err += 1
                print(f'  {h[:10]} {status}')
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f'  [{i}/{len(todo)}] ok={ok} err={err} | '
                      f'{rate:.1f}/s | ETA {eta:.1f} min')

    dt = (time.time() - t0) / 60
    print(f'\nDone in {dt:.1f} min. {ok} ONNX models in {a.out_dir} (errors: {err})')
    print('Next: quantize to int8, then time on the NPU.')


if __name__ == '__main__':
    main()
