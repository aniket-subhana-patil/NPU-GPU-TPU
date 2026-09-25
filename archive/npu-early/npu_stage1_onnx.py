#!/usr/bin/env python3
"""
npu_stage1_onnx.py -- Build the SAME 1000 NAS-Bench architectures from the Coral
run and export each to ONNX (float32), ready for int8 quantization + NPU timing.

Reads the architecture hashes from the Coral run's results.csv, looks each up in
the .tfrecord to get its (matrix, ops), rebuilds it with nasbench_keras.py, and
exports to ONNX opset 17 (AMD Ryzen AI recommended). This guarantees the NPU
comparison uses byte-identical architectures to the Coral study.

Requires: tensorflow, tf2onnx, onnx, numpy, and nasbench_keras.py in same folder.

Usage:
    python npu_stage1_onnx.py --tfrecord nasbench_only108.tfrecord \
        --results-csv "C:\\...\\nasbench_study\\results.csv" \
        --out-dir npu_study\\onnx

Resumable: skips architectures whose .onnx already exists.
"""
import argparse, csv, json, os, sys
import numpy as np


def load_hashes(results_csv):
    """Read the architecture hashes (and params) from the Coral run's results.csv."""
    wanted = {}
    with open(results_csv, newline='') as f:
        for r in csv.DictReader(f):
            # only the ones that actually converted in the Coral run
            if r.get('status') == 'converted' or r.get('tflite_file'):
                wanted[r['hash']] = r.get('params', '')
    return wanted


def build_spec_index(tfrecord, wanted_hashes):
    """Scan the .tfrecord once, returning {hash: (matrix, ops)} for wanted hashes."""
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
        found[h] = (adj, ops)
        if len(found) == len(wanted_hashes):
            break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrecord', required=True)
    ap.add_argument('--results-csv', required=True, help="Coral run's results.csv")
    ap.add_argument('--out-dir', default='npu_study/onnx')
    ap.add_argument('--opset', type=int, default=17)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    try:
        import tensorflow as tf
        import tf2onnx, onnx
        from nasbench_keras import build_keras_model
    except ImportError as e:
        sys.exit(f'Missing dependency: {e}\n'
                 f'Install: pip install tensorflow tf2onnx onnx')

    print(f'Reading hashes from {a.results_csv} ...')
    wanted = load_hashes(a.results_csv)
    print(f'  {len(wanted)} architectures to rebuild.')

    print(f'Indexing specs from {a.tfrecord} (one pass) ...')
    specs = build_spec_index(a.tfrecord, wanted)
    print(f'  found {len(specs)} / {len(wanted)} specs in the dataset.')
    missing = set(wanted) - set(specs)
    if missing:
        print(f'  WARNING: {len(missing)} hashes not found in tfrecord (skipping).')

    done = 0
    errors = 0
    hashes = sorted(specs.keys())
    for i, h in enumerate(hashes, 1):
        out_path = os.path.join(a.out_dir, f'{h}.onnx')
        if os.path.exists(out_path):
            done += 1
            continue

        matrix, ops = specs[h]
        try:
            model = build_keras_model(matrix, ops)
            spec = (tf.TensorSpec((1, 32, 32, 3), tf.float32, name='input'),)
            model.output_names = ['output']
            onnx_model, _ = tf2onnx.convert.from_keras(
                model, input_signature=spec, opset=a.opset)
            onnx.save(onnx_model, out_path)
            done += 1
            tf.keras.backend.clear_session()
        except Exception as e:
            errors += 1
            print(f'  [{i}/{len(hashes)}] {h[:10]} ERROR {type(e).__name__}: {str(e)[:60]}')

        if i % 25 == 0:
            print(f'  [{i}/{len(hashes)}] exported={done} errors={errors}')

    print(f'\nDone. {done} ONNX models in {a.out_dir} (errors: {errors})')
    print('Next: quantize these to int8, then time on the NPU.')


if __name__ == '__main__':
    main()
