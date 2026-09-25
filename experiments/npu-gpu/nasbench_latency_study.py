#!/usr/bin/env python3
"""
nasbench_latency_study.py

Three-stage harness to measure Edge TPU inference latency vs. parameter count
across a stratified sample of NAS-Bench-101 architectures.

Because the Edge TPU compiler is Linux-only but the Coral runtime works on
Windows, the three stages run in different environments. This one script drives
all three via a --stage flag, resuming through a shared checkpoint CSV so each
stage can run where its tools live.

    Stage 1  (native Windows, TensorFlow):
        Sample specs, build Keras models, convert to int8 .tflite, count params.
        --> writes results.csv rows + .tflite files
        python nasbench_latency_study.py --stage 1 \
            --nasbench path/to/nasbench_only108.tfrecord --n 5000

    Stage 2  (WSL2 / Linux, edgetpu_compiler installed):
        Compile each .tflite, parse the log for on-TPU vs on-CPU ops and the
        on-chip/off-chip memory report. Marks each model mapped / not-mapped.
        --> updates results.csv, writes _edgetpu.tflite files
        python nasbench_latency_study.py --stage 2

    Stage 3  (native Windows, PyCoral + Coral USB plugged in):
        For each mapped model, time invoke() on the Coral. Records median latency.
        --> updates results.csv with latency_ms
        python nasbench_latency_study.py --stage 3

Design choices baked in (per your answers):
  - Fixed seed (SEED below) -> the SAME 5000 specs every run, reproducible.
  - Skip-and-log on any failure -> one bad spec never halts the batch; the
    reason is recorded in the 'status' column.
  - All state in results.csv so stages resume independently across environments.

Only Stage 1 needs TensorFlow. Only Stage 2 needs the compiler. Only Stage 3
needs PyCoral + hardware. Each stage imports lazily so you can run stage 2 in a
Python that doesn't even have TF installed.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

import tensorflow as tf

# TF1->TF2 compatibility shim for the (TF1-era) nasbench package.
# We only use nasbench's dataset-READING path, never its training code, but the
# import chain pulls in training_time.py which references many TF1 tf.train.*
# symbols at module load. Stub them so the import completes.
class _Stub:
    """Inert placeholder; nasbench subclasses these but we never instantiate them."""
    def __init__(self, *a, **k):
        pass

if not hasattr(tf, 'train'):
    tf.train = type(sys)('train')

for _name in (
    'SessionRunHook', 'SessionRunArgs', 'CheckpointSaverListener',
    'SessionRunContext', 'SessionRunValues', 'LoggingTensorHook',
    'CheckpointSaverHook', 'SecondOrStepTimer', 'NanLossDuringTrainingError',
    'Saver', 'get_or_create_global_step',
):
    if not hasattr(tf.train, _name):
        setattr(tf.train, _name, _Stub)

for _name in ('gfile', 'logging', 'Session', 'placeholder'):
    if not hasattr(tf, _name) and hasattr(tf.compat.v1, _name):
        setattr(tf, _name, getattr(tf.compat.v1, _name))
# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 42
NUM_BINS = 20            # stratification bins across the param range
OUTPUT_DIR = 'nasbench_study'
TFLITE_DIR = os.path.join(OUTPUT_DIR, 'tflite')
EDGETPU_DIR = os.path.join(OUTPUT_DIR, 'edgetpu')
CSV_PATH = os.path.join(OUTPUT_DIR, 'results.csv')
REP_SAMPLES = 100        # representative-dataset size for int8 calibration
WARMUP_INVOKES = 5
TIMED_INVOKES = 100

CSV_FIELDS = [
    'hash', 'params',            # from NAS-Bench
    'tflite_path', 'tflite_bytes', 'status',   # stage 1
    'edgetpu_path', 'ops_total', 'ops_on_tpu', 'ops_on_cpu',
    'onchip_used_bytes', 'offchip_streamed_bytes', 'mapped',   # stage 2
    'latency_ms_median', 'latency_ms_min', 'latency_ms_p95',   # stage 3
]


# ----------------------------------------------------------------------------
# CSV helpers (the shared checkpoint across stages/environments)
# ----------------------------------------------------------------------------
def load_rows():
    if not os.path.exists(CSV_PATH):
        return {}
    with open(CSV_PATH, newline='') as f:
        return {r['hash']: r for r in csv.DictReader(f)}


def write_rows(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp = CSV_PATH + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for h in rows:
            w.writerow({k: rows[h].get(k, '') for k in CSV_FIELDS})
    os.replace(tmp, CSV_PATH)   # atomic; safe if interrupted


def blank_row(h):
    return {k: '' for k in CSV_FIELDS} | {'hash': h}


# ----------------------------------------------------------------------------
# STAGE 1: sample + build + int8 convert
# ----------------------------------------------------------------------------
def stage1(nasbench_path, n):
    import numpy as np
    from nasbench import api
    from nasbench_keras import build_keras_model, convert_int8_tflite

    os.makedirs(TFLITE_DIR, exist_ok=True)
    rows = load_rows()

    print(f'Loading NAS-Bench from {nasbench_path} (slow)...')
    nb = api.NASBench(nasbench_path)

    # Cheap pass: read (hash, param count) for the whole space from the table.
    print('Reading all architecture param counts...')
    all_specs = []
    for h in nb.hash_iterator():
        fixed, _ = nb.get_metrics_from_hash(h)
        all_specs.append((h, int(fixed['trainable_parameters'])))
    params = np.array([p for _, p in all_specs])
    print(f'  {len(all_specs)} architectures, '
          f'params {params.min()}..{params.max()}')

    # Stratified sample: even coverage across the param range.
    bins = np.linspace(params.min(), params.max() + 1, NUM_BINS + 1)
    which = np.digitize(params, bins)
    per_bin = max(1, n // NUM_BINS)
    rng = np.random.default_rng(SEED)   # fixed seed -> reproducible sample

    chosen = []
    for b in range(1, NUM_BINS + 1):
        idx = np.where(which == b)[0]
        if len(idx):
            take = min(per_bin, len(idx))
            chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    # Top up to exactly n if bin rounding left us short.
    if len(chosen) < n:
        remaining = list(set(range(len(all_specs))) - set(chosen))
        extra = rng.choice(remaining, size=min(n - len(chosen), len(remaining)),
                           replace=False).tolist()
        chosen.extend(extra)
    chosen = chosen[:n]
    print(f'Sampled {len(chosen)} specs across {NUM_BINS} bins '
          f'(seed={SEED}, reproducible).')

    done = sum(1 for h, _ in (all_specs[i] for i in chosen)
               if h in rows and rows[h].get('status') == 'converted')
    print(f'{done} already converted; resuming.\n')

    for count, i in enumerate(chosen, 1):
        h, pcount = all_specs[i]
        if h in rows and rows[h].get('status') == 'converted' \
                and os.path.exists(rows[h].get('tflite_path', '')):
            continue

        row = rows.get(h, blank_row(h))
        row['params'] = pcount
        try:
            fixed, _ = nb.get_metrics_from_hash(h)
            model = build_keras_model(fixed['module_adjacency'],
                                      fixed['module_operations'])
            tfl = convert_int8_tflite(model, rep_samples=REP_SAMPLES)
            path = os.path.join(TFLITE_DIR, f'{h}.tflite')
            with open(path, 'wb') as f:
                f.write(tfl)
            row['tflite_path'] = path
            row['tflite_bytes'] = len(tfl)
            row['status'] = 'converted'
        except Exception as e:
            # Skip-and-log: record the reason, keep going.
            row['status'] = f'convert_error:{type(e).__name__}'
            print(f'  [{count}/{len(chosen)}] {h[:12]} SKIP {row["status"]}')

        rows[h] = row
        if count % 25 == 0:
            write_rows(rows)   # periodic checkpoint
            print(f'  [{count}/{len(chosen)}] converted, checkpointed')

    write_rows(rows)
    ok = sum(1 for r in rows.values() if r.get('status') == 'converted')
    print(f'\nStage 1 done. {ok} models converted to int8 .tflite in {TFLITE_DIR}')
    print('Next: run --stage 2 inside WSL2 (edgetpu_compiler).')


# ----------------------------------------------------------------------------
# STAGE 2: compile + parse mapping report
# ----------------------------------------------------------------------------
# Regexes for the edgetpu_compiler stdout.
_RE_ON_TPU = re.compile(r'Number of operations that will run on Edge TPU:\s*(\d+)')
_RE_ON_CPU = re.compile(r'Number of operations that will run on CPU:\s*(\d+)')
_RE_ONCHIP = re.compile(r'On-chip memory used for caching model parameters:\s*([\d.]+)([KMG]?i?B)')
_RE_OFFCHIP = re.compile(r'Off-chip memory used for streaming uncached model parameters:\s*([\d.]+)([KMG]?i?B)')

_UNIT = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3,
         'KB': 1000, 'MB': 1000**2, 'GB': 1000**3}


def _to_bytes(num, unit):
    return int(float(num) * _UNIT.get(unit, 1))


def stage2(compiler='edgetpu_compiler'):
    os.makedirs(EDGETPU_DIR, exist_ok=True)
    rows = load_rows()
    if not rows:
        sys.exit('No results.csv found. Run stage 1 first.')

    # Sanity: is the compiler actually here?
    try:
        v = subprocess.run([compiler, '--version'],
                           capture_output=True, text=True)
        print('Compiler:', v.stdout.strip().splitlines()[0] if v.stdout else '?')
    except FileNotFoundError:
        sys.exit(f"'{compiler}' not found. Stage 2 must run where the Edge TPU "
                 f"compiler is installed (Linux / WSL2). "
                 f"Install: add Coral's apt repo, then "
                 f"`sudo apt-get install edgetpu-compiler`.")

    todo = [h for h, r in rows.items()
            if r.get('status') == 'converted' and not r.get('mapped')]
    print(f'{len(todo)} models to compile.\n')

    for count, h in enumerate(todo, 1):
        row = rows[h]
        tfl = row['tflite_path']
        # WSL2 sees Windows paths under /mnt/c/... ; if the CSV stored a Windows
        # path, translate it. Otherwise use as-is.
        if not os.path.exists(tfl) and re.match(r'^[A-Za-z]:\\', tfl):
            drive, rest = tfl[0].lower(), tfl[2:].replace('\\', '/')
            tfl = f'/mnt/{drive}{rest}'
        if not os.path.exists(tfl):
            row['mapped'] = 'FALSE'
            row['status'] = 'tflite_missing_stage2'
            continue

        try:
            proc = subprocess.run(
                [compiler, '-o', EDGETPU_DIR, '-s', tfl],
                capture_output=True, text=True, timeout=300)
            log = proc.stdout + proc.stderr

            on_tpu = _RE_ON_TPU.search(log)
            on_cpu = _RE_ON_CPU.search(log)
            onchip = _RE_ONCHIP.search(log)
            offchip = _RE_OFFCHIP.search(log)

            row['ops_on_tpu'] = on_tpu.group(1) if on_tpu else ''
            row['ops_on_cpu'] = on_cpu.group(1) if on_cpu else ''
            if on_tpu and on_cpu:
                row['ops_total'] = str(int(on_tpu.group(1)) + int(on_cpu.group(1)))
            if onchip:
                row['onchip_used_bytes'] = _to_bytes(*onchip.groups())
            if offchip:
                row['offchip_streamed_bytes'] = _to_bytes(*offchip.groups())

            edgetpu_file = os.path.join(
                EDGETPU_DIR, os.path.basename(tfl).replace('.tflite', '_edgetpu.tflite'))

            # "Mapped" = compiler produced a delegated model AND nothing (or
            # nothing past the first block) fell back to CPU. We treat 0 CPU ops
            # as cleanly mapped; >0 CPU ops means partitioned -> flag it.
            cpu_ops = int(on_cpu.group(1)) if on_cpu else 999
            if os.path.exists(edgetpu_file) and cpu_ops == 0:
                row['mapped'] = 'TRUE'
                row['edgetpu_path'] = edgetpu_file
                row['status'] = 'mapped'
            elif os.path.exists(edgetpu_file):
                row['mapped'] = 'PARTIAL'
                row['edgetpu_path'] = edgetpu_file
                row['status'] = f'partial_{cpu_ops}_cpu_ops'
            else:
                row['mapped'] = 'FALSE'
                row['status'] = 'compile_no_output'
        except Exception as e:
            row['mapped'] = 'FALSE'
            row['status'] = f'compile_error:{type(e).__name__}'
            print(f'  [{count}/{len(todo)}] {h[:12]} SKIP {row["status"]}')

        rows[h] = row
        if count % 25 == 0:
            write_rows(rows)
            print(f'  [{count}/{len(todo)}] compiled, checkpointed')

    write_rows(rows)
    mapped = sum(1 for r in rows.values() if r.get('mapped') == 'TRUE')
    partial = sum(1 for r in rows.values() if r.get('mapped') == 'PARTIAL')
    print(f'\nStage 2 done. {mapped} cleanly mapped, {partial} partial. '
          f'Next: --stage 3 on native Windows with the Coral plugged in.')


# ----------------------------------------------------------------------------
# STAGE 3: time on the Coral
# ----------------------------------------------------------------------------
def stage3():
    import numpy as np
    try:
        from pycoral.utils.edgetpu import make_interpreter
    except ImportError:
        sys.exit('PyCoral not found. Stage 3 needs pycoral + the Coral USB '
                 'accelerator plugged in (native Windows is fine).')

    rows = load_rows()
    if not rows:
        sys.exit('No results.csv found. Run stages 1 and 2 first.')

    todo = [h for h, r in rows.items()
            if r.get('mapped') == 'TRUE' and not r.get('latency_ms_median')]
    print(f'{len(todo)} mapped models to time on the Coral.\n')

    for count, h in enumerate(todo, 1):
        row = rows[h]
        model_path = row.get('edgetpu_path', '')
        if not os.path.exists(model_path):
            row['status'] = 'edgetpu_file_missing_stage3'
            rows[h] = row
            continue
        try:
            interp = make_interpreter(model_path)
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            x = np.random.randint(-128, 128, size=inp['shape'], dtype=np.int8)
            interp.set_tensor(inp['index'], x)

            for _ in range(WARMUP_INVOKES):   # discard one-time load-to-TPU cost
                interp.invoke()

            times = []
            for _ in range(TIMED_INVOKES):
                t0 = time.perf_counter()
                interp.invoke()
                times.append((time.perf_counter() - t0) * 1000.0)  # ms

            times = np.array(times)
            row['latency_ms_median'] = f'{np.median(times):.4f}'
            row['latency_ms_min'] = f'{times.min():.4f}'
            row['latency_ms_p95'] = f'{np.percentile(times, 95):.4f}'
            row['status'] = 'timed'
        except Exception as e:
            row['status'] = f'timing_error:{type(e).__name__}'
            print(f'  [{count}/{len(todo)}] {h[:12]} SKIP {row["status"]}')

        rows[h] = row
        if count % 25 == 0:
            write_rows(rows)
            print(f'  [{count}/{len(todo)}] timed, checkpointed')

    write_rows(rows)
    timed = sum(1 for r in rows.values() if r.get('status') == 'timed')
    print(f'\nStage 3 done. {timed} models timed. '
          f'Latency + params are in {CSV_PATH} -- ready to plot.')


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--nasbench', help='path to nasbench .tfrecord (stage 1)')
    ap.add_argument('--n', type=int, default=5000, help='sample size (stage 1)')
    ap.add_argument('--compiler', default='edgetpu_compiler',
                    help='compiler binary name/path (stage 2)')
    args = ap.parse_args()

    if args.stage == 1:
        if not args.nasbench:
            ap.error('--nasbench is required for stage 1')
        stage1(args.nasbench, args.n)
    elif args.stage == 2:
        stage2(args.compiler)
    elif args.stage == 3:
        stage3()


if __name__ == '__main__':
    main()
