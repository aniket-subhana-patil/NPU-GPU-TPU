#!/usr/bin/env python3
"""
stage2_compile.py  --  run INSIDE WSL2 / Linux (needs edgetpu_compiler)

Runs edgetpu_compiler over every .tflite from stage 1, parses each compiler log,
and records for each model:
  - ops_on_tpu / ops_on_cpu        (how much mapped to the Edge TPU)
  - onchip_used_bytes              (model params cached on-chip)
  - offchip_streamed_bytes         (params streamed from host -> the cache cliff)
  - mapped: TRUE / PARTIAL / FALSE (the verdict)

Writes stage2_results.csv next to the tflite folder. This CSV is your core
result: which NAS-Bench architectures run on the Edge TPU, and how cleanly.

Usage (from inside Ubuntu/WSL2):
    python3 stage2_compile.py --tflite-dir "/mnt/c/Users/.../nasbench_study/tflite"

    # optional: where compiled outputs + logs go (default: sibling 'edgetpu' dir)
    python3 stage2_compile.py --tflite-dir ... --out-dir ...

Resumes: models already in stage2_results.csv are skipped, so you can Ctrl+C
and rerun. If a stage-1 results.csv is found (for param counts), it's merged in.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys

# ---- compiler-log regexes -------------------------------------------------
RE_ON_TPU = re.compile(r'Number of operations that will run on Edge TPU:\s*(\d+)')
RE_ON_CPU = re.compile(r'Number of operations that will run on CPU:\s*(\d+)')
RE_ONCHIP = re.compile(r'On-chip memory used for caching model parameters:\s*([\d.]+)([KMG]?i?B)')
RE_OFFCHIP = re.compile(r'Off-chip memory used for streaming uncached model parameters:\s*([\d.]+)([KMG]?i?B)')

_UNIT = {'B': 1, 'KiB': 1024, 'MiB': 1024**2, 'GiB': 1024**3,
         'KB': 1000, 'MB': 1000**2, 'GB': 1000**3}


def to_bytes(num, unit):
    return int(float(num) * _UNIT.get(unit, 1))


FIELDS = ['hash', 'params', 'ops_on_tpu', 'ops_on_cpu',
          'onchip_used_bytes', 'offchip_streamed_bytes', 'mapped', 'status']


def load_csv(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        return {r['hash']: r for r in csv.DictReader(f)}


def save_csv(path, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for h in rows:
            w.writerow({k: rows[h].get(k, '') for k in FIELDS})
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tflite-dir', required=True,
                    help='folder with stage-1 .tflite files (use /mnt/c/... path)')
    ap.add_argument('--out-dir', default=None,
                    help='where compiled _edgetpu.tflite + logs go')
    ap.add_argument('--compiler', default='edgetpu_compiler')
    args = ap.parse_args()

    tflite_dir = args.tflite_dir
    if not os.path.isdir(tflite_dir):
        sys.exit(f'tflite dir not found: {tflite_dir}\n'
                 f'(From WSL2, Windows files are under /mnt/c/...)')

    study_dir = os.path.dirname(tflite_dir.rstrip('/'))
    out_dir = args.out_dir or os.path.join(study_dir, 'edgetpu')
    csv_path = os.path.join(study_dir, 'stage2_results.csv')
    os.makedirs(out_dir, exist_ok=True)

    # Confirm the compiler exists before doing anything.
    try:
        v = subprocess.run([args.compiler, '--version'],
                           capture_output=True, text=True)
        print('Compiler:', (v.stdout or v.stderr).strip().splitlines()[0])
    except FileNotFoundError:
        sys.exit(f"'{args.compiler}' not found. Run this inside WSL2/Linux with "
                 f"edgetpu-compiler installed.")

    files = sorted(glob.glob(os.path.join(tflite_dir, '*.tflite')))
    # don't re-compile already-compiled outputs if they slipped into the dir
    files = [f for f in files if not f.endswith('_edgetpu.tflite')]
    print(f'{len(files)} .tflite files found in {tflite_dir}')

    rows = load_csv(csv_path)

    # Merge param counts from stage-1 results.csv if present.
    stage1_csv = os.path.join(study_dir, 'results.csv')
    params_by_hash = {}
    if os.path.exists(stage1_csv):
        with open(stage1_csv, newline='') as f:
            for r in csv.DictReader(f):
                params_by_hash[r['hash']] = r.get('params', '')
        print(f'Merged param counts from {stage1_csv}')

    todo = [f for f in files
            if os.path.splitext(os.path.basename(f))[0] not in rows]
    print(f'{len(files) - len(todo)} already compiled; {len(todo)} to go.\n')

    for i, tfl in enumerate(todo, 1):
        h = os.path.splitext(os.path.basename(tfl))[0]
        row = {'hash': h, 'params': params_by_hash.get(h, ''),
               'ops_on_tpu': '', 'ops_on_cpu': '',
               'onchip_used_bytes': '', 'offchip_streamed_bytes': '',
               'mapped': '', 'status': ''}
        try:
            proc = subprocess.run(
                [args.compiler, '-o', out_dir, '-s', tfl],
                capture_output=True, text=True, timeout=300)
            log = proc.stdout + proc.stderr

            m_tpu = RE_ON_TPU.search(log)
            m_cpu = RE_ON_CPU.search(log)
            m_on = RE_ONCHIP.search(log)
            m_off = RE_OFFCHIP.search(log)

            row['ops_on_tpu'] = m_tpu.group(1) if m_tpu else ''
            row['ops_on_cpu'] = m_cpu.group(1) if m_cpu else ''
            if m_on:
                row['onchip_used_bytes'] = to_bytes(*m_on.groups())
            if m_off:
                row['offchip_streamed_bytes'] = to_bytes(*m_off.groups())

            compiled = os.path.join(
                out_dir, os.path.basename(tfl).replace('.tflite', '_edgetpu.tflite'))
            cpu_ops = int(m_cpu.group(1)) if m_cpu else 999

            if os.path.exists(compiled) and cpu_ops == 0:
                row['mapped'] = 'TRUE'
                row['status'] = 'mapped'
            elif os.path.exists(compiled):
                row['mapped'] = 'PARTIAL'
                row['status'] = f'partial_{cpu_ops}_cpu_ops'
            else:
                row['mapped'] = 'FALSE'
                row['status'] = 'no_output'
        except subprocess.TimeoutExpired:
            row['mapped'] = 'FALSE'
            row['status'] = 'compile_timeout'
        except Exception as e:
            row['mapped'] = 'FALSE'
            row['status'] = f'error:{type(e).__name__}'

        rows[h] = row
        if i % 20 == 0 or i == len(todo):
            save_csv(csv_path, rows)
            mapped = sum(1 for r in rows.values() if r.get('mapped') == 'TRUE')
            partial = sum(1 for r in rows.values() if r.get('mapped') == 'PARTIAL')
            no = sum(1 for r in rows.values() if r.get('mapped') == 'FALSE')
            print(f'  [{i}/{len(todo)}]  mapped={mapped} partial={partial} '
                  f'unmapped={no}')

    save_csv(csv_path, rows)
    mapped = sum(1 for r in rows.values() if r.get('mapped') == 'TRUE')
    partial = sum(1 for r in rows.values() if r.get('mapped') == 'PARTIAL')
    no = sum(1 for r in rows.values() if r.get('mapped') == 'FALSE')
    total = mapped + partial + no
    print(f'\nStage 2 complete. {total} models compiled:')
    print(f'  cleanly mapped : {mapped}')
    print(f'  partial (CPU fallback ops): {partial}')
    print(f'  not mapped     : {no}')
    print(f'\nResults: {csv_path}')
    print('Next: stage 3 -- time the "mapped" (and optionally "partial") models '
          'on the Coral, on native Windows with PyCoral.')


if __name__ == '__main__':
    main()
