#!/usr/bin/env python3
"""
compare_coral_npu.py -- Compare Edge TPU (Coral) vs Ryzen AI NPU latency across
the same NAS-Bench architectures. Reads npu_latency.csv (which has both Coral and
NPU latency per model, merged during NPU timing) and produces comparison plots +
stats.

Runs anywhere with matplotlib + pandas (no hardware needed).

Usage:
    python compare_coral_npu.py --csv npu_study\\npu_latency.csv --out-dir compare_plots
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out-dir', default='.')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.csv)
    df = df[df['status'] == 'timed'].copy()
    for col in ['params', 'coral_latency_ms', 'npu_latency_ms_median', 'coral_offchip_bytes']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # keep rows with both latencies
    df = df.dropna(subset=['coral_latency_ms', 'npu_latency_ms_median'])
    df['overflow'] = df['coral_offchip_bytes'].fillna(0) > 0

    print(f'{len(df)} models with both Coral and NPU latency')
    print(f'  Coral median: {df["coral_latency_ms"].median():.3f} ms')
    print(f'  NPU median:   {df["npu_latency_ms_median"].median():.3f} ms')

    # ---- PLOT 1: direct scatter, Coral latency (x) vs NPU latency (y) ----
    fig, ax = plt.subplots(figsize=(8, 8))
    fit = df[~df['overflow']]; over = df[df['overflow']]
    ax.scatter(fit['coral_latency_ms'], fit['npu_latency_ms_median'], s=16, alpha=0.5,
               c='#2a9d8f', label=f'Coral cache-fit (n={len(fit)})', edgecolors='none')
    ax.scatter(over['coral_latency_ms'], over['npu_latency_ms_median'], s=16, alpha=0.5,
               c='#e76f51', label=f'Coral cache-overflow (n={len(over)})', edgecolors='none')
    lim = max(df['coral_latency_ms'].max(), df['npu_latency_ms_median'].max())
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='equal latency')
    ax.set_xlabel('Coral Edge TPU latency (ms)')
    ax.set_ylabel('Ryzen AI NPU latency (ms)')
    ax.set_title('Edge TPU vs NPU latency (same architectures)')
    ax.legend(); ax.grid(True, alpha=0.25)
    p1 = os.path.join(a.out_dir, 'coral_vs_npu_scatter.png')
    fig.tight_layout(); fig.savefig(p1, dpi=140); plt.close(fig)

    # ---- PLOT 2: both vs params ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df['params']/1e6, df['coral_latency_ms'], s=14, alpha=0.5,
               c='#264653', label='Coral Edge TPU', edgecolors='none')
    ax.scatter(df['params']/1e6, df['npu_latency_ms_median'], s=14, alpha=0.5,
               c='#e9c46a', label='Ryzen AI NPU', edgecolors='none')
    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency vs parameters: Edge TPU vs NPU')
    ax.legend(); ax.grid(True, alpha=0.25)
    p2 = os.path.join(a.out_dir, 'both_vs_params.png')
    fig.tight_layout(); fig.savefig(p2, dpi=140); plt.close(fig)

    # ---- PLOT 3: speedup ratio histogram ----
    df['speedup'] = df['coral_latency_ms'] / df['npu_latency_ms_median']
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(df['speedup'], bins=50, alpha=0.75, color='#457b9d')
    ax.axvline(1.0, color='k', linestyle='--', alpha=0.6, label='equal')
    ax.axvline(df['speedup'].median(), color='r', linestyle='-', alpha=0.7,
               label=f'median {df["speedup"].median():.2f}x')
    ax.set_xlabel('Speedup (Coral latency / NPU latency)  — >1 means NPU faster')
    ax.set_ylabel('Number of models')
    ax.set_title('Per-model speedup: NPU vs Edge TPU')
    ax.legend(); ax.grid(True, alpha=0.25)
    p3 = os.path.join(a.out_dir, 'speedup_histogram.png')
    fig.tight_layout(); fig.savefig(p3, dpi=140); plt.close(fig)

    # ---- summary ----
    stats = os.path.join(a.out_dir, 'comparison_summary.txt')
    with open(stats, 'w') as f:
        f.write('Edge TPU (Coral) vs Ryzen AI NPU -- latency comparison\n')
        f.write('='*55 + '\n\n')
        f.write(f'Matched models: {len(df)}\n\n')
        for name, col in [('Coral Edge TPU', 'coral_latency_ms'),
                          ('Ryzen AI NPU', 'npu_latency_ms_median')]:
            s = df[col]
            f.write(f'{name} latency (ms): median={s.median():.3f} '
                    f'mean={s.mean():.3f} min={s.min():.3f} max={s.max():.3f}\n')
        f.write(f'\nPer-model speedup (Coral/NPU): median={df["speedup"].median():.2f}x '
                f'mean={df["speedup"].mean():.2f}x\n')
        npu_faster = (df['speedup'] > 1).sum()
        f.write(f'NPU faster on {npu_faster}/{len(df)} models '
                f'({100*npu_faster/len(df):.0f}%)\n')
        # split by coral cache residency
        f.write(f'\nBy Coral cache residency:\n')
        for name, g in [('cache-fit', df[~df['overflow']]),
                        ('cache-overflow', df[df['overflow']])]:
            if len(g):
                f.write(f'  {name} (n={len(g)}): '
                        f'Coral {g["coral_latency_ms"].median():.3f}ms  '
                        f'NPU {g["npu_latency_ms_median"].median():.3f}ms\n')

    print(f'\nWrote 3 plots + summary to {a.out_dir}/')
    for p in [p1, p2, p3, stats]:
        print('  ', p)


if __name__ == '__main__':
    main()
