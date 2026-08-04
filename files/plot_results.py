#!/usr/bin/env python3
"""
plot_results.py -- visualize the NAS-Bench Edge TPU latency study.

Reads stage3_latency.csv and produces plots showing the central finding:
latency is governed by cache residency (does the model fit in the ~8MB on-chip
memory), not raw parameter count.

Runs in any Python with matplotlib + pandas (no Coral needed).

Usage:
    python plot_results.py --csv stage3_latency.csv
    python plot_results.py --csv stage3_latency.csv --out-dir plots
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # file output, no display needed
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out-dir', default='.')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.csv)
    # keep only successfully-timed rows with numeric fields
    df = df[df['status'] == 'timed'].copy()
    for col in ['params', 'latency_ms_median', 'offchip_streamed_bytes', 'onchip_used_bytes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['params', 'latency_ms_median'])

    # classify cache residency
    df['offchip_streamed_bytes'] = df['offchip_streamed_bytes'].fillna(0)
    df['overflow'] = df['offchip_streamed_bytes'] > 0
    fit = df[~df['overflow']]
    over = df[df['overflow']]

    print(f'{len(df)} models | cache-fit: {len(fit)} | overflow: {len(over)}')
    print(f'median latency  fit: {fit["latency_ms_median"].median():.3f} ms  '
          f'overflow: {over["latency_ms_median"].median():.3f} ms')

    # ---- PLOT 1: latency vs params, colored by cache residency (the headline) ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fit['params']/1e6, fit['latency_ms_median'], s=18, alpha=0.6,
               c='#2a9d8f', label=f'Fits in on-chip cache (n={len(fit)})', edgecolors='none')
    ax.scatter(over['params']/1e6, over['latency_ms_median'], s=18, alpha=0.6,
               c='#e76f51', label=f'Overflows cache → streams (n={len(over)})', edgecolors='none')
    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Median Edge TPU latency (ms)')
    ax.set_title('NAS-Bench-101 on Coral Edge TPU: latency is set by cache residency,\nnot parameter count')
    ax.legend()
    ax.grid(True, alpha=0.25)
    p1 = os.path.join(a.out_dir, 'latency_vs_params.png')
    fig.tight_layout(); fig.savefig(p1, dpi=140); plt.close(fig)

    # ---- PLOT 2: same, log-y, to show both populations clearly ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fit['params']/1e6, fit['latency_ms_median'], s=18, alpha=0.6,
               c='#2a9d8f', label=f'Fits in cache (n={len(fit)})', edgecolors='none')
    ax.scatter(over['params']/1e6, over['latency_ms_median'], s=18, alpha=0.6,
               c='#e76f51', label=f'Overflows cache (n={len(over)})', edgecolors='none')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Median Edge TPU latency (ms, log scale)')
    ax.set_title('Latency vs parameters (log scale) — two distinct populations')
    ax.legend(); ax.grid(True, alpha=0.25, which='both')
    p2 = os.path.join(a.out_dir, 'latency_vs_params_log.png')
    fig.tight_layout(); fig.savefig(p2, dpi=140); plt.close(fig)

    # ---- PLOT 3: latency vs ON-CHIP MEMORY USED, showing the cliff at ~8MB ----
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df['onchip_used_bytes']/1e6, df['latency_ms_median'],
                    s=18, alpha=0.6, c=df['overflow'].map({False:'#2a9d8f', True:'#e76f51'}),
                    edgecolors='none')
    ax.set_xlabel('On-chip memory used (MB)')
    ax.set_ylabel('Median Edge TPU latency (ms)')
    ax.set_title('The cliff: latency jumps when the model no longer fits on-chip (~8MB)')
    ax.grid(True, alpha=0.25)
    p3 = os.path.join(a.out_dir, 'latency_vs_onchip_memory.png')
    fig.tight_layout(); fig.savefig(p3, dpi=140); plt.close(fig)

    # ---- PLOT 4: distribution (histogram) of latencies, two groups ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(fit['latency_ms_median'], bins=40, alpha=0.6, color='#2a9d8f', label='Fits in cache')
    ax.hist(over['latency_ms_median'], bins=40, alpha=0.6, color='#e76f51', label='Overflows cache')
    ax.set_xlabel('Median latency (ms)')
    ax.set_ylabel('Number of models')
    ax.set_title('Latency distribution: bimodal, split by cache residency')
    ax.legend(); ax.grid(True, alpha=0.25)
    p4 = os.path.join(a.out_dir, 'latency_distribution.png')
    fig.tight_layout(); fig.savefig(p4, dpi=140); plt.close(fig)

    # summary stats file
    stats = os.path.join(a.out_dir, 'summary_stats.txt')
    with open(stats, 'w') as f:
        f.write('NAS-Bench-101 Edge TPU latency study — summary\n')
        f.write('='*50 + '\n\n')
        f.write(f'Total models timed: {len(df)}\n')
        f.write(f'  Cache-fit:     {len(fit)}\n')
        f.write(f'  Cache-overflow: {len(over)}\n\n')
        for name, g in [('Cache-fit', fit), ('Cache-overflow', over), ('All', df)]:
            l = g['latency_ms_median']
            f.write(f'{name} latency (ms): median={l.median():.3f}  '
                    f'mean={l.mean():.3f}  min={l.min():.3f}  max={l.max():.3f}\n')
        ratio = over['latency_ms_median'].median() / fit['latency_ms_median'].median()
        f.write(f'\nOverflow/fit median ratio: {ratio:.1f}x\n')
        # correlation of latency with params, within each group
        f.write(f'\nCorrelation latency~params:\n')
        f.write(f'  within cache-fit:     {fit["params"].corr(fit["latency_ms_median"]):.3f}\n')
        f.write(f'  within cache-overflow: {over["params"].corr(over["latency_ms_median"]):.3f}\n')
        f.write(f'  overall:              {df["params"].corr(df["latency_ms_median"]):.3f}\n')

    print(f'\nWrote 4 plots + summary to {a.out_dir}/')
    for p in [p1,p2,p3,p4,stats]: print('  ', p)

if __name__=='__main__': main()
