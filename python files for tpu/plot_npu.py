#!/usr/bin/env python3
"""
plot_npu.py -- Plot the Ryzen AI NPU latency results (standalone, mirrors the
Coral plot_results.py). Reads npu_latency.csv for NPU latencies, and pulls
params + Coral cache-residency from the Coral stage3 csv by hash (since the NPU
csv's own params column came out blank).

Runs anywhere with matplotlib + pandas.

Usage:
    python plot_npu.py --npu-csv npu_study\\npu_latency.csv \
        --coral-csv "C:\\...\\nasbench_study\\stage3_latency.csv" \
        --out-dir npu_plots
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npu-csv', required=True)
    ap.add_argument('--coral-csv', required=True, help='for params + cache residency')
    ap.add_argument('--out-dir', default='.')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    npu = pd.read_csv(a.npu_csv)
    npu = npu[npu['status'] == 'timed'].copy()
    npu['npu_latency_ms_median'] = pd.to_numeric(npu['npu_latency_ms_median'], errors='coerce')
    # drop npu's blank params/coral cols so the coral merge brings clean ones
    npu = npu.drop(columns=[c for c in ['params','coral_latency_ms','coral_offchip_bytes'] if c in npu.columns])

    # pull params + offchip from Coral csv by hash
    coral = pd.read_csv(a.coral_csv)
    coral = coral[['hash', 'params', 'offchip_streamed_bytes']].copy()
    coral['params'] = pd.to_numeric(coral['params'], errors='coerce')
    coral['offchip_streamed_bytes'] = pd.to_numeric(coral['offchip_streamed_bytes'], errors='coerce')

    df = npu.merge(coral, on='hash', how='left')
    df = df.dropna(subset=['npu_latency_ms_median', 'params'])
    df['overflow'] = df['offchip_streamed_bytes'].fillna(0) > 0

    fit = df[~df['overflow']]; over = df[df['overflow']]
    print(f'{len(df)} NPU models with params')
    print(f'  NPU median latency: {df["npu_latency_ms_median"].median():.3f} ms')
    print(f'  (Coral cache-fit n={len(fit)}, overflow n={len(over)})')

    # ---- PLOT 1: NPU latency vs params, colored by Coral cache residency ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fit['params']/1e6, fit['npu_latency_ms_median'], s=16, alpha=0.6,
               c='#e9c46a', label=f'Coral cache-fit (n={len(fit)})', edgecolors='none')
    ax.scatter(over['params']/1e6, over['npu_latency_ms_median'], s=16, alpha=0.6,
               c='#e76f51', label=f'Coral cache-overflow (n={len(over)})', edgecolors='none')
    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Ryzen AI NPU latency (ms)')
    ax.set_title('NAS-Bench-101 on Ryzen AI NPU: latency vs parameters')
    ax.legend(); ax.grid(True, alpha=0.25)
    p1 = os.path.join(a.out_dir, 'npu_latency_vs_params.png')
    fig.tight_layout(); fig.savefig(p1, dpi=140); plt.close(fig)

    # ---- PLOT 2: log-y ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fit['params']/1e6, fit['npu_latency_ms_median'], s=16, alpha=0.6,
               c='#e9c46a', label=f'cache-fit (n={len(fit)})', edgecolors='none')
    ax.scatter(over['params']/1e6, over['npu_latency_ms_median'], s=16, alpha=0.6,
               c='#e76f51', label=f'cache-overflow (n={len(over)})', edgecolors='none')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('NPU latency (ms, log scale)')
    ax.set_title('NPU latency vs parameters (log scale)')
    ax.legend(); ax.grid(True, alpha=0.25, which='both')
    p2 = os.path.join(a.out_dir, 'npu_latency_vs_params_log.png')
    fig.tight_layout(); fig.savefig(p2, dpi=140); plt.close(fig)

    # ---- PLOT 3: distribution ----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(df['npu_latency_ms_median'], bins=50, alpha=0.75, color='#457b9d')
    ax.axvline(df['npu_latency_ms_median'].median(), color='r', linestyle='-',
               alpha=0.7, label=f'median {df["npu_latency_ms_median"].median():.2f}ms')
    ax.set_xlabel('NPU latency (ms)')
    ax.set_ylabel('Number of models')
    ax.set_title('NPU latency distribution')
    ax.legend(); ax.grid(True, alpha=0.25)
    p3 = os.path.join(a.out_dir, 'npu_latency_distribution.png')
    fig.tight_layout(); fig.savefig(p3, dpi=140); plt.close(fig)

    # ---- summary ----
    stats = os.path.join(a.out_dir, 'npu_summary.txt')
    with open(stats, 'w') as f:
        f.write('Ryzen AI NPU -- NAS-Bench-101 latency summary\n')
        f.write('='*45 + '\n\n')
        f.write(f'Models timed: {len(df)}\n\n')
        s = df['npu_latency_ms_median']
        f.write(f'NPU latency (ms): median={s.median():.3f} mean={s.mean():.3f} '
                f'min={s.min():.3f} max={s.max():.3f}\n\n')
        f.write('By Coral cache residency:\n')
        for name, g in [('cache-fit', fit), ('cache-overflow', over)]:
            if len(g):
                f.write(f'  {name} (n={len(g)}): NPU median '
                        f'{g["npu_latency_ms_median"].median():.3f}ms\n')
        f.write(f'\nCorrelation NPU latency ~ params: '
                f'{df["params"].corr(df["npu_latency_ms_median"]):.3f}\n')

    print(f'\nWrote 3 plots + summary to {a.out_dir}/')
    for p in [p1, p2, p3, stats]:
        print('  ', p)


if __name__ == '__main__':
    main()
