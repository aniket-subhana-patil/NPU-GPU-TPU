#!/usr/bin/env python3
"""
compare_all_three.py -- Three-way accelerator comparison across the same 1000
NAS-Bench-101 architectures: Coral Edge TPU vs Ryzen AI NPU vs RTX 4060 GPU.

Joins the three per-accelerator CSVs by hash, pulls params + cache-residency from
the Coral CSV, and produces comparison plots + a stats summary.

Runs anywhere with matplotlib + pandas.

Usage:
    python compare_all_three.py \
        --coral "C:\\...\\nasbench_study\\stage3_latency.csv" \
        --npu   npu_study\\npu_latency.csv \
        --gpu   npu_study\\gpu_latency_cuda.csv \
        --out-dir three_way_plots
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coral', required=True)
    ap.add_argument('--npu', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--out-dir', default='.')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # --- Coral: has params, cache residency, and its own latency ---
    coral = pd.read_csv(a.coral)
    coral = coral[coral['status'] == 'timed'].copy()
    coral['coral_ms'] = pd.to_numeric(coral['latency_ms_median'], errors='coerce')
    coral['params'] = pd.to_numeric(coral['params'], errors='coerce')
    coral['offchip'] = pd.to_numeric(coral['offchip_streamed_bytes'], errors='coerce').fillna(0)
    coral = coral[['hash', 'params', 'offchip', 'coral_ms']]

    # --- NPU ---
    npu = pd.read_csv(a.npu)
    npu = npu[npu['status'] == 'timed'].copy()
    npu['npu_ms'] = pd.to_numeric(npu['npu_latency_ms_median'], errors='coerce')
    npu = npu[['hash', 'npu_ms']]

    # --- GPU ---
    gpu = pd.read_csv(a.gpu)
    gpu = gpu[gpu['status'] == 'timed'].copy()
    gpu['gpu_ms'] = pd.to_numeric(gpu['gpu_latency_ms_median'], errors='coerce')
    gpu = gpu[['hash', 'gpu_ms']]

    # inner-join all three: only models timed on ALL accelerators
    df = coral.merge(npu, on='hash', how='inner').merge(gpu, on='hash', how='inner')
    df = df.dropna(subset=['coral_ms', 'npu_ms', 'gpu_ms', 'params'])
    df['overflow'] = df['offchip'] > 0

    print(f'{len(df)} models timed on all three accelerators')
    print(f'  Coral median: {df["coral_ms"].median():.3f} ms')
    print(f'  NPU median:   {df["npu_ms"].median():.3f} ms')
    print(f'  GPU median:   {df["gpu_ms"].median():.3f} ms')

    C, N, G = '#e76f51', '#2a9d8f', '#e9c46a'  # coral, npu, gpu colors

    # ---- PLOT 1: all three latency vs params ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['params']/1e6, df['coral_ms'], s=12, alpha=0.45, c=C, label='Coral Edge TPU', edgecolors='none')
    ax.scatter(df['params']/1e6, df['npu_ms'], s=12, alpha=0.45, c=N, label='Ryzen AI NPU', edgecolors='none')
    ax.scatter(df['params']/1e6, df['gpu_ms'], s=12, alpha=0.45, c=G, label='RTX 4060 GPU', edgecolors='none')
    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Three accelerators, same 1000 architectures: latency vs parameters')
    ax.legend(); ax.grid(True, alpha=0.25)
    p1 = os.path.join(a.out_dir, 'three_way_vs_params.png')
    fig.tight_layout(); fig.savefig(p1, dpi=140); plt.close(fig)

    # ---- PLOT 2: log-y version (spans big range) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, c, lab in [('coral_ms', C, 'Coral Edge TPU'), ('npu_ms', N, 'Ryzen AI NPU'), ('gpu_ms', G, 'RTX 4060 GPU')]:
        ax.scatter(df['params']/1e6, df[col], s=12, alpha=0.45, c=c, label=lab, edgecolors='none')
    ax.set_yscale('log')
    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Latency (ms, log scale)')
    ax.set_title('Three accelerators (log scale)')
    ax.legend(); ax.grid(True, alpha=0.25, which='both')
    p2 = os.path.join(a.out_dir, 'three_way_vs_params_log.png')
    fig.tight_layout(); fig.savefig(p2, dpi=140); plt.close(fig)

    # ---- PLOT 3: median latency bar chart ----
    fig, ax = plt.subplots(figsize=(7, 6))
    meds = [df['coral_ms'].median(), df['npu_ms'].median(), df['gpu_ms'].median()]
    bars = ax.bar(['Coral\nEdge TPU', 'Ryzen AI\nNPU', 'RTX 4060\nGPU'], meds, color=[C, N, G])
    for b, m in zip(bars, meds):
        ax.text(b.get_x()+b.get_width()/2, m, f'{m:.2f}ms', ha='center', va='bottom')
    ax.set_ylabel('Median latency (ms)')
    ax.set_title('Median latency across 1000 architectures')
    ax.grid(True, alpha=0.25, axis='y')
    p3 = os.path.join(a.out_dir, 'median_latency_bars.png')
    fig.tight_layout(); fig.savefig(p3, dpi=140); plt.close(fig)

    # ---- PLOT 4: latency distributions overlaid ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['coral_ms'], bins=50, alpha=0.5, color=C, label='Coral Edge TPU')
    ax.hist(df['npu_ms'], bins=50, alpha=0.5, color=N, label='Ryzen AI NPU')
    ax.hist(df['gpu_ms'], bins=50, alpha=0.5, color=G, label='RTX 4060 GPU')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Number of models')
    ax.set_title('Latency distributions')
    ax.legend(); ax.grid(True, alpha=0.25)
    p4 = os.path.join(a.out_dir, 'latency_distributions.png')
    fig.tight_layout(); fig.savefig(p4, dpi=140); plt.close(fig)

    # ---- summary ----
    stats = os.path.join(a.out_dir, 'three_way_summary.txt')
    with open(stats, 'w') as f:
        f.write('Three-way accelerator comparison -- NAS-Bench-101 (1000 architectures)\n')
        f.write('='*68 + '\n\n')
        f.write(f'Models timed on all three: {len(df)}\n\n')
        for name, col in [('Coral Edge TPU (USB, int8)', 'coral_ms'),
                          ('Ryzen AI NPU (PCIe, int8)', 'npu_ms'),
                          ('RTX 4060 GPU (CUDA, int8 models)', 'gpu_ms')]:
            s = df[col]
            f.write(f'{name}:\n')
            f.write(f'  median={s.median():.3f}  mean={s.mean():.3f}  '
                    f'min={s.min():.3f}  max={s.max():.3f} ms\n')
            f.write(f'  correlation with params: {df["params"].corr(s):.3f}\n\n')
        # cache residency effect on each
        f.write('Effect of Coral cache-overflow on each accelerator:\n')
        fit = df[~df['overflow']]; ovf = df[df['overflow']]
        for name, col in [('Coral', 'coral_ms'), ('NPU', 'npu_ms'), ('GPU', 'gpu_ms')]:
            rf = fit[col].median(); ro = ovf[col].median()
            f.write(f'  {name}: cache-fit {rf:.3f}ms vs overflow {ro:.3f}ms '
                    f'({ro/rf:.1f}x)\n')
        # who wins
        f.write('\nFastest accelerator per model (of the three):\n')
        winner = df[['coral_ms', 'npu_ms', 'gpu_ms']].idxmin(axis=1)
        for ep, lab in [('coral_ms', 'Coral'), ('npu_ms', 'NPU'), ('gpu_ms', 'GPU')]:
            n = (winner == ep).sum()
            f.write(f'  {lab} fastest on {n}/{len(df)} models ({100*n/len(df):.0f}%)\n')

    print(f'\nWrote 4 plots + summary to {a.out_dir}/')
    for p in [p1, p2, p3, p4, stats]:
        print('  ', p)


if __name__ == '__main__':
    main()
