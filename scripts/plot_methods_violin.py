import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main(csv_path='results/violin/data.csv', out_png='results/violin/york_methods_violin.png', out_pdf='results/violin/york_methods_violin.pdf'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)

    # Methods in columns (order to present on x-axis)
    methods = ['GmmSeg', 'SS-Net', 'DC-Net', 'ABD', 'LSRL-Net', 'beta-FFT', 'ALHVR']
    # Collect values per method (drop NaN)
    data = [df[m].dropna().values for m in methods]

    # Figure styling: paper-friendly (use matplotlib defaults and manual grid)
    plt.style.use('default')
    ensure_dir(os.path.dirname(out_png) or '.')

    fig, ax = plt.subplots(figsize=(10, 5))

    # Color palette: muted tableau/tab10
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(methods))]

    # Violin plot
    parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.7)

    # style violin bodies with distinct but muted colors and black edges
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
        pc.set_linewidth(0.8)

    # Overlay boxplot (white box, black edges and median)
    box = ax.boxplot(data, positions=np.arange(1, len(methods) + 1), widths=0.12,
                     patch_artist=True, showfliers=False)
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)
    for k in ('whiskers', 'caps', 'medians'):
        for line in box[k]:
            line.set_color('black')
            line.set_linewidth(1.0)

    # Draw mean markers as small black dots
    means = [np.nanmean(arr) for arr in data]
    ax.scatter(np.arange(1, len(methods) + 1), means, color='black', s=20, zorder=3, marker='o', label='Mean')

    # Axes labels and ticks
    ax.set_xticks(np.arange(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=25, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Dice')
    ax.set_title('York: Test-set Dice by Method')

    # Legend (mean marker)
    ax.legend(frameon=False)

    # Tight layout and save
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f'Saved: {out_png}')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    main()
