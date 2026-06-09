"""
plot_seamount_volumes.py

Read Balleny_seamount_volumes_ref.xlsx and plot a 2×2 panel figure:
  [0,0] vol_edifice_km3        per seamount
  [0,1] vol_infill_km3         per seamount
  [1,0] vol_underplating_km3   per seamount
  [1,1] vol_edifice_km3 / GEBCO_volume_km3  per seamount
         (dashed reference lines at 0.8 and 1.2)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── default style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    'black',
    'axes.labelcolor':   'black',
    'xtick.color':       'black',
    'ytick.color':       'black',
    'text.color':        'black',
    'savefig.facecolor': 'white',
    'grid.color':        '#cccccc',
})


def plot_volume_panels(
    xlsx_path: str = 'results/Balleny_seamount_volumes_ref.xlsx',
    save_path='results/Balleny_volume_panels.png',
    dpi=150,
) -> plt.Figure:
    """
    Read *xlsx_path* and draw a 2×2 bar-chart figure.

    Parameters
    ----------
    xlsx_path  : path to the reference volumes Excel file
    save_path  : output PNG path (None = do not save)
    dpi        : resolution for the saved figure

    Returns
    -------
    fig : matplotlib Figure
    """

    df = pd.read_excel(xlsx_path)
    names = df['seamount_name'].tolist()
    x     = np.arange(len(names))
    bar_w = 0.55
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']   # blue / green / red / purple

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Balleny Seamount Volumes', fontsize=14, fontweight='bold', y=1.01)

    # ── helper: common bar-chart formatting ───────────────────────────────────
    def _bar(ax, values, ylabel, color, title):
        bars = ax.bar(x, values, width=bar_w, color=color, edgecolor='white',
                      linewidth=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f'{v:,.0f}'))
        ax.grid(axis='y', linewidth=0.5, zorder=0)
        ax.set_xlim(-0.5, len(names) - 0.5)
        # value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f'{val:,.0f}',
                    ha='center', va='bottom', fontsize=7.5, color='black')

    # ── [0,0] edifice ─────────────────────────────────────────────────────────
    _bar(axes[0, 0],
         df['vol_edifice_km3'].values,
         'Volume (km³)',
         colors[0],
         'Edifice Volume')

    # ── [0,1] infill ──────────────────────────────────────────────────────────
    _bar(axes[0, 1],
         df['vol_infill_km3'].values,
         'Volume (km³)',
         colors[1],
         'Infill Volume')

    # ── [1,0] underplating ────────────────────────────────────────────────────
    _bar(axes[1, 0],
         df['vol_underplating_km3'].values,
         'Volume (km³)',
         colors[2],
         'Underplating Volume')

    # ── [1,1] ratio: edifice / GEBCO ─────────────────────────────────────────
    ratio = (df['vol_edifice_km3'] / df['GEBCO_volume_km3']).values
    ax4   = axes[1, 1]
    bars  = ax4.bar(x, ratio, width=bar_w, color=colors[3], edgecolor='white',
                    linewidth=0.6, zorder=3)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
    ax4.set_ylabel('Edifice / GEBCO volume', fontsize=10)
    ax4.set_title('Edifice / GEBCO Volume Ratio', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', linewidth=0.5, zorder=0)
    ax4.set_xlim(-0.5, len(names) - 0.5)

    # reference lines
    ax4.axhline(0.8, color='steelblue', linewidth=1.4,
                linestyle='--', zorder=4, label='0.8')
    ax4.axhline(1.2, color='tomato',    linewidth=1.4,
                linestyle='--', zorder=4, label='1.2')
    ax4.legend(fontsize=9, framealpha=0.9, title='Reference',
               title_fontsize=8)

    # value labels
    for bar, val in zip(bars, ratio):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 1.02,
                 f'{val:.2f}',
                 ha='center', va='bottom', fontsize=7.5, color='black')

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f'Saved → {save_path}')

    return fig


# ── run directly ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    fig = plot_volume_panels()
    plt.show()
