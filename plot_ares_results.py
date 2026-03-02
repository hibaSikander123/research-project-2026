import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
import os
try:
    import scienceplots
    plt.style.use(["science", "ieee", "no-latex"])
except ImportError:
    plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#for using um symbol
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'regular',
})
UNIT_UM = r'$\mu$m'

# Global font size 
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 13
FONT_SIZE_TICK = 13
FONT_SIZE_LEGEND = 13
FONT_SIZE_ANNOTATION = 13
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': FONT_SIZE_TITLE,
    'axes.labelsize': FONT_SIZE_LABEL,
    'xtick.labelsize': FONT_SIZE_TICK,
    'ytick.labelsize': FONT_SIZE_TICK,
    'legend.fontsize': FONT_SIZE_LEGEND,
})

def load_data(data_dir: str = "data/") -> Dict[str, pd.DataFrame]:
    file_mapping = {
        "Nelder-Mead": "NM_mismatched.csv",
        "BO (zero mean)": "BO_mismatched.csv",
        "BO prior (mismatched)": "BO_prior_mismatched.csv",
        "BO prior (matched)": "BO_prior_matched_prior_newtask.csv",
    }
    data = {}
    for name, filename in file_mapping.items():
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'best_mae' not in df.columns:
                df['best_mae'] = df.groupby('run')['mae'].cummin()
            data[name] = df
            print(f"Loaded {name}: {len(df)} rows, {df['run'].nunique()} runs")  
    return data

def plot_convergence_curves(
    data: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    styles = {
        "Nelder-Mead": {"ls": "-", "color": COLORS[0], "lw": 2},
        "BO (zero mean)": {"ls": "-.", "color": COLORS[1], "lw": 2},
        "BO prior (mismatched)": {"ls": "--", "color": COLORS[2], "lw": 2.5},
        "BO prior (matched)": {"ls": ":", "color": COLORS[3], "lw": 2.5},
    }
    for name, df in data.items():
        style = styles.get(name, {"ls": "-", "color": "gray", "lw": 1.5})
        df_plot = df.copy()
        df_plot['best_mae_um'] = df_plot['best_mae'] * 1e6
        sns.lineplot(data=df_plot, x="step", y="best_mae_um", ax=ax, label=name, **style)
    
    ax.set_xlabel("Iteration", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Best MAE (" + UNIT_UM + ")", fontsize=FONT_SIZE_LABEL)
    ax.set_xlim(0, None)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    return fig

def plot_boxplot_comparison(
    data: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 5),
    threshold: float = 4e-5,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    FS_TITLE = 15
    FS_LABEL = 14
    FS_TICK = 13
    FS_NOTE = 12

    styles = {
        "Nelder-Mead": {"color": COLORS[0]},
        "BO (zero mean)": {"color": COLORS[1]},
        "BO prior (mismatched)": {"color": COLORS[2]},
        "BO prior (matched)": {"color": COLORS[3]},
    }
    best_maes_data = []
    final_maes_data = []
    steps_to_target_data = []
    names = list(data.keys())

    for name, df in data.items():
        # Best MAE per run
        best_maes = df.groupby('run')['best_mae'].last().values * 1e6
        best_maes_data.append(best_maes)

        # Final MAE per run
        final_maes = df.groupby('run')['mae'].last().values * 1e6
        final_maes_data.append(final_maes)

        # Steps to target per run
        steps = []
        for run_id in df['run'].unique():
            df_run = df[df['run'] == run_id].sort_values('step')
            best_mae_history = df_run['best_mae'].values
            below = np.where(best_mae_history < threshold)[0]
            if len(below) > 0:
                steps.append(int(below[0]))
        steps_to_target_data.append(steps)

    # Panel 1: Best MAE
    ax1 = axes[0]
    bp1 = ax1.boxplot(best_maes_data, tick_labels=names, patch_artist=True)
    for patch, name in zip(bp1['boxes'], names):
        patch.set_facecolor(styles[name]['color'])
        patch.set_alpha(0.6)
    ax1.set_ylabel("Best MAE (" + UNIT_UM + ")", fontsize=FS_LABEL)
    ax1.set_title("Best MAE", fontsize=FS_TITLE)
    ax1.tick_params(axis='x', rotation=45, labelsize=FS_TICK)
    ax1.tick_params(axis='y', labelsize=FS_TICK)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Final MAE
    ax2 = axes[1]
    bp2 = ax2.boxplot(final_maes_data, tick_labels=names, patch_artist=True)
    for patch, name in zip(bp2['boxes'], names):
        patch.set_facecolor(styles[name]['color'])
        patch.set_alpha(0.6)
    ax2.set_ylabel("Final MAE (" + UNIT_UM + ")", fontsize=FS_LABEL)
    ax2.set_title("Final MAE", fontsize=FS_TITLE)
    ax2.tick_params(axis='x', rotation=45, labelsize=FS_TICK)
    ax2.tick_params(axis='y', labelsize=FS_TICK)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Steps to Target
    ax3 = axes[2]
    # plotting only methods with at least one successful run
    stt_plot_data = []
    stt_plot_names = []
    stt_plot_colors = []
    for i, name in enumerate(names):
        if len(steps_to_target_data[i]) > 0:
            stt_plot_data.append(steps_to_target_data[i])
            stt_plot_names.append(name)
            stt_plot_colors.append(styles[name]['color'])
    if len(stt_plot_data) > 0:
        bp3 = ax3.boxplot(stt_plot_data, tick_labels=stt_plot_names, patch_artist=True)
        for patch, color in zip(bp3['boxes'], stt_plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax3.set_ylabel("Steps", fontsize=FS_LABEL)
    ax3.set_title(f"Steps to Target ({threshold*1e6:.0f} " + UNIT_UM + ")", fontsize=FS_TITLE)
    ax3.tick_params(axis='x', rotation=45, labelsize=FS_TICK)
    ax3.tick_params(axis='y', labelsize=FS_TICK)
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax3.grid(True, alpha=0.3, axis='y')
    no_success = [names[i] for i in range(len(names)) if len(steps_to_target_data[i]) == 0]
    if no_success:
        note = "Not shown (0% success):\n" + ", ".join(no_success)
        ax3.text(0.95, 0.95, note, transform=ax3.transAxes,
                 fontsize=FS_NOTE, ha='right', va='top',
                 style='italic', color='black',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    return fig

def plot_best_mae_vs_convergence(
    data: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    convergence_threshold: float = 4e-5,
) -> plt.Figure:
    n_methods = len(data)
    ncols = min(2, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    colors = {
        "Nelder-Mead": COLORS[0],
        "BO (zero mean)": COLORS[1],
        "BO prior (mismatched)": COLORS[2],
        "BO prior (matched)": COLORS[3],
    }
    for idx, (name, df) in enumerate(data.items()):
        ax = axes[idx]
        color = colors.get(name, COLORS[idx % len(COLORS)])
        best_maes = []
        steps_to_conv = []
        for run_id in df['run'].unique():
            df_run = df[df['run'] == run_id].sort_values('step')
            best_mae_history = df_run['best_mae'].values
            best_maes.append(best_mae_history[-1])
            conv_step = len(best_mae_history) - 1
            for i in range(1, len(best_mae_history)):
                future = best_mae_history[i:]
                if np.max(future) - np.min(future) < convergence_threshold:
                    conv_step = i
                    break
            steps_to_conv.append(conv_step)
        best_maes = np.array(best_maes)
        steps_to_conv = np.array(steps_to_conv)
        ax.scatter(steps_to_conv, best_maes * 1e6, alpha=0.6, s=50, c=color, label=name)
        ax.set_xlabel("Steps to Convergence", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel("Best MAE (" + UNIT_UM + ")", fontsize=FONT_SIZE_LABEL)
        ax.set_title(name, fontsize=FONT_SIZE_TITLE)
        # Would use log scale only if data spans more than one order of magnitude
        y_vals = best_maes * 1e6
        if y_vals.max() / y_vals.min() > 10:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: ''))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
        ax.grid(True, alpha=0.3, linestyle='--')
        stats_text = f"Median: {np.median(best_maes)*1e6:.2f} " + UNIT_UM + f"\nMean: {np.mean(best_maes)*1e6:.2f} " + UNIT_UM
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=FONT_SIZE_ANNOTATION, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    for idx in range(len(data), len(axes)):
        axes[idx].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    return fig
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate ARES optimization plots")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="results/")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args.data_dir)
    if len(data) == 0:
        print("No data files found!")
        exit(1)    
    plot_convergence_curves(data, save_path=os.path.join(args.output_dir, "convergence_curves.pdf"))
    plot_boxplot_comparison(data, save_path=os.path.join(args.output_dir, "boxplot_comparison.pdf"))
    plot_best_mae_vs_convergence(data, save_path=os.path.join(args.output_dir, "best_mae_vs_convergence.pdf"))