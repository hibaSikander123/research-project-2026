import os
import sys
import argparse
import numpy as np
import pandas as pd

from eval_ares_metrics import (
    load_and_evaluate,
    print_table,
    create_results_dataframe,
)
from plot_ares_results import (
    load_data,
    plot_convergence_curves,
    plot_boxplot_comparison,
    plot_best_mae_vs_convergence,
)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ARES BO optimization results")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory containing CSV result files (default: data/)")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Directory to save outputs (default: results/)")
    parser.add_argument("--threshold", type=float, default=4e-5,
                        help="MAE threshold for 'target reached' in meters (default: 4e-5 = 40µm)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--show_plots", action="store_true",
                        help="Display plots interactively")
    args = parser.parse_args()
    
    # Creating output directory
    os.makedirs(args.output_dir, exist_ok=True)
    file_paths = {
        "Nelder-Mead": os.path.join(args.data_dir, "NM_mismatched.csv"),
        "BO (zero mean)": os.path.join(args.data_dir, "BO_mismatched.csv"),
        "BO_prior (mismatched)": os.path.join(args.data_dir, "BO_prior_mismatched.csv"),
        "BO_prior (matched)": os.path.join(args.data_dir, "BO_prior_matched_prior_newtask.csv"),
    }
    # Checking which files exist
    existing_files = {k: v for k, v in file_paths.items() if os.path.exists(v)} 
    for name, path in file_paths.items():
        status = "found" if os.path.exists(path) else "not found"
        print(f"  {status} {name}: {path}")
    if len(existing_files) == 0:
        print("No data files found. Please check the data folder.")
        return 1
    
    # Loading and computing metrics
    studies, episodes = load_and_evaluate(
        existing_files,
        threshold=args.threshold
    )
    threshold_um = args.threshold * 1e6
    print_table(studies, threshold_um=threshold_um)
    results_df = create_results_dataframe(studies)
    csv_path = os.path.join(args.output_dir, "evaluation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f" Results saved to: {csv_path}")
    
    # Generating plots
    if not args.no_plots:
        print("  GENERATING PLOTS")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data = load_data(args.data_dir)
        if len(data) > 0:
            plot_convergence_curves(data, save_path=os.path.join(args.output_dir, "convergence_curves.pdf"))
            plot_boxplot_comparison(data, save_path=os.path.join(args.output_dir, "boxplot_comparison.pdf"))
            plot_best_mae_vs_convergence(data, save_path=os.path.join(args.output_dir, "mae_vs_convergence.pdf"))
            plt.close('all')
            print(f"All plots saved to: {args.output_dir}")
            if args.show_plots:
                plt.show()
    return 0

if __name__ == "__main__":
    sys.exit(main())