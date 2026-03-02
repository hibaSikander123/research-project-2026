import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Metrics for a single optimisation run
@dataclass
class EpisodeMetrics:
    run_id: int
    final_mae: float
    best_mae: float
    initial_mae: float
    rmse: float
    improvement: float
    improvement_best: float
    steps_to_target: Optional[int]
    steps_to_convergence: int
    mae_history: np.ndarray
    best_mae_history: np.ndarray

# Aggregated metrics (multiple runs of same optimiser)
@dataclass  
class StudyMetrics:
    name: str
    n_runs: int
    #Best MAE cumulative minimum
    best_mae_median: float
    best_mae_mean: float
    best_mae_std: float
    #Final MAE
    final_mae_median: float
    final_mae_mean: float
    final_mae_std: float
    #RMSE
    rmse_median: float
    rmse_mean: float
    rmse_std: float
    #Improvement percentage
    improvement_median: float
    improvement_mean: float
    improvement_std: float
    #Improvement percentage (based on best MAE)
    improvement_best_median: float
    improvement_best_mean: float
    improvement_best_std: float
    #steps to target (40micrometers)
    steps_to_target_median: Optional[float]
    steps_to_target_mean: Optional[float]
    steps_to_target_std: Optional[float]
    success_rate: float
    steps_to_convergence_median: float
    steps_to_convergence_mean: float
    

def compute_episode_metrics(
    df_run: pd.DataFrame,
    threshold: float = 4e-5,
    convergence_threshold: float = 4e-5,
) -> EpisodeMetrics:
    df_run = df_run.sort_values('step').reset_index(drop=True)
    mae_history = df_run['mae'].values
    if 'best_mae' in df_run.columns:
        best_mae_history = df_run['best_mae'].values
    else:
        best_mae_history = np.minimum.accumulate(mae_history)
    #mae
    initial_mae = mae_history[0]
    final_mae = mae_history[-1]
    best_mae = best_mae_history[-1]
    #RMSE
    beam_params = df_run[['mu_x', 'mu_y', 'sigma_x', 'sigma_y']].values
    rmse = np.sqrt(np.mean(np.square(beam_params)))
    if initial_mae > 0:
        improvement = (initial_mae - final_mae) / initial_mae * 100
        improvement_best = (initial_mae - best_mae) / initial_mae * 100
    else:
        improvement = 0.0
        improvement_best = 0.0
    below_threshold = np.where(best_mae_history < threshold)[0]
    if len(below_threshold) > 0:
        steps_to_target = int(below_threshold[0])
    else:
        steps_to_target = None
    steps_to_convergence = len(mae_history) - 1
    for i in range(1, len(best_mae_history)):
        future_maes = best_mae_history[i:]
        if len(future_maes) > 0:
            variation = np.max(future_maes) - np.min(future_maes)
            if variation < convergence_threshold:
                steps_to_convergence = i
                break
    
    return EpisodeMetrics(
        run_id=int(df_run['run'].iloc[0]) if 'run' in df_run.columns else 0,
        final_mae=final_mae,
        best_mae=best_mae,
        initial_mae=initial_mae,
        rmse=rmse,
        improvement=improvement,
        improvement_best=improvement_best,
        steps_to_target=steps_to_target,
        steps_to_convergence=steps_to_convergence,
        mae_history=mae_history,
        best_mae_history=best_mae_history,
    )

def compute_study_metrics(
    df: pd.DataFrame,
    name: str,
    threshold: float = 4e-5,
    convergence_threshold: float = 4e-5,
) -> Tuple[StudyMetrics, List[EpisodeMetrics]]:
    episodes = []
    for run_id in df['run'].unique():
        df_run = df[df['run'] == run_id]
        episode = compute_episode_metrics(df_run, threshold, convergence_threshold)
        episodes.append(episode)
    n_runs = len(episodes)
    best_maes = np.array([ep.best_mae for ep in episodes])
    final_maes = np.array([ep.final_mae for ep in episodes])    
    # Improvement percentages
    improvements_best = np.array([ep.improvement_best for ep in episodes])
    improvements_final = np.array([ep.improvement for ep in episodes])
    #RMSES
    rmses = np.array([ep.rmse for ep in episodes])
    # Steps to target and success rate
    steps_to_targets = [ep.steps_to_target for ep in episodes if ep.steps_to_target is not None]
    success_rate = len(steps_to_targets) / n_runs * 100

    if len(steps_to_targets) > 0:
        steps_arr = np.array(steps_to_targets)
        stt_median = np.median(steps_arr)
        stt_mean = np.mean(steps_arr)
        stt_std = np.std(steps_arr)
    else:
        stt_median = None
        stt_mean = None
        stt_std = None
    convergence_steps = np.array([ep.steps_to_convergence for ep in episodes])
    
    return StudyMetrics(
        name=name,
        n_runs=n_runs,
        best_mae_median=np.median(best_maes),
        best_mae_mean=np.mean(best_maes),
        best_mae_std=np.std(best_maes),
        final_mae_median=np.median(final_maes),
        final_mae_mean=np.mean(final_maes),
        final_mae_std=np.std(final_maes),
        rmse_median=np.median(rmses),
        rmse_mean=np.mean(rmses),
        rmse_std=np.std(rmses),
        improvement_best_median=np.median(improvements_best),
        improvement_best_mean=np.mean(improvements_best),
        improvement_best_std=np.std(improvements_best),
        improvement_median=np.median(improvements_final),
        improvement_mean=np.mean(improvements_final),
        improvement_std=np.std(improvements_final),
        steps_to_target_median=stt_median,
        steps_to_target_mean=stt_mean,
        steps_to_target_std=stt_std,
        success_rate=success_rate,
        steps_to_convergence_median=np.median(convergence_steps),
        steps_to_convergence_mean=np.mean(convergence_steps),
    ), episodes

def print_table(studies: List[StudyMetrics], threshold_um: float = 40.0):
    print(f"  METRICS TABLE")
    print(f"  Target threshold: {threshold_um:.0f} µm")    
    print(f"\n{'Optimizer':<30s} | {'Best MAE (µm)':<25s} | {'Final MAE (µm)':<25s} | {'RMSE (µm)':<25s} | {'Improv. Best (%)':<25s} | {'Improv. Final (%)':<25s} | {'Steps to Target':<20s} | {'Success':<8s}")
    print("-" * 100)
    
    for s in studies:
        best_err_str = f"{s.best_mae_median*1e6:.2f} / {s.best_mae_mean*1e6:.2f} ± {s.best_mae_std*1e6:.2f}"
        final_err_str = f"{s.final_mae_median*1e6:.2f} / {s.final_mae_mean*1e6:.2f} ± {s.final_mae_std*1e6:.2f}"
        rmse_str = f"{s.rmse_median*1e6:.2f} / {s.rmse_mean*1e6:.2f} ± {s.rmse_std*1e6:.2f}"
        improvement_str = f"{s.improvement_median:.1f} / {s.improvement_mean:.1f} ± {s.improvement_std:.1f}"
        improvement_best_str = f"{s.improvement_best_median:.1f} / {s.improvement_best_mean:.1f} ± {s.improvement_best_std:.1f}"
        
        if s.steps_to_target_median is not None:
            steps_str = f"{s.steps_to_target_median:.0f} / {s.steps_to_target_mean:.0f} ± {s.steps_to_target_std:.0f}"
        else:
            steps_str = "-"
        success_str = f"{s.success_rate:.0f}%"
        print(f"{s.name:<30s} | {best_err_str:<25s} | {final_err_str:<25s} | {rmse_str:<25s} | {improvement_best_str:<25s} | {improvement_str:<25s} | {steps_str:<20s} | {success_str:<8s}")

def create_results_dataframe(studies: List[StudyMetrics]) -> pd.DataFrame:
    data = []
    for s in studies:
        row = {
            'Optimizer': s.name,
            'N_runs': s.n_runs,
            'Best_MAE_Median_um': s.best_mae_median * 1e6,
            'Best_MAE_Mean_um': s.best_mae_mean * 1e6,
            'Best_MAE_Std_um': s.best_mae_std * 1e6,
            'Final_MAE_Median_um': s.final_mae_median * 1e6,
            'Final_MAE_Mean_um': s.final_mae_mean * 1e6,
            'Final_MAE_Std_um': s.final_mae_std * 1e6,
            'RMSE_Median_um': s.rmse_median * 1e6,
            'RMSE_Mean_um': s.rmse_mean * 1e6,
            'RMSE_Std_um': s.rmse_std * 1e6,
            'Improvement_Best_Median_pct': s.improvement_best_median,
            'Improvement_Best_Mean_pct': s.improvement_best_mean,
            'Improvement_Best_Std_pct': s.improvement_best_std,
            'Improvement_Final_Median_pct': s.improvement_median,
            'Improvement_Final_Mean_pct': s.improvement_mean,
            'Improvement_Final_Std_pct': s.improvement_std,
            'Steps_to_Target_Median': s.steps_to_target_median,
            'Steps_to_Target_Mean': s.steps_to_target_mean,
            'Steps_to_Target_Std': s.steps_to_target_std,
            'Success_Rate_pct': s.success_rate,
        }
        data.append(row)
    return pd.DataFrame(data)

def load_and_evaluate(
    file_paths: Dict[str, str],
    threshold: float = 4e-5,
) -> Tuple[List[StudyMetrics], Dict[str, List[EpisodeMetrics]]]:
    studies = []
    all_episodes = {}
    
    for name, path in file_paths.items():
        try:
            df = pd.read_csv(path)
            if 'mae' not in df.columns:
                raise ValueError("Missing required column: mae")
            for col in ['mu_x', 'mu_y', 'sigma_x', 'sigma_y']:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            if 'run' not in df.columns:
                df['run'] = 0
            if 'step' not in df.columns:
                df['step'] = df.groupby('run').cumcount()
            if 'best_mae' not in df.columns:
                df['best_mae'] = df.groupby('run')['mae'].cummin()
            study_metrics, episodes = compute_study_metrics(
                df, name, threshold=threshold
            )
            studies.append(study_metrics)
            all_episodes[name] = episodes
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
    return studies, all_episodes

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Evaluate ARES optimization results")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory containing CSV result files")
    parser.add_argument("--threshold", type=float, default=4e-5,
                        help="MAE threshold for target (in meters, default 40µm)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file for results table")
    args = parser.parse_args()
    
    file_paths = {
        "Nelder-Mead": os.path.join(args.data_dir, "NM_mismatched.csv"),
        "BO (zero mean)": os.path.join(args.data_dir, "BO_mismatched.csv"),
        "BO_prior (mismatched)": os.path.join(args.data_dir, "BO_prior_mismatched.csv"),
        "BO_prior (matched)": os.path.join(args.data_dir, "BO_prior_matched_prior_newtask.csv"),
    }
    existing_files = {k: v for k, v in file_paths.items() if os.path.exists(v)}
    
    if len(existing_files) == 0:
        print(f"No data files found in {args.data_dir}")
        exit(1)
    studies, episodes = load_and_evaluate(existing_files, threshold=args.threshold)
    threshold_um = args.threshold * 1e6
    print_table(studies, threshold_um=threshold_um)
    if args.output:
        results_df = create_results_dataframe(studies)
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")