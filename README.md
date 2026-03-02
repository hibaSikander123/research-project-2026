# Physics informed Bayesian Optimisation for ARES Beam Tuning

This project implements Bayesian Optimisation(BO) with a physics informed prior for tuning the transverse beam parameters on a simulation of the ARES Experimental Area at DESY. The core idea is that instead of letting the Gaussian Process(GP) start from an uninformed (zero-mean) prior, a differentiable simulation library ([Cheetah](https://github.com/desy-ml/cheetah)) is embedded directly into the GP mean function, allowing the optimiser 
to leverage physics knowledge from the start and learn real world discrepancies online during optimisation. Quadrupole magnet misalignments are dealt under this project's usecase as the real world discrepancy. The BO implementation is used from the Xopt package. For detailed project description and analysis, feel free to refer the accompanying project report `research-project-report.pdf` available at the root of the repository.


## What this does

The ARES experimental area has three quadrupole magnets (Q1, Q2 and Q3) and two steering/corrector magnets (CH, CV). We want to find magnet settings that minimize the beam error (MAE of beam position and size) at a downstream screen.

## Features
- Physics informed GP prior mean with trainable misalignment parameters.
- Three optimisers compared: standard BO, BO with physics prior and Nelder-Mead.
- Online misalignment learning via constrained hyperparameter optimisation.
- Automated evaluation pipeline with metrics CSV table and publication-quality plots.
- Three experimental scenarios are tested: matched, mismatched and matched_prior_newtask. 

## Folder Structure

- `bo_cheetah_prior_ares.py` Cheetah simulation wrapper and `ARESPriorMean` GP mean module
- `eval_ares.py` Main optimisation script (BO/BO+prior/Nelder-Mead via Xopt)
- `eval_ares_metrics.py` Aggregated evaluation metrics (Best MAE, Final MAE, RMSE, Improvement% (over Best MAE), Improvement% (over Final MAE), steps to target and success rate)
- `plot_ares_results.py` Convergence curves plot, box plots and scatter plot.
- `ARESlatticeStage3v1_9.json` ARES lattice in json format used to load Ares Experimental Area subsection.
- `run_evaluation.py` End-to-end evaluation, metrics table (CSV) and plots (PDFs) 
- `data/` stores the Xopt run results (`pd.Dataframe`) as CSV files.
- `results/` stores all the results generated in the form of plots(PDFs) and aggregated metrics result table (CSV file)
- `research-project-report.pdf` Complete project report
- `project-report-latex/` contains the LaTeX code of the project report generated

## Installataion

```bash
git clone https://github.com/hibaSikander123/research-project-2026.git
cd research-project-2026

python -m venv venv
venv\Scripts\activate    # on Windows
pip install -r requirements.txt
```
Note that `ocelot` is installed from GitHub, so `git` must be available.

## Run Optimisation
```bash
# Nelder-Mead
python eval_ares.py --optimizer NM --task mimatched -n 10 -s 100

# Standard BO
python eval_ares.py --optimizer BO --task mimatched -n 10 -s 100

# Standard BO with physics prior (mismatched)
python eval_ares.py --optimizer BO_prior --task mismatched -n 10 -s 100

# Standard BO with physics prior (matched_prior_newtask)
python eval_ares.py --optimizer BO_prior --task matched_prior_newtask -n 10 -s 100
```

- --optimizer  :  `BO`, `BO_prior`, `NM`
- -- task / -t :  `matched`, `mismatched`, `matched_prior_newtask`
- --n_trials / -n : by default it is 10, any number of trials can be set
- --max_evaluation_steps / -s : by default it is 50, any number of steps can be set
- --output_dir / -o : by default `data/` is set, data folder can be renamed. 

## Run Evaluation
```bash
# Full pipeline (metrics + plots)
python run_evaluation.py 
```

## Dependencies
- cheetah
- xopt
- torch/botorch/gpytorch
- matplotlib/seaborn/SciencePlots

See `requirements.txt` for the full list.
