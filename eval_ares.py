import os
import numpy as np
import bo_cheetah_prior_ares
import cheetah
import pandas as pd
import torch
import tqdm
from xopt import VOCS, Evaluator, Xopt
from xopt.generators.bayesian import UpperConfidenceBoundGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.sequential.neldermead import NelderMeadGenerator


def main(args):
    vocs_config = """
        variables:
            q1: [-30, 30]
            q2: [-30, 30]
            cv: [-0.006, 0.006]
            q3: [-30, 30]
            ch: [-0.006, 0.006]
        objectives:
            mae: minimize
    """
    vocs = VOCS.from_yaml(vocs_config)

    # Evaluator
    if args.task == "matched":
        incoming_beam = None
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_problem,
            function_kwargs={"incoming_beam": incoming_beam},
        )
    elif args.task == "mismatched":
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0001),
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
        )        
        misalignment_config = {
            "AREAMQZM1": (0.0000, 0.0002),   
            "AREAMQZM2": (0.0001, -0.0003),  
            "AREAMQZM3": (-0.0001, 0.00015),
        }
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_problem,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "misalignment_config": misalignment_config,
            },
        )
    elif args.task == "matched_prior_newtask":  
        incoming_beam = cheetah.ParameterBeam.from_parameters(
            mu_x=torch.tensor(8.2413e-07),
            mu_px=torch.tensor(5.9885e-08),
            mu_y=torch.tensor(-1.7276e-06),
            mu_py=torch.tensor(-1.1746e-07),
            sigma_x=torch.tensor(0.0002),
            sigma_px=torch.tensor(3.6794e-06),
            sigma_y=torch.tensor(0.0001),
            sigma_py=torch.tensor(3.6941e-06),
            sigma_tau=torch.tensor(8.0116e-06),
            sigma_p=torch.tensor(0.0023),
            energy=torch.tensor(1.0732e+08),
            total_charge=torch.tensor(5.0e-13),
        )
        misalignment_config = {
            "AREAMQZM1": (0.0000, 0.0002),
            "AREAMQZM2": (0.0001, -0.0003),
            "AREAMQZM3": (-0.0001, 0.00015),
        }
        evaluator = Evaluator(
            function=bo_cheetah_prior_ares.ares_problem,
            function_kwargs={
                "incoming_beam": incoming_beam,
                "misalignment_config": misalignment_config,
            },
        )
    else:
        raise ValueError(f"Invalid task: {args.task}")

    df = pd.DataFrame()

    # Run n_trials
    for i in range(args.n_trials):
        print(f"Trial {i+1}/{args.n_trials}")

        # Initialize Generator
        if args.optimizer == "BO":
            generator = UpperConfidenceBoundGenerator(beta=2.0, vocs=vocs)
            
        elif args.optimizer == "BO_prior":
            if args.task == "matched":  
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean()
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )
            elif args.task == "mismatched":
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(
                    incoming_beam=incoming_beam
                )
                prior_mean_module.q1_misalign_x = 0.0
                prior_mean_module.q1_misalign_y = 0.0
                prior_mean_module.q2_misalign_x = 0.0
                prior_mean_module.q2_misalign_y = 0.0
                prior_mean_module.q3_misalign_x = 0.0
                prior_mean_module.q3_misalign_y = 0.0 
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module},
                    trainable_mean_keys=["mae"], # Allows the prior mean to be trained
                     use_low_noise_prior=False,  
                )                
            elif args.task == "matched_prior_newtask":
                incoming_beam = cheetah.ParameterBeam.from_parameters(
                    mu_x=torch.tensor(8.2413e-07),
                    mu_px=torch.tensor(5.9885e-08),
                    mu_y=torch.tensor(-1.7276e-06),
                    mu_py=torch.tensor(-1.1746e-07),
                    sigma_x=torch.tensor(0.0002),
                    sigma_px=torch.tensor(3.6794e-06),
                    sigma_y=torch.tensor(0.0001),
                    sigma_py=torch.tensor(3.6941e-06),
                    sigma_tau=torch.tensor(8.0116e-06),
                    sigma_p=torch.tensor(0.0023),
                    energy=torch.tensor(1.0732e+08),
                    total_charge=torch.tensor(5.0e-13),
                )
                prior_mean_module = bo_cheetah_prior_ares.AresPriorMean(
                    incoming_beam=incoming_beam
                )
                prior_mean_module.q1_misalign_x = 0.0000
                prior_mean_module.q1_misalign_y = 0.0002
                prior_mean_module.q2_misalign_x = 0.0001
                prior_mean_module.q2_misalign_y = -0.0003
                prior_mean_module.q3_misalign_x = -0.0001
                prior_mean_module.q3_misalign_y = 0.00015           
                gp_constructor = StandardModelConstructor(
                    mean_modules={"mae": prior_mean_module}
                )       
            generator = UpperConfidenceBoundGenerator(
                beta=2.0, vocs=vocs, gp_constructor=gp_constructor
            )
        elif args.optimizer == "NM":
            generator = NelderMeadGenerator(vocs=vocs)           
        else:
            raise ValueError(f"Invalid optimizer: {args.optimizer}")

        xopt = Xopt(
            vocs=vocs,
            evaluator=evaluator,
            generator=generator,
            max_evaluations=args.max_evaluation_steps,
        )
        
        # Initial point
        print(f"  Evaluating initial point...")
        initial_point = {
            "q1": 10.0,
            "q2": -10.0,
            "cv": 0.0,
            "q3": 10.0,
            "ch": 0.0,
        }
        xopt.evaluate_data(initial_point)

        # Starting Optimization
        for step_num in tqdm.tqdm(range(args.max_evaluation_steps)):
            xopt.step()

        # Post processing the dataframes
        xopt.data.index.name = "step"
        xopt.data["run"] = i
        xopt.data["best_mae"] = xopt.data["mae"].cummin()

        # Saves learned misalignments to raw csv, if using BO_prior
        if args.optimizer == "BO_prior" and args.task in ["mismatched", "matched_prior_newtask"]:
            try:
                model = xopt.generator.model
                gp_model = model.models[0]
                learned_mean = gp_model.mean_module._model
                xopt.data["q1_misalign_x"] = float(learned_mean.q1_misalign_x.item())
                xopt.data["q1_misalign_y"] = float(learned_mean.q1_misalign_y.item())
                xopt.data["q2_misalign_x"] = float(learned_mean.q2_misalign_x.item())
                xopt.data["q2_misalign_y"] = float(learned_mean.q2_misalign_y.item())
                xopt.data["q3_misalign_x"] = float(learned_mean.q3_misalign_x.item())
                xopt.data["q3_misalign_y"] = float(learned_mean.q3_misalign_y.item())
            except Exception as e:
                print(f"  Warning: Could not save learned misalignments: {e}")
        
        for col in xopt.data.columns:
            xopt.data[col] = xopt.data[col].astype(float)
        df = pd.concat([df, xopt.data])
        
    # Saving results
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_filename = f"{args.output_dir}/{args.optimizer}_{args.task}.csv"

    df.to_csv(out_filename)

if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Run ARES optimization task.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="BO",
        choices=["BO", "BO_prior", "NM"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--n_trials",
        "-n",
        type=int,
        default=10,
        help="Number of trials to run for each optimizer.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="matched",
        choices=["matched", "mismatched", "matched_prior_newtask"],
        help="Task to run. See bo_cheetah_prior_ares.py for options.",
    )
    parser.add_argument(
        "--max_evaluation_steps",
        "-s",
        type=int,
        default=50,
        help="Maximum number of evaluations to run for each trial.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n_workers",
        "-w",
        type=int,
        default=mp.cpu_count() - 1,
        help="Number of workers to use for parallel evaluation.",
    )
    args = parser.parse_args()
    
    # Sets random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    main(args)