"""
optuna_ask.py

This script is designed to interact with Optuna, a hyperparameter optimization framework, 
to perform hyperparameter optimization for the MATSim simulation framework. 
It generates new trial suggestions based on a defined search space. The script writes the 
suggested parameters to a CSV file for further use. It supports multiple samplers and 
allows for reproducibility with an optional random seed.

Functions:
    - parse_args: Parses command-line arguments for the script.
    - build_sampler: Constructs an Optuna sampler based on the specified type and seed.
    - suggest_parameters: Suggests trial parameters based on the defined search space.
    - write_trial_to_csv: Appends the suggested trial parameters to a CSV file.
    - main: Orchestrates the process of generating and saving a new trial suggestion.

Usage:
    Run this script from the command line with the required arguments to generate and save 
    a new trial suggestion.

Exemple:
    python optuna_ask.py --journal path/to/journal.log \
        --study-name my_study --csv-out path/to/output.csv --sampler tpe --seed 42
"""

import argparse
import csv
import sys
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

# Make the repository root importable so we can load the search space definition.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from Sensitivity_Analysis.problem_definitions import BERLIN, BERLIN_SOBOL, BERLIN_DEFAULT_VALUES


INT_PARAMETERS = {"mutationRange", "maxAgentPlanMemorySize", "timeStepSize", "numberOfIterations"}
NORMALIZED_PARAMETERS = {"TimeAllocationMutator", "ReRoute", "SubtourModeChoice", "ChangeExpBeta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Optuna for a new trial and write the parameters to CSV."
    )
    parser.add_argument("--journal", required=True, help="Path to the Optuna journal (.log).")
    parser.add_argument("--study-name", required=True, help="Study name inside the journal.")
    parser.add_argument("--csv-out", required=True, help="CSV file to append the suggested trial.")
    parser.add_argument(
        "--sampler",
        default="tpe",
        choices=["tpe", "cmaes", "gp", "random"],
        help="Sampler to use for suggesting the next point.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional random seed for deterministic samplers."
    )
    parser.add_argument(
        "--search_space", default="berlin", choices=["berlin", "berlin_sobol"],
        help="Search space definition to use from problem_definitions."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of trials to ask and append to the CSV in one call (default: 1).",
    )
    return parser.parse_args()


def build_sampler(name: str, seed: int | None) -> optuna.samplers.BaseSampler:
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    if name == "gp":
        try:
            from optuna.samplers import GPSampler
        except Exception as exc:  # pragma: no cover - handled at runtime only
            raise RuntimeError("GPSampler is not available in this Optuna installation. You need to upgrade to Optuna >= 3.6.0") from exc
        return GPSampler(seed=seed)
    return optuna.samplers.RandomSampler(seed=seed)


def suggest_parameters(trial: optuna.trial.Trial, search_space: dict[str, float | int]) -> dict[str, float | int]:
    params: dict[str, float | int] = {}

    for name, bounds in zip(search_space["names"], search_space["bounds"], strict=True):
        low, high = bounds
        if name in INT_PARAMETERS:
            params[name] = trial.suggest_int(name, int(low), int(high))
        else:
            params[name] = trial.suggest_float(name, float(low), float(high), step=0.0001)
    
    return params

def post_process_params(params: dict[str, float | int], default_values: dict[str, float | int]) -> dict[str, float | int]:
    """
    Post-process parameters to ensure they meet the following criteria:
    1. Add all missing parameters with default values.
    2. Integer parameters are rounded to the nearest integer.
    3. All parameters meant to be normalized are normalized to sum to 1.0. 
    """

    # 1. Add missing parameters with default values
    for key, value in default_values.items():
        if key not in params:
            params[key] = value

    # 2. Round integer parameters
    for key in INT_PARAMETERS:
        if key in params:
            params[key] = int(round(params[key]))

    # 3. Normalize specified parameters
    # Check that all normalized parameters are present
    for key in NORMALIZED_PARAMETERS:
        if key not in params:
            raise ValueError(f"Parameter '{key}' required for normalization is missing from the suggested parameters.")

    total = sum(params[key] for key in NORMALIZED_PARAMETERS)
    if total > 0:
        for key in NORMALIZED_PARAMETERS:
            if key in params:
                params[key] = params[key] / total
    else:
        raise ValueError("Cannot normalize parameters: sum of normalized parameters is zero.")

    # 4. Ensure all parameters are rounded to 4 decimal places
    for key in params:
        if isinstance(params[key], float):
            params[key] = round(params[key], 4)

    return params


def write_trial_to_csv(csv_path: Path, params: dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*params.keys()]
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({**params})


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    sampler = build_sampler(args.sampler, args.seed)
    storage = JournalStorage(JournalFileBackend(str(args.journal)))

    if args.search_space == "berlin_sobol":
        space = BERLIN_SOBOL
    else:
        space = BERLIN

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )

    csv_path = Path(args.csv_out)
    for _ in range(args.batch_size):
        trial = study.ask()
        params = suggest_parameters(trial, space)
        params = post_process_params(params, BERLIN_DEFAULT_VALUES)
        write_trial_to_csv(csv_path, params)
        print(f"Suggested trial #{trial.number} written to {csv_path}")


if __name__ == "__main__":
    main()
