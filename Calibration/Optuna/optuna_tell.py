"""
optuna_tell.py

Read simulator evaluation results from a CSV file and report the chosen metric back to
the Optuna study so the journal records the completed trial.

Usage:
    python optuna_tell.py --journal path/to/journal.log --study-name my_study \
        --results Calibration/Optuna/.cache/simulation_outputs.csv \
        --metric counts_rmse --trial-number 3
"""

import argparse
import csv
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report simulator results back to an Optuna study."
    )
    parser.add_argument("--journal", required=True, help="Path to the Optuna journal (.log).")
    parser.add_argument("--study-name", required=True, help="Study name inside the journal.")
    parser.add_argument("--results", required=True, help="CSV file with simulation outputs.")
    parser.add_argument(
        "--metric",
        default="counts_rmse",
        help="Column name in the results CSV to use as the objective value (default: counts_rmse).",
    )
    parser.add_argument(
        "--simulation-id",
        help="Optional value from the 'Simulation' column to select a specific row.",
    )
    parser.add_argument(
        "--trial-number",
        type=int,
        help="Optional trial number to mark as complete. Defaults to the latest RUNNING trial.",
    )
    return parser.parse_args()


def read_metric_from_csv(
    results_path: Path, metric: str, simulation_id: str | None
) -> tuple[float, str]:
    with results_path.open() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(f"Metric column '{metric}' not found in {results_path}")
        selected_row: dict[str, str] | None = None
        for row in reader:
            if simulation_id is None:
                selected_row = row
            elif row.get("Simulation") == simulation_id:
                selected_row = row
                break
        if selected_row is None:
            raise ValueError(f"No matching row found in {results_path}")
        try:
            value = float(selected_row[metric])
        except Exception as exc:  # pragma: no cover - runtime validation only
            raise ValueError(
                f"Cannot convert metric '{metric}' to float from row {selected_row}"
            ) from exc
        label = selected_row.get("Simulation", "<unknown>")
        return value, label


def pick_trial_number(study: optuna.Study, provided: int | None) -> int:
    if provided is not None:
        return provided
    running_trials = study.get_trials(states=(TrialState.RUNNING,))
    if not running_trials:
        raise RuntimeError("No RUNNING trial found; specify --trial-number explicitly.")
    return max(t.number for t in running_trials)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    value, sim_label = read_metric_from_csv(results_path, args.metric, args.simulation_id)

    storage = JournalStorage(JournalFileBackend(str(args.journal)))
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    trial_number = pick_trial_number(study, args.trial_number)
    study.tell(trial_number, value)

    print(
        f"Reported metric '{args.metric}'={value} from simulation '{sim_label}' "
        f"to trial #{trial_number}."
    )


if __name__ == "__main__":
    main()
