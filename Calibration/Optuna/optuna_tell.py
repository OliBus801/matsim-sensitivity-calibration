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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of trials/rows to report in one call (default: 1). Uses the last N rows.",
    )
    return parser.parse_args()


def read_metrics_from_csv(
    results_path: Path, metric: str, simulation_id: str | None, batch_size: int
) -> list[tuple[float, str]]:
    with results_path.open() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise ValueError(f"Metric column '{metric}' not found in {results_path}")
        rows = list(reader)

    if simulation_id is not None:
        rows = [r for r in rows if r.get("Simulation") == simulation_id]

    if not rows:
        raise ValueError(f"No matching row found in {results_path}")

    batch = rows[-batch_size:]
    values: list[tuple[float, str]] = []
    for selected_row in batch:
        try:
            value = float(selected_row[metric])
        except Exception as exc:  # pragma: no cover - runtime validation only
            raise ValueError(
                f"Cannot convert metric '{metric}' to float from row {selected_row}"
            ) from exc
        label = selected_row.get("Simulation", "<unknown>")
        values.append((value, label))
    return values


def pick_trial_numbers(study: optuna.Study, provided: int | None, batch_size: int) -> list[int]:
    if provided is not None:
        return list(range(provided, provided + batch_size))
    running_trials = study.get_trials(states=(TrialState.RUNNING,))
    if len(running_trials) < batch_size:
        raise RuntimeError("No RUNNING trial found; specify --trial-number explicitly.")
    running_trials = sorted(running_trials, key=lambda t: t.number)
    return [t.number for t in running_trials[:batch_size]]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    metrics = read_metrics_from_csv(
        results_path, args.metric, args.simulation_id, args.batch_size
    )

    storage = JournalStorage(JournalFileBackend(str(args.journal)))
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    trial_numbers = pick_trial_numbers(study, args.trial_number, args.batch_size)
    if len(trial_numbers) != len(metrics):
        raise RuntimeError(
            f"Batch size mismatch: {len(trial_numbers)} trials vs {len(metrics)} metrics."
        )

    for trial_number, (value, sim_label) in zip(trial_numbers, metrics, strict=True):
        study.tell(trial_number, value)
        print(
            f"Reported metric '{args.metric}'={value} from simulation '{sim_label}' "
            f"to trial #{trial_number}."
        )


if __name__ == "__main__":
    main()
