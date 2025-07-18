import os
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import gzip
import xml.etree.ElementTree as ET
from collect_data import (
    calculate_average_trip_stat,
    calculate_vc_ratio
)

BASELINE_MODE_STATS = {
    "car": 0.6206039709903698,
    "pt": 0.26440970158126265,
    "walk": 0.11498632742836762,
}

BERLIN_MODE_STATS = {
    "car": 0.2007,
    "ride": 0.0596,
    "pt": 0.2651,
    "bike": 0.1779,
    "walk": 0.2968
}

def collect_iteration_data(sim_dir, prefix, baseline):
    iteration_data_file = os.path.join(sim_dir, "iteration_data.csv")

    # Check if iteration_data.csv exists
    if os.path.exists(iteration_data_file):
        existing_data = pd.read_csv(iteration_data_file)
    else:
        existing_data = pd.DataFrame()

    # Retrieve the average executed plan scores for all iterations
    try:
        avg_scores = retrieve_all_executed_plan_scores(sim_dir, prefix)
    except Exception as e:
        print(f"Error retrieving executed plan scores: {e}")
        avg_scores = pd.Series(dtype=float)

    # Retrieve modestats and calculate the RMSE for mode statistics
    try:
        modes_stats = retrieve_all_mode_stats(sim_dir, prefix)
        rmse_mode_stats = calculate_rmse_mode_stats(modes_stats, baseline)
    except Exception as e:
        print(f"Error retrieving or calculating RMSE mode stats: {e}")
        rmse_mode_stats = pd.Series(dtype=float)

    # Iterate through the ITERS directory to collect data for each iteration
    iters_dir = os.path.join(sim_dir, "ITERS")
    if not os.path.exists(iters_dir):
        raise FileNotFoundError(f"No ITERS directory in {sim_dir}")

    iteration_dirs = sorted(
        [d for d in os.listdir(iters_dir) if d.startswith("it.")],
        key=lambda x: int(x.split(".")[1])
    )

    for it_dir in tqdm(iteration_dirs, desc=f"Processing iterations..."):
        iteration_num = int(it_dir.split(".")[1])

        # Skip if iteration is already processed
        if not existing_data.empty and iteration_num in existing_data["iteration"].values:
            print(f"Iteration {iteration_num} already processed. Skipping.")
            continue

        it_path = os.path.join(iters_dir, it_dir)

        # Calculate average trip distance and time
        try:
            avg_dist, avg_time = calculate_average_trip_stat(it_path, iteration=it_dir.split(".")[1], prefix=prefix)
        except Exception as e:
            print(f"Error calculating average trip stats in {it_path}: {e}")
            avg_dist, avg_time = None, None

        # Calculate VC ratio statistics
        try:
            vc_stats = calculate_vc_ratio(it_path, iteration=it_dir.split(".")[1], prefix=prefix)
            avg_vc = vc_stats.get("overall_average_vc", None)
            std_vc = vc_stats.get("overall_std_vc", None)
        except Exception as e:
            print(f"Error calculating VC ratio in {it_path}: {e}")
            avg_vc, std_vc = None, None

        # Calculate RMSE for counts
        try:
            rmse_counts = calculate_rmse_counts(it_path, iteration=it_dir.split(".")[1], prefix=prefix)
        except Exception as e:
            print(f"Error calculating RMSE counts in {it_path}: {e}")
            rmse_counts = None

        try:
            avg_score = avg_scores.get(iteration_num, None)

            # If the avg_score isn't found, recompute it
            if avg_score is None:
                avg_score = calculate_avg_score(it_path, iteration=it_dir.split(".")[1], prefix=prefix)
        except Exception as e:
            print(f"Error retrieving average score for {iteration_num}: {e}")
            avg_score = None

        # Retrieve the RMSE for the current iteration from the DataFrame
        if iteration_num in rmse_mode_stats.index:
            rmse_mode = rmse_mode_stats.loc[iteration_num, "total_rmse"]
        else:
            # If the rmse_mode isn't found, we'll try and recompute it
            try:
                rmse_mode = calculate_iter_mode_rmse(it_path, iteration=it_dir.split(".")[1], prefix=prefix)
            except Exception as e:
                print(f"Error computing RMSE of mode stats for {iteration_num}: {e}")
                rmse_mode = None

        # Append data for the current iteration
        iteration_data = {
            "iteration": iteration_num,
            "average_executed_plan_score": avg_score,
            "average_trip_distance": avg_dist,
            "average_trip_time": avg_time,
            "average_vc_ratio": avg_vc,
            "std_dev_vc_ratio": std_vc,
            "counts_rmse": rmse_counts,
            "rmse_mode_stats": rmse_mode
        }

        # Write the data to the CSV file
        new_data = pd.DataFrame([iteration_data])
        if not os.path.exists(iteration_data_file):
            new_data.to_csv(iteration_data_file, index=False)
        else:
            new_data.to_csv(iteration_data_file, mode='a', header=False, index=False)

        print(f"Saved data for iteration {iteration_num} to {iteration_data_file}")

    print(f"All iterations processed and saved to {iteration_data_file}")
    return pd.read_csv(iteration_data_file)

def retrieve_all_executed_plan_scores(sub_dir_path, prefix=None):
    """
    Retrieve the average executed plan score for all iterations from scorestats.csv.

    Args:
        sub_dir_path (str): Path to the subdirectory.

    Returns:
        pd.Series: Series with iteration as index and avg_executed as values.
    """
    scorestats_file = os.path.join(sub_dir_path, f"{prefix}.scorestats.csv" if prefix is not None else "scorestats.csv")

    if not os.path.exists(scorestats_file):
        raise FileNotFoundError(f"File {scorestats_file} does not exist.")

    try:
        df = pd.read_csv(scorestats_file, sep=';')
    except Exception as e:
        raise ValueError(f"Error reading {scorestats_file}: {e}")

    if 'iteration' not in df.columns or 'avg_executed' not in df.columns:
        raise ValueError(f"Columns 'iteration' and/or 'avg_executed' not found in {scorestats_file}.")

    # Return a Series: index=iteration, value=avg_executed
    return df.set_index('iteration')['avg_executed']

def retrieve_all_mode_stats(sub_dir_path, prefix=None):
    """
    Retrieve the mode statistics for all iterations from modestats.csv.

    Args:
        sub_dir_path (str): Path to the subdirectory.

    Returns:
        pd.DataFrame: DataFrame with iteration as index and mode statistics as columns.
    """
    modestats_file = os.path.join(sub_dir_path, f"{prefix}.modestats.csv" if prefix is not None else "modestats.csv")

    if not os.path.exists(modestats_file):
        raise FileNotFoundError(f"File {modestats_file} does not exist.")

    try:
        df = pd.read_csv(modestats_file, sep=';')
    except Exception as e:
        raise ValueError(f"Error reading {modestats_file}: {e}")

    if 'iteration' not in df.columns:
        raise ValueError(f"Column 'iteration' not found in {modestats_file}.")

    return df.set_index('iteration')

def calculate_avg_score(it_path, iteration, prefix=None):
    """
    Calculate the average score of selected plans from a plans XML file.

    Args:
        it_path (str): Path to the iteration directory.
        iteration (str): The iteration number as a string.
        prefix (str): Optional prefix for the file name.

    Returns:
        float: The average score of selected plans.
    """
    plans_file = os.path.join(it_path, f"{prefix}.{iteration}.plans.xml.gz" if prefix is not None else f"{iteration}.plans.xml.gz")

    if not os.path.exists(plans_file):
        raise FileNotFoundError(f"Plans file {plans_file} does not exist.")

    total_score = 0.0
    count = 0

    try:
        with gzip.open(plans_file, 'rb') as f:
            # Use iterparse for memory efficiency
            context = ET.iterparse(f, events=("end",))
            for event, elem in context:
                if elem.tag == "plan" and elem.get("selected") == "yes":
                    score = elem.get("score")
                    if score is not None:
                        total_score += float(score)
                        count += 1
                    # Only one selected plan per person, so break after finding it
                    # Clear the parent 'person' element to free memory
                    parent = elem.getparent() if hasattr(elem, "getparent") else None
                    if parent is not None:
                        parent.clear()
                elif elem.tag == "person":
                    elem.clear()  # Free memory for processed person
            # Final cleanup
            elem.clear()
    except Exception as e:
        raise ValueError(f"Error parsing {plans_file}: {e}")

    if count == 0:
        raise ValueError(f"No selected plans found in {plans_file}.")

    return total_score / count

def calculate_iter_mode_rmse(it_path, iteration, prefix=None, baseline=BASELINE_MODE_STATS):
    """
    Calculate the RMSE for mode proportions from the trips CSV file.

    Args:
        it_path (str): Path to the iteration directory.
        iteration (str): The iteration number as a string.
        prefix (str): Optional prefix for the file name.

    Returns:
        float: The RMSE value for mode proportions.
    """
    trips_file = os.path.join(it_path, f"{prefix}.{iteration}.trips.csv.gz" if prefix is not None else f"{iteration}.trips.csv.gz")

    if not os.path.exists(trips_file):
        raise FileNotFoundError(f"Trip file {trips_file} does not exist.")

    try:
        # Read the gzipped CSV file with ';' as the separator
        df = pd.read_csv(trips_file, sep=';', compression='gzip')
    except Exception as e:
        raise ValueError(f"Error reading {trips_file}: {e}")

    if 'main_mode' not in df.columns:
        raise ValueError(f"Column 'main_mode' not found in {trips_file}.")

    # Calculate the proportion of each mode
    mode_counts = df['main_mode'].value_counts(normalize=True)

    # Compute RMSE against the baseline mode stats
    squared_errors = [(mode_counts.get(mode, 0) - baseline.get(mode, 0)) ** 2 for mode in baseline.keys()]
    mse = np.mean(squared_errors)

    return math.sqrt(mse)

def calculate_rmse_counts(iter_dir, iteration, prefix=None):
    """
    Calculate the RMSE for counts in the given iteration directory.

    Args:
        iter_dir (str): Path to the iteration directory.
        iteration (str): The iteration number as a string.

    Returns:
        float: The RMSE value.
    """
    counts_file = os.path.join(iter_dir, f"{prefix}.{iteration}.countscompare.txt" if prefix is not None else f"{iteration}.countscompare.txt")
    
    if not os.path.exists(counts_file):
        raise FileNotFoundError(f"Counts file {counts_file} does not exist.")

    try:
        df = pd.read_csv(counts_file, sep='\t')
    except Exception as e:
        raise ValueError(f"Error reading {counts_file}: {e}")

    if 'MATSIM volumes' not in df.columns or 'Count volumes' not in df.columns:
        raise ValueError(f"Columns 'MATSIM volumes' and/or 'Count volumes' not found in {counts_file}.")
    
    # Calculate RMSE between MATSIM volumes and Count volumes
    diff = df['MATSIM volumes'] - df['Count volumes']
    mse = np.mean(diff ** 2)

    return math.sqrt(mse)

def calculate_rmse_mode_stats(modes_stats, baseline):
    """
    Calculate the RMSE for mode statistics for all iterations in the DataFrame.

    Args:
        modes_stats (pd.DataFrame): DataFrame with mode stats for each iteration.
        baseline (dict): Baseline mode proportions.

    Returns:
        pd.DataFrame: DataFrame with total_rmse for each iteration.
    """
    modes = [col for col in modes_stats.columns if col != "iteration"]
    
    # Compute the squared error for each mode, per row
    squared_errors = [(modes_stats[mode] - baseline.get(mode, 0)) ** 2 for mode in modes]
    total_mse = np.sum(squared_errors, axis=0)
    total_rmse = np.sqrt(total_mse)

    # Construction of the output DataFrame
    result_df = pd.DataFrame({
        "total_rmse": total_rmse
    })
    return result_df

def main(root_dir, prefix=None, baseline=BERLIN_MODE_STATS):
    all_results = []

    for seed_dir in tqdm(sorted(os.listdir(root_dir)), desc="Processing simulations (seeds)"):
        sim_path = os.path.join(root_dir, seed_dir)
        if not os.path.isdir(sim_path):
            print(f"Skipping {sim_path}, not a directory.")
            continue

        try:
            df = collect_iteration_data(sim_path, prefix=prefix, baseline=baseline)
            # Extract the seed number from directory name like "simulation_1"
            try:
                df["seed"] = int(seed_dir.split("_")[-1])
            except Exception:
                df["seed"] = seed_dir  # fallback to directory name if parsing fails

            all_results.append(df)
        except Exception as e:
            print(f"Failed to process {seed_dir}: {e}")

    if not all_results:
        print("No data collected.")
        return

    full_df = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(root_dir, "all_iterations_metrics.csv")
    full_df.to_csv(output_path, index=False)
    print(f"Saved aggregated results to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate MATSim iteration data across multiple seeds.")
    parser.add_argument("root_directory", type=str, help="Path to directory containing subdirectories.")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix for output files.")
    parser.add_argument("--reference_modestats", type=str, default=BASELINE_MODE_STATS, help="Path to reference modestats CSV file.")
    args = parser.parse_args()
    main(args.root_directory, prefix=args.prefix, baseline=args.reference_modestats)


