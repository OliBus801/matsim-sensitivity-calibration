import os
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
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

    # We start by checking if iteration_data.csv already exists
    iteration_data_file = os.path.join(sim_dir, "iteration_data.csv")
    if os.path.exists(iteration_data_file):
        print(f"File {iteration_data_file} already exists. Skipping data collection.")
        return pd.read_csv(iteration_data_file)


    
# We start by retrieving the average executed plan scores for all iterations
    avg_scores = retrieve_all_executed_plan_scores(sim_dir, prefix)

    # Then we retrieve modestats and calculate the RMSE for mode statistics
    modes_stats = retrieve_all_mode_stats(sim_dir, prefix)
    rmse_mode_stats = calculate_rmse_mode_stats(modes_stats, baseline)

    # Then we need to iterate through the ITERS directory to collect data for each iteration
    iters_dir = os.path.join(sim_dir, "ITERS")
    if not os.path.exists(iters_dir):
        raise FileNotFoundError(f"No ITERS directory in {sim_dir}")
    
    iteration_dirs = sorted(
        [d for d in os.listdir(iters_dir) if d.startswith("it.")],
        key=lambda x: int(x.split(".")[1])
    )

    all_data = []

    for it_dir in tqdm(iteration_dirs, desc=f"Processing iterations..."):
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

        iteration_num = int(it_dir.split(".")[1])

        try:
            avg_score = avg_scores.get(iteration_num, None)
        except Exception as e:
            print(f"Error retrieving average score for {iteration_num}: {e}")

        # Retrieve the RMSE for the current iteration from the DataFrame
        if iteration_num in rmse_mode_stats.index:
            rmse_mode = rmse_mode_stats.loc[iteration_num, "total_rmse"]
        else:
            rmse_mode = None

        all_data.append({
            "iteration": iteration_num,
            "average_executed_plan_score": avg_score,
            "average_trip_distance": avg_dist,
            "average_trip_time": avg_time,
            "average_vc_ratio": avg_vc,
            "std_dev_vc_ratio": std_vc,
            "counts_rmse": rmse_counts,
            "rmse_mode_stats": rmse_mode
        })
    
    # Convert the list of dictionaries to a DataFrame
    all_data_df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    output_file = os.path.join(sim_dir, "iteration_data.csv")
    all_data_df.to_csv(output_file, index=False)
    print(f"Saved iteration data to {output_file}")

    return all_data_df

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
    Calculate the RMSE for mode statistics for all iteration in the DataFrame.

    Args:
        modes_stats (Pandas DataFrame): A pandas DataFrame with the mode stats for each given modes.

    Returns:
        float: The RMSE value.
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
    parser.add_argument("--reference_modestats", type=str, default=BERLIN_MODE_STATS, help="Path to reference modestats CSV file.")
    args = parser.parse_args()
    main(args.root_directory, prefix=args.prefix, baseline=args.reference_modestats)


