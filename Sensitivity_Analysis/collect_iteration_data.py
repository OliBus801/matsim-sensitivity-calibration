import os
import math
import pickle
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

def get_cached_result(cache_file, compute_func, *args, **kwargs):
    """
    Generic caching function to avoid recomputing expensive operations.
    
    Args:
        cache_file (str): Path to the cache file
        compute_func (callable): Function to compute the result if not cached
        *args, **kwargs: Arguments to pass to compute_func
    
    Returns:
        The cached or computed result
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            # If cache is corrupted, recompute
            pass
    
    # Compute the result
    result = compute_func(*args, **kwargs)
    
    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    except:
        # If caching fails, just continue without caching
        pass
    
    return result

def collect_iteration_data(sim_dir, prefix, baseline, iteration_range=None):
    # Cache the expensive computations that are done once
    avg_scores = None
    rmse_mode_stats = None

    # Iterate through the ITERS directory to collect data for each iteration
    iters_dir = os.path.join(sim_dir, "ITERS")
    if not os.path.exists(iters_dir):
        raise FileNotFoundError(f"No ITERS directory in {sim_dir}")

    iteration_dirs = sorted(
        [d for d in os.listdir(iters_dir) if d.startswith("it.")],
        key=lambda x: int(x.split(".")[1])
    )
    
    # Filter iteration directories based on the specified range
    if iteration_range is not None:
        start_iter, end_iter = iteration_range
        iteration_dirs = [
            d for d in iteration_dirs 
            if start_iter <= int(d.split(".")[1]) <= end_iter
        ]
        print(f"Filtering iterations to range [{start_iter}, {end_iter}]: {len(iteration_dirs)} iterations to process")
    
    if not iteration_dirs:
        print("No iterations to process in the specified range.")
        return pd.DataFrame()

    all_data = []

    for it_dir in tqdm(iteration_dirs, desc=f"Processing iterations..."):
        iteration_num = int(it_dir.split(".")[1])
        it_path = os.path.join(iters_dir, it_dir)
        
        # Check if iteration data already exists
        iteration_data_file = os.path.join(it_path, "iteration_metrics.csv")
        if os.path.exists(iteration_data_file):
            print(f"Iteration {iteration_num} already processed (found {iteration_data_file}). Skipping.")
            # Load existing data for final aggregation
            try:
                existing_data = pd.read_csv(iteration_data_file)
                all_data.append(existing_data)
            except:
                print(f"Warning: Could not read existing data from {iteration_data_file}")
            continue

        # Load expensive computations only when needed (lazy loading with caching)
        if avg_scores is None:
            cache_file = os.path.join(sim_dir, ".cache_scorestats.pkl")
            try:
                avg_scores = get_cached_result(cache_file, retrieve_all_executed_plan_scores, sim_dir, prefix)
                # Filter scores to the specified iteration range if needed
                if iteration_range is not None:
                    start_iter, end_iter = iteration_range
                    if isinstance(avg_scores, pd.Series):
                        avg_scores = avg_scores[(avg_scores.index >= start_iter) & (avg_scores.index <= end_iter)]
            except Exception as e:
                print(f"Error retrieving executed plan scores: {e}")
                avg_scores = pd.Series(dtype=float)
        
        if rmse_mode_stats is None:
            cache_file = os.path.join(sim_dir, ".cache_modestats.pkl")
            try:
                def compute_rmse_mode_stats():
                    modes_stats = retrieve_all_mode_stats(sim_dir, prefix)
                    # Filter mode stats to the specified iteration range if needed
                    if iteration_range is not None:
                        start_iter, end_iter = iteration_range
                        modes_stats = modes_stats[(modes_stats.index >= start_iter) & (modes_stats.index <= end_iter)]
                    return calculate_rmse_mode_stats(modes_stats, baseline)
                rmse_mode_stats = get_cached_result(cache_file, compute_rmse_mode_stats)
            except Exception as e:
                print(f"Error retrieving or calculating RMSE mode stats: {e}")
                rmse_mode_stats = pd.Series(dtype=float)
        
        if modal_share_hhi_stats is None:
            cache_file = os.path.join(sim_dir, ".cache_modal_share_hhi.pkl")
            try:
                def compute_modal_share_hhi_stats():
                    modes_stats = retrieve_all_mode_stats(sim_dir, prefix)
                    # Filter mode stats to the specified iteration range if needed
                    if iteration_range is not None:
                        start_iter, end_iter = iteration_range
                        modes_stats = modes_stats[(modes_stats.index >= start_iter) & (modes_stats.index <= end_iter)]
                    return calculate_modal_share_hhi_stats(modes_stats)
                modal_share_hhi_stats = get_cached_result(cache_file, compute_modal_share_hhi_stats)
            except Exception as e:
                print(f"Error retrieving or calculating modal share HHI stats: {e}")
                modal_share_hhi_stats = pd.Series(dtype=float)

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
            avg_score = avg_scores.get(iteration_num, None) if avg_scores is not None else None

            # If the avg_score isn't found, recompute it
            if avg_score is None:
                avg_score = calculate_avg_score(it_path, iteration=it_dir.split(".")[1], prefix=prefix)
        except Exception as e:
            print(f"Error retrieving average score for {iteration_num}: {e}")
            avg_score = None

        # Retrieve the RMSE for the current iteration from the DataFrame
        if rmse_mode_stats is not None and iteration_num in rmse_mode_stats.index:
            rmse_mode = rmse_mode_stats.loc[iteration_num, "total_rmse"]
        else:
            # If the rmse_mode isn't found, we'll try and recompute it
            try:
                rmse_mode = calculate_iter_mode_rmse(it_path, iteration=it_dir.split(".")[1], baseline=baseline, prefix=prefix)
            except Exception as e:
                print(f"Error computing RMSE of mode stats for {iteration_num}: {e}")
                rmse_mode = None
        
        # Retrieve the Modal Share HHI for the current iteration from the Series
        if modal_share_hhi_stats is not None and iteration_num in modal_share_hhi_stats.index:
            modal_share_hhi = modal_share_hhi_stats.loc[iteration_num]
        else:
            modal_share_hhi = None

        # Create data for the current iteration
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

        # Write the data to a CSV file in the iteration directory
        iteration_data_file = os.path.join(it_path, "iteration_metrics.csv")
        iteration_df = pd.DataFrame([iteration_data])
        iteration_df.to_csv(iteration_data_file, index=False)
        print(f"Saved data for iteration {iteration_num} to {iteration_data_file}")
        
        # Add to aggregation list
        all_data.append(iteration_df)

    # Aggregate all data and save to main directory
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        output_file = os.path.join(sim_dir, "iteration_data.csv")
        full_df.to_csv(output_file, index=False)
        print(f"All iterations processed and aggregated data saved to {output_file}")
        return full_df
    else:
        print("No data was processed.")
        return pd.DataFrame()

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
        # Try UTF-8 first
        with gzip.open(plans_file, 'rt', encoding='utf-8') as f:
            context = ET.iterparse(f, events=("end",))
            for event, elem in context:
                if elem.tag == "plan" and elem.get("selected") == "yes":
                    score = elem.get("score")
                    if score is not None:
                        total_score += float(score)
                        count += 1
                # Clear memory efficiently
                elem.clear()
                # Clear parent to free memory for large files
                if hasattr(elem, 'getparent'):
                    parent = elem.getparent()
                    if parent is not None:
                        parent.clear()
    except UnicodeDecodeError:
        # Fallback without encoding
        with gzip.open(plans_file, 'rt') as f:
            context = ET.iterparse(f, events=("end",))
            for event, elem in context:
                if elem.tag == "plan" and elem.get("selected") == "yes":
                    score = elem.get("score")
                    if score is not None:
                        total_score += float(score)
                        count += 1
                # Clear memory efficiently
                elem.clear()
                # Clear parent to free memory for large files
                if hasattr(elem, 'getparent'):
                    parent = elem.getparent()
                    if parent is not None:
                        parent.clear()
    except Exception as e:
        raise ValueError(f"Error parsing {plans_file}: {e}")

    if count == 0:
        raise ValueError(f"No selected plans found in {plans_file}.")

    return total_score / count

def calculate_iter_mode_rmse(it_path, iteration, baseline, prefix=None):
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
        # Use chunking for very large files to reduce memory usage
        chunk_size = 50000  # Adjust based on available memory
        mode_counts = {}
        total_trips = 0
        
        for chunk in pd.read_csv(trips_file, sep=';', compression='gzip', chunksize=chunk_size):
            if 'main_mode' not in chunk.columns:
                raise ValueError(f"Column 'main_mode' not found in {trips_file}.")
            
            # Count modes in this chunk
            chunk_counts = chunk['main_mode'].value_counts()
            total_trips += len(chunk)
            
            # Accumulate counts
            for mode, count in chunk_counts.items():
                mode_counts[mode] = mode_counts.get(mode, 0) + count
        
        # Convert to proportions
        if total_trips > 0:
            mode_proportions = {mode: count / total_trips for mode, count in mode_counts.items()}
        else:
            mode_proportions = {}
            
    except Exception as e:
        raise ValueError(f"Error reading {trips_file}: {e}")

    # Compute RMSE against the baseline mode stats
    squared_errors = [(mode_proportions.get(mode, 0) - baseline.get(mode, 0)) ** 2 for mode in baseline.keys()]
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
    total_mse = np.mean(squared_errors, axis=0)
    total_rmse = np.sqrt(total_mse)

    # Construction of the output DataFrame
    result_df = pd.DataFrame({
        "total_rmse": total_rmse
    })
    return result_df

def calculate_modal_share_hhi_stats(modes_stats):
    """
    Calculate the Modal Share HHI (Herfindahl-Hirschman Index) for mode statistics for all iterations in the DataFrame.

    Args:
        modes_stats (pd.DataFrame): DataFrame with mode stats for each iteration.

    Returns:
        pd.Series: Series with HHI values for each iteration.
    """
    # Compute the HHI for each iteration
    hhi_values = modes_stats.drop(columns=["iteration"]).apply(lambda x: (x ** 2).sum(), axis=1)
    return hhi_values

def load_reference_modestats(reference_modestats):
    """
    Load reference modestats from a CSV file or use a dictionary if provided directly.

    Args:
        reference_modestats (str): Path to CSV file or a dictionary.

    Returns:
        dict: Dictionary of mode proportions.
    """
    if isinstance(reference_modestats, str) and os.path.isfile(reference_modestats):
        try:
            df = pd.read_csv(reference_modestats)
            # Expect columns: mode, proportion
            if "mode" in df.columns and "proportion" in df.columns:
                return dict(zip(df["mode"], df["proportion"]))
            # Or: first column is mode, second is value
            elif df.shape[1] >= 2:
                return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            else:
                raise ValueError("Reference modestats CSV must have at least two columns (mode, proportion).")
        except Exception as e:
            raise ValueError(f"Error reading reference modestats from {reference_modestats}: {e}")
    raise ValueError("reference_modestats must be a dict or a path to a CSV file.")

def main(root_dir, baseline, prefix=None, iteration_range=None):
    
    # Load the reference mode stats
    if baseline is None:
        baseline = BASELINE_MODE_STATS # TODO: Should not accept default mode stats values 
    else:
        baseline = load_reference_modestats(baseline)
    
    all_results = []

    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"{root_dir} is not a valid directory.")
    
    try:
        df = collect_iteration_data(root_dir, prefix=prefix, baseline=baseline, iteration_range=iteration_range)
        try:
            df["seed"] = int(seed_dir.split("_")[-1])
        except Exception:
            df["seed"] = seed_dir  # fallback to directory name if parsing fails
    except Exception as e:
        print(f"Failed to process {root_dir}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate MATSim iteration data across multiple seeds.")
    parser.add_argument("root_directory", type=str, help="Path to directory containing subdirectories.")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix for output files.")
    parser.add_argument("--reference_modestats", type=str, default=None, help="Path to reference modestats CSV file.")
    parser.add_argument("--iteration-range", type=int, nargs=2, metavar=('START', 'END'), 
                        help="Specify iteration range to parse (e.g., --iteration-range 200 300)")
    args = parser.parse_args()
    
    iteration_range = None
    if args.iteration_range:
        start, end = args.iteration_range
        if start > end:
            raise ValueError(f"Start iteration ({start}) must be <= end iteration ({end})")
        iteration_range = (start, end)
        print(f"Processing iterations in range [{start}, {end}]")
    
    main(args.root_directory, baseline=args.reference_modestats, prefix=args.prefix, iteration_range=iteration_range)

