import os
import pandas as pd
import gzip
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
import tqdm
import math
import csv
import argparse

BASELINE_MODE_SIOUX_FALLS = {
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

def collect_data_from_directory(directory_path):
    """
    Traverse each subdirectory in the given directory path to collect data.

    The metrics include:
    - average_executed_plan_score: Average score of executed plans.
    - average_travel_distance: Average travel distance.
    - average_vc_ratio: Average volume-to-capacity ratio.
    - average_travel_speed: Average travel speed.
    - std_dev_vc_ratio: Standard deviation of volume-to-capacity ratio.

    Args:
        directory_path (str): Path to the main directory.

    Returns:
        dict: A dictionary containing collected metrics for each subdirectory.
    """
    collected_data = {}

    print(f"Walking directory to compute stats: {directory_path}")

    # Traverse through each subdirectory (only one level deep)
    sub_dirs = next(os.walk(directory_path))[1]
    for sub_dir in tqdm.tqdm(sub_dirs, desc="Processing subdirectories"):
        print(f"Processing subdirectory: {sub_dir}")
        sub_dir_path = os.path.join(directory_path, sub_dir)
        
        # Initialize metrics for the current subdirectory
        metrics = {
            "average_executed_plan_score": None,
            "average_trip_distance": None,
            "average_trip_time": None,
            "average_vc_ratio": None,
            "std_dev_vc_ratio": None,
            "modal_share_hhi": None,
            "counts_rmse": None,
            "rmse_mode_stats": None
        }
        # Retrieve average executed plan score
        try:
            metrics["average_executed_plan_score"] = retrieve_average_executed_plan_score(sub_dir_path, prefix=args.prefix)
        except Exception as e:
            print(f"Error retrieving average_executed_plan_score for {sub_dir}: {e}")

        # Retrieve average trip distance and time
        try:
            metrics["average_trip_distance"], metrics["average_trip_time"] = calculate_average_trip_stat(sub_dir_path, prefix=args.prefix)
        except Exception as e:
            print(f"Error retrieving average_trip_distance/average_trip_time for {sub_dir}: {e}")
        # Calculate VC ratio stats
        try:
            vc_stats = calculate_vc_ratio(sub_dir_path, prefix=args.prefix)
            metrics["average_vc_ratio"] = vc_stats['overall_average_vc']
            metrics["std_dev_vc_ratio"] = vc_stats['overall_std_vc']
        except Exception as e:
            print(f"Error calculating VC ratio stats for {sub_dir}: {e}")

        # Calculate modal share HHI
        try:
            metrics["modal_share_hhi"] = calculate_modal_share_hhi(sub_dir_path, prefix=args.prefix)
        except Exception as e:
            print(f"Error calculating modal_share_hhi for {sub_dir}: {e}")

        # Calculate counts RMSE
        try:
            metrics["counts_rmse"] = calculate_counts_rmse(sub_dir_path, prefix=args.prefix)
        except Exception as e:
            print(f"Error calculating counts_rmse for {sub_dir}: {e}")

        # Calculate RMSE mode stats
        try:
            metrics["rmse_mode_stats"] = calculate_rmse_mode_stats(sub_dir_path, BERLIN_MODE_STATS, prefix=args.prefix)
        except Exception as e:
            print(f"Error calculating rmse_mode_stats for {sub_dir}: {e}")

        collected_data[sub_dir] = metrics
    
    # Convert the collected data to a DataFrame and save it to a CSV file
    print("Data collection complete.")
    df = pd.DataFrame.from_dict(collected_data, orient='index')
    df.index.name = 'Experiment Number'

    # Strip "simulation_" from the index and sort by the numeric value
    df.index = df.index.str.replace("simulation_", "", regex=False).astype(int)
    df = df.sort_index()

    output_file = os.path.join(directory_path, 'collected_data.csv')
    df.to_csv(output_file)
    print(f"Collected data saved to {output_file}")
    print("---------------------------------------")

    return df

def collect_data_from_simulation(simulation_path):
    """
    Collect data from a single simulation directory.

    The metrics include:
    - average_executed_plan_score: Average score of executed plans.
    - average_trip_distance: Average trip distance.
    - average_trip_time: Average trip time.
    - average_vc_ratio: Average volume-to-capacity ratio.
    - std_dev_vc_ratio: Standard deviation of volume-to-capacity ratio.
    - modal_share_hhi: Herfindahl–Hirschman Index of modal shares.
    - counts_rmse: RMSE between MATSim volumes and Count volumes.
    - rmse_mode_stats: RMSE for mode statistics.

    Args:
        simulation_path (str): Path to the simulation directory.

    Returns:
        dict: A dictionary containing collected metrics for the simulation.
    """
    print(f"Processing simulation directory: {simulation_path}")
    
    # Initialize metrics for the simulation
    metrics = {
        "average_executed_plan_score": None,
        "average_trip_distance": None,
        "average_trip_time": None,
        "average_vc_ratio": None,
        "std_dev_vc_ratio": None,
        "modal_share_hhi": None,
        "counts_rmse": None,
        "rmse_mode_stats": None
    }
    
    # Retrieve average executed plan score
    try:
        metrics["average_executed_plan_score"] = retrieve_average_executed_plan_score(simulation_path, prefix=args.prefix)
        print("✓ Retrieved average executed plan score")
    except Exception as e:
        print(f"Error retrieving average_executed_plan_score: {e}")

    # Retrieve average trip distance and time
    try:
        metrics["average_trip_distance"], metrics["average_trip_time"] = calculate_average_trip_stat(simulation_path, prefix=args.prefix)
        print("✓ Retrieved average trip distance and time")
    except Exception as e:
        print(f"Error retrieving average_trip_distance/average_trip_time: {e}")
        
    # Calculate VC ratio stats
    try:
        vc_stats = calculate_vc_ratio(simulation_path, prefix=args.prefix)
        metrics["average_vc_ratio"] = vc_stats['overall_average_vc']
        metrics["std_dev_vc_ratio"] = vc_stats['overall_std_vc']
        print("✓ Calculated VC ratio statistics")
    except Exception as e:
        print(f"Error calculating VC ratio stats: {e}")

    # Calculate modal share HHI
    try:
        metrics["modal_share_hhi"] = calculate_modal_share_hhi(simulation_path, prefix=args.prefix)
        print("✓ Calculated modal share HHI")
    except Exception as e:
        print(f"Error calculating modal_share_hhi: {e}")

    # Calculate counts RMSE
    try:
        metrics["counts_rmse"] = calculate_counts_rmse(simulation_path, prefix=args.prefix)
        print("✓ Calculated counts RMSE")
    except Exception as e:
        print(f"Error calculating counts_rmse: {e}")

    # Calculate RMSE mode stats
    try:
        metrics["rmse_mode_stats"] = calculate_rmse_mode_stats(simulation_path, BERLIN_MODE_STATS, prefix=args.prefix)
        print("✓ Calculated RMSE mode statistics")
    except Exception as e:
        print(f"Error calculating rmse_mode_stats: {e}")

    # Create a DataFrame with the single simulation data
    df = pd.DataFrame([metrics])
    df.index = [os.path.basename(simulation_path)]
    df.index.name = 'Simulation'

    # Save to CSV file
    output_file = os.path.join(simulation_path, 'simulation_data.csv')
    df.to_csv(output_file)
    print(f"Simulation data saved to {output_file}")
    print("---------------------------------------")

    return metrics

def retrieve_average_executed_plan_score(sub_dir_path, prefix=None):
    """
    Function to retrieve the average executed plan score from the subdirectory.
    The information is in file scorestats.csv under column 'avg_executed'.
    We automatically retrieve the last value of the column as it represents the last executed plan score.

    Args:
        sub_dir_path (str): Path to the subdirectory.
    Returns:
        float: The average executed plan score.
    """

    # Define the path to the scorestats.csv file
    filename = f"{prefix}.scorestats.csv" if prefix is not None else "scorestats.csv"
    scorestats_file = os.path.join(sub_dir_path, filename)

    # Check if the file exists
    if not os.path.exists(scorestats_file):
        raise FileNotFoundError(f"File {scorestats_file} does not exist.")

    # Read the CSV file
    try:
        df = pd.read_csv(scorestats_file, sep=';')
    except Exception as e:
        raise ValueError(f"Error reading {scorestats_file}: {e}")

    # Check if the 'avg_executed' column exists
    if 'avg_executed' not in df.columns:
        raise ValueError(f"Column 'avg_executed' not found in {scorestats_file}.")

    # Retrieve the last value of the 'avg_executed' column
    if df['avg_executed'].empty:
        raise ValueError(f"No data found in column 'avg_executed' of {scorestats_file}.")

    return df['avg_executed'].iloc[-1]

def calculate_average_trip_stat(sub_dir_path, iteration=None, prefix=None):
    """
    Calculate the average trip distance and average trip speed from the MATSim output_trips.csv.gz file.

    Args:
        sub_dir_path (str): Path to the directory containing the MATSim simulation files.

    Returns:
        dict: A dictionary containing:
            - 'average_trip_distance' (float): Average trip distance (in meters).
            - 'average_trip_speed' (float): Average trip speed (in km/h).
    """
    if iteration:
        # If an iteration is specified, adjust the path to the output_trips file accordingly
        trips_file = os.path.join(sub_dir_path, f'{prefix}.{iteration}.trips.csv.gz' if prefix is not None else f'{iteration}.trips.csv.gz')
    else:
        trips_file = os.path.join(sub_dir_path, f'{prefix}.output_trips.csv.gz' if prefix is not None else 'output_trips.csv.gz')

    # Check if the file exists
    if not os.path.exists(trips_file):
        raise FileNotFoundError(f"File {trips_file} does not exist.")

    # Read the CSV file
    try:
        df = pd.read_csv(trips_file, sep=';', na_values=['', 'null'], dtype={'start_facility_id': str, 'end_facility_id': str})
    except Exception as e:
        raise ValueError(f"Error reading {trips_file}: {e}")

    # Check if required columns exist
    required_columns = ['traveled_distance', 'trav_time']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {trips_file}.")

    # Calculate average trip distance
    if df['traveled_distance'].empty:
        raise ValueError(f"No data found in column 'distance' of {trips_file}.")
    average_trip_distance = df['traveled_distance'].mean()

    # Calculate average trip time in minutes
    if df['trav_time'].empty:
        raise ValueError(f"No data found in column 'trav_time' of {trips_file}.")

    df['trav_time'] = pd.to_timedelta(df['trav_time'])
    average_trip_time = df['trav_time'].dt.total_seconds().mean() / 60  # Convert seconds to minutes

    return average_trip_distance, average_trip_time

def calculate_counts_rmse(sub_dir_path, prefix=None):
    """
    Calculates the RMSE between MATSim volumes and Count volumes
    from the countscompare.txt file.

    Args:
        sub_dir_path (str): Path to the simulation subdirectory.

    Returns:
        float: RMSE value.
    """
    # Go to the ITERS folder and find the last iteration folder
    iters_dir = os.path.join(sub_dir_path, 'ITERS')
    if not os.path.exists(iters_dir):
        raise FileNotFoundError(f"Directory {iters_dir} not found.")
    # Find the iteration folder with the highest number
    it_folders = [d for d in os.listdir(iters_dir) if d.startswith('it.')]
    if not it_folders:
        raise FileNotFoundError(f"No iteration folder found in {iters_dir}.")
    last_it = max(it_folders, key=lambda x: int(x.split('.')[-1]))
    last_it_dir = os.path.join(iters_dir, last_it)

    # Strip 'it.' from last_it to get the iteration number
    it_number = last_it.replace('it.', '')
    countscompare_path = os.path.join(last_it_dir, f"{prefix}.{it_number}.countscompare.txt" if prefix is not None else f"{it_number}.countscompare.txt")

    if not os.path.exists(countscompare_path):
        raise FileNotFoundError(f"File {countscompare_path} not found.")
    
    # Read the countscompare.txt file
    df = pd.read_csv(countscompare_path, sep='\t', engine='python')

    # Calculate RMSE
    diff = df['MATSIM volumes'] - df['Count volumes']
    mse = np.mean(diff ** 2)

    return math.sqrt(mse)

def calculate_rmse_mode_stats(sub_dir_path, baseline, prefix=None):
    """
    Calculate the RMSE for mode statistics for the last iteration in the simulation.

    Args:
        sub_dir_path (str): Path to the simulation subdirectory.

    Returns:
        float: The RMSE value.
    """
    # Define the path to the modestats.csv file
    modestats_file = os.path.join(sub_dir_path, f"{prefix}.modestats.csv" if prefix is not None else "modestats.csv")
    modestats_df = pd.read_csv(modestats_file, sep=';')

    modes = [col for col in modestats_df.columns if col != "iteration"]
    final_modes_stats = modestats_df.iloc[-1]

    # Compute the squared error for each mode, per row
    squared_errors = [(final_modes_stats[mode] - baseline.get(mode, 0)) ** 2 for mode in modes]
    total_mse = np.sum(squared_errors, axis=0)
    total_rmse = np.sqrt(total_mse)


    return total_rmse

def calculate_modal_share_hhi(sub_dir_path, prefix=None):
    """
    Calculates the Herfindahl–Hirschman Index (HHI) of modal shares from the MATSim modestats.csv file.
    The HHI is computed as the sum of squared modal shares from the last available iteration.

    Args:
        sub_dir_path (str): Path to the directory containing the modestats.csv file.

    Returns:
        float: Herfindahl–Hirschman Index (HHI), ranging from 1/n (maximum diversity) to 1 (single dominant mode).
    """
    filename = f"{prefix}.modestats.csv" if prefix is not None else "modestats.csv"
    modal_share_file = os.path.join(sub_dir_path, filename)

    if not os.path.exists(modal_share_file):
        raise FileNotFoundError(f"File {modal_share_file} not found.")

    try:
        df = pd.read_csv(modal_share_file, sep=';')
    except Exception as e:
        raise ValueError(f"Error reading {modal_share_file}: {e}")

    # Get the last row (latest iteration)
    last_row = df.iloc[-1]

    # Drop the 'iteration' column
    modal_shares = last_row.drop('iteration').astype(float)

    # Compute HHI
    hhi = np.sum(modal_shares ** 2)

    return hhi

def parse_events(sub_dir_path, save_to_file=True, time_bin_size=3600, iteration=None, prefix=None):
    """
    Analyzes a MATSim events file to calculate traffic volumes
    on each link for specified time intervals, including periods
    with no traffic (zero volumes).

    Args:
        sub_dir_path (str): Path to the directory containing the events file.
        time_bin_size (float): Size of the time intervals in seconds.

    Returns:
        dict: Dictionary where the keys are link IDs (str), and the values
              are dictionaries with time bins (int) as keys and volumes (int) as values.
    """
    if iteration:
        events_file = os.path.join(sub_dir_path, f'{prefix}.{iteration}.events.xml.gz' if prefix is not None else f'{iteration}.events.xml.gz')
    else:
        events_file = os.path.join(sub_dir_path, f'{prefix}.output_events.xml.gz' if prefix is not None else 'output_events.xml.gz')

    link_volumes = defaultdict(lambda: defaultdict(int))
    max_time = 0.0

    # Single pass to process events and calculate volumes with encoding fallback
    try:
        with gzip.open(events_file, 'rt', encoding='utf-8') as f:
            context = ET.iterparse(f, events=('end',))
            for event, elem in context:
                if elem.tag == 'event' and elem.attrib.get('type') == 'left link':
                    time = float(elem.attrib['time'])
                    link_id = elem.attrib['link']
                    time_bin = int(time // time_bin_size)
                    link_volumes[link_id][time_bin] += 1
                    max_time = max(max_time, time)
                elem.clear()
                # Clear the root to free memory for large files
                if hasattr(elem, 'getparent'):
                    parent = elem.getparent()
                    if parent is not None:
                        parent.remove(elem)
    except UnicodeDecodeError:
        # Fallback: try without explicit encoding
        with gzip.open(events_file, 'rt') as f:
            context = ET.iterparse(f, events=('end',))
            for event, elem in context:
                if elem.tag == 'event' and elem.attrib.get('type') == 'left link':
                    time = float(elem.attrib['time'])
                    link_id = elem.attrib['link']
                    time_bin = int(time // time_bin_size)
                    link_volumes[link_id][time_bin] += 1
                    max_time = max(max_time, time)
                elem.clear()
                # Clear the root to free memory for large files
                if hasattr(elem, 'getparent'):
                    parent = elem.getparent()
                    if parent is not None:
                        parent.remove(elem)

    # Calculate the total number of time bins
    num_bins = int(math.ceil(max_time / time_bin_size))

    # Ensure all links have zero volumes for missing bins
    for link_id, bins in link_volumes.items():
        for bin_index in range(num_bins):
            if bin_index not in bins:
                bins[bin_index] = 0

    if save_to_file:
        # Save the link volumes to a CSV file
        output_file = os.path.join(sub_dir_path, 'link_volumes.csv')
        save_link_volumes_to_csv(link_volumes, link_volumes.keys(), num_bins, output_file)

    return link_volumes

def save_link_volumes_to_csv(link_volumes, link_ids, num_bins, output_file):
    """
    Saves traffic volumes by link and time interval to a CSV file.

    Args:
        link_volumes (dict): Dictionary containing volumes by link and time bin.
        link_ids (list): List of link identifiers to include.
        num_bins (int): Total number of time intervals.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['link_id'] + [f'time_bin_{i}' for i in range(num_bins)]
        writer.writerow(header)

        for link_id in link_ids:
            volumes = link_volumes.get(link_id, {})
            row = [link_id] + [volumes.get(bin_idx, 0) for bin_idx in range(num_bins)]
            writer.writerow(row)

def load_link_capacities(sub_dir_path, iteration=None, prefix=None, scaling_factor=1.0):
    """
    Loads link capacities from either output_links.csv.gz or the fallback linkstats file.

    Args:
        sub_dir_path (str): Path to the ITERS/it.X directory.
        iteration (int, optional): Iteration number. If None, the latest iteration is used.
        prefix (str, optional): File prefix (e.g. 'berlin-v6.0').
        scaling_factor (float): Factor to scale down capacities (e.g., 0.01 for 1%).


    Returns:
        dict: A dictionary {link_id (str): capacity (float)}
    """
    def find_latest_iteration(iters_path):
        # Ex: finds it.0, it.10, it.100 → returns 100
        dirs = [d for d in os.listdir(iters_path) if d.startswith("it.")]
        iterations = [int(d.split(".")[1]) for d in dirs if d.split(".")[1].isdigit()]
        return max(iterations) if iterations else None

    # Déterminer les chemins
    root_dir = os.path.abspath(os.path.join(sub_dir_path, "..", ".."))
    iters_dir = os.path.abspath(os.path.join(root_dir, "ITERS"))

    if iteration is None:
        iteration = find_latest_iteration(iters_dir)
        if iteration is None:
            raise FileNotFoundError(f"Aucune itération trouvée dans {iters_dir}")

    output_links_name = f"{prefix}.output_links.csv.gz" if prefix else "output_links.csv.gz"
    output_links_path = os.path.join(root_dir, output_links_name)

    if os.path.exists(output_links_path):
        # ✅ Lecture depuis output_links.csv.gz
        df = pd.read_csv(output_links_path, sep=";", dtype={"link": str})
        df = df[["link", "capacity"]].dropna()
        df["capacity"] = df["capacity"].astype(float) * scaling_factor
        return df.set_index("link")["capacity"].to_dict()

    else:
        # 🔁 Fallback: lecture depuis linkstats
        stats_name = f"{prefix}.{iteration}.linkstats.txt.gz" if prefix else f"{iteration}.linkstats.txt.gz"
        stats_path = os.path.join(iters_dir, f"it.{iteration}", stats_name)

        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Fichier introuvable : {output_links_path} ni {stats_path}")

        with gzip.open(stats_path, "rt") as f:
            df = pd.read_csv(f, sep="\t", dtype={"LINK": str}, usecols=["LINK", "CAPACITY"])

        df = df.dropna(subset=["CAPACITY"])
        df["CAPACITY"] = df["CAPACITY"].astype(float) * scaling_factor
        return df.set_index("LINK")["CAPACITY"].to_dict()

def calculate_vc_ratio(sub_dir_path, time_bin_size=3600, iteration=None, prefix=None, scaling_factor=1.0):
    link_volumes_file = os.path.join(sub_dir_path, "link_volumes.csv")

    if os.path.exists(link_volumes_file):
        # Parse the existing link_volumes.csv file
        link_volumes = defaultdict(lambda: defaultdict(int))
        with open(link_volumes_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            time_bins = header[1:]  # Skip the 'link_id' column
            for row in reader:
                link_id = row[0]
                for i, volume in enumerate(row[1:]):
                    link_volumes[link_id][int(time_bins[i].split('_')[-1])] = int(volume)
    else:
        # Call parse_events to generate link_volumes
        link_volumes = parse_events(sub_dir_path, time_bin_size, iteration=iteration, prefix=prefix)

    link_capacities = load_link_capacities(sub_dir_path, iteration=iteration, prefix=prefix, scaling_factor=scaling_factor)

    vc_ratios = defaultdict(dict)
    time_bin_totals = defaultdict(list)

    for link_id, bins in link_volumes.items():
        capacity = link_capacities.get(link_id)
        if not capacity or capacity == 0:
            print(f"Warning: Link {link_id} has no capacity or zero capacity.")
            continue
        for time_bin, volume in bins.items():
            vc = volume / capacity
            vc_ratios[link_id][time_bin] = vc
            time_bin_totals[time_bin].append(vc)

    time_bin_stats = {}
    for time_bin, vc_list in time_bin_totals.items():
        avg_vc = np.mean(vc_list)
        std_vc = np.std(vc_list)
        time_bin_stats[time_bin] = {'average_vc': avg_vc, 'std_vc': std_vc}

    all_vc_values = [vc for vc_list in time_bin_totals.values() for vc in vc_list]
    overall_avg = np.mean(all_vc_values)
    overall_std = np.std(all_vc_values)

    return {
        'vc_ratios': vc_ratios,
        'time_bin_stats': time_bin_stats,
        'overall_average_vc': overall_avg,
        'overall_std_vc': overall_std
    }



if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Collect data from MATSim simulation results.")
    parser.add_argument("directory_path", type=str, help="Path to the directory containing simulation results.")
    parser.add_argument("--prefix", type=str, default=None, help="Optional prefix to add to the output file names.")
    parser.add_argument("--single", action="store_true", help="Process a single simulation directory instead of multiple subdirectories.")

    # Parse arguments
    args = parser.parse_args()

    # Call the appropriate function based on the mode
    if args.single:
        collect_data_from_simulation(args.directory_path)
    else:
        collect_data_from_directory(args.directory_path)




