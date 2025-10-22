"""
Generate Latin Hypercube Sampling (LHS) samples for sensitivity analysis.
See generate_sample_LHS_README.md for configuration details.
"""

from SALib.sample import latin
import problem_definitions
import pandas as pd
import os


def normalize_replanning_columns(df):
    """Normalize replanning strategy columns to sum to 1 for each row."""
    replanning_columns = [
        'TimeAllocationMutator',
        'ReRoute', 
        'SubtourModeChoice',
        'ChangeExpBeta'
    ]
    
    # Check which replanning columns exist in the dataframe
    existing_columns = [col for col in replanning_columns if col in df.columns]
    
    if not existing_columns:
        return df
    
    # Calculate the sum of the columns for each row
    row_sums = df[existing_columns].sum(axis=1)
    # Avoid division by zero
    row_sums = row_sums.replace(0, 1)
    # Divide each column by the corresponding row sum
    df[existing_columns] = df[existing_columns].div(row_sums, axis=0)

    # Round values to 2 decimals
    df[existing_columns] = df[existing_columns].round(2)

    return df


def add_default_values(df, default_values):
    """Add default values for columns not present in the DataFrame."""
    for col_name, default_value in default_values.items():
        if col_name not in df.columns:
            df[col_name] = default_value
            print(f"  Added column '{col_name}' with default value: {default_value}")
    return df


def round_integer_columns(df, int_columns):
    """Round specified columns to integers."""
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].round(0).astype(int)
    return df


def lhs_samples(problem, n_samples, default_values, out_csv, seed=None,
                         normalize_replanning=True, round_decimals=5, 
                         integer_columns=None):
    """Generate Latin Hypercube Sampling (LHS) samples for a given problem."""
    print(f"\nGenerating {n_samples} LHS samples...")
    print(f"Problem has {problem['num_vars']} variables")
    
    # Generate LHS samples
    param_values = latin.sample(problem, n_samples, seed=seed)

    # Convert to DataFrame for easier manipulation
    df_samples = pd.DataFrame(param_values, columns=problem["names"])

    # Round values to specified decimals
    df_samples = df_samples.round(round_decimals)
    print(f"  Rounded values to {round_decimals} decimals")

    # Round integer columns if specified
    if integer_columns:
        df_samples = round_integer_columns(df_samples, integer_columns)
        print(f"  Rounded integer columns: {integer_columns}")

    # Normalize replanning strategy columns
    if normalize_replanning:
        df_samples = normalize_replanning_columns(df_samples)
        print(f"  Normalized replanning strategy columns")

    # Add default values for columns not in the problem definition
    df_samples = add_default_values(df_samples, default_values)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Save DataFrame to CSV
    df_samples.to_csv(out_csv, index=False)
    print(f"\n✓ Saved {n_samples} samples to: {out_csv}")
    
    return df_samples

def generate_lhs_samples(
    problem_scenario,
    default_values,
    N_per_var,
    out_csv,
    random_seed=42,
    integer_columns=None,
    normalize_replanning=True,
    round_decimals=5
):
        # Calculate total number of samples
        n_samples = N_per_var * problem_scenario["num_vars"]

        print("=" * 80)
        print("LHS Sample Generation Configuration")
        print("=" * 80)

        # Find scenario name
        scenario_name = "Unknown"
        for k, v in vars(problem_definitions).items():
            if v is problem_scenario:
                scenario_name = k
                break

        print(f"Problem scenario: {scenario_name}")
        print(f"Number of variables: {problem_scenario['num_vars']}")
        print(f"Samples per variable: {N_per_var}")
        print(f"Total samples: {n_samples}")
        print(f"Random seed: {random_seed}")
        print(f"Output file: {out_csv}")
        print(f"Default values: {default_values}")
        print("=" * 80)

        # Generate samples
        df_samples = lhs_samples(
            problem=problem_scenario,
            n_samples=n_samples,
            default_values=default_values,
            out_csv=out_csv,
            seed=random_seed,
            normalize_replanning=normalize_replanning,
            round_decimals=round_decimals,
            integer_columns=integer_columns
        )

        print("\n" + "=" * 80)
        print("Sample generation completed successfully!")
        print("=" * 80)
        print(f"\nDataFrame shape: {df_samples.shape}")
        print(f"Columns: {list(df_samples.columns)}")
        print(f"\nFirst 5 rows:")
        print(df_samples.head())
        return df_samples




if __name__ == "__main__":
    # ========================================================================
    # ------------------------ EXAMPLE USAGE ---------------------------------
    # ========================================================================
    
    # 1. Choose the problem scenario
    problem_scenario = problem_definitions.BERLIN_CONSTRAINED
    
    # 2. Set default values
    default_values = problem_definitions.BERLIN_DEFAULT_VALUES
    
    # 3. Number of samples per variable (will be multiplied by num_vars)
    N_per_var = 50  # Generates: 50 * num_vars samples
    
    # 4. Output path for the CSV file
    out_csv = "cache/LHS/lhs_samples_berlin_constrained.csv"
    
    # 5. Random seed for reproducibility
    random_seed = 42  # Set to None for a random seed
    
    # 6. Columns to round to integer (optional)
    integer_columns = ['maxAgentPlanMemorySize', 'numberOfIterations', 'timeStepSize']
    
    # Example usage: Call main() with the current configuration
    generate_lhs_samples(
        problem_scenario=problem_scenario,
        default_values=default_values,
        N_per_var=N_per_var,
        out_csv=out_csv,
        random_seed=random_seed,
        integer_columns=integer_columns,
        normalize_replanning=True,
        round_decimals=5
    )