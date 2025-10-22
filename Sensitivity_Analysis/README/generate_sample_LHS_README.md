# generate_sample_LHS.py - Documentation

## Description
This script generates Latin Hypercube Sampling (LHS) samples for sensitivity analysis on the Multi-Agent Transport Simulator (MATSim). It automatically normalizes replanning strategy columns and adds default values for parameters not included in the sampling space.

## Usage

### Quick Configuration
Simply modify the 6 parameters in the configuration block:

```python
if __name__ == "__main__":
    # 1. Choose the scenario
    problem_scenario = problem_definitions.BERLIN_CONSTRAINED
    
    # 2. Default values
    default_values = problem_definitions.BERLIN_DEFAULT_VALUES
    
    # 3. Samples per variable
    N_per_var = 50
    
    # 4. Output path
    out_csv = "cache/LHS/lhs_samples_berlin_constrained.csv"
    
    # 5. Random seed
    random_seed = 42
    
    # 6. Integer columns (optional)
    integer_columns = ['maxAgentPlanMemorySize', 'numberOfIterations', 'timeStepSize']
```

### Run the Script
```bash
python generate_sample_LHS.py
```

## Configuration Parameters

### 1. `problem_scenario`
Problem definition to use. Design your own or see available options in `problem_definitions.py`:
- `problem_definitions.BERLIN` - Full Berlin problem
- `problem_definitions.BERLIN_CONSTRAINED` - Berlin problem after filtering with plausibility checks
- `problem_definitions.SIOUXFALLS` - Full Sioux Falls problem
- `problem_definitions.SIOUXFALLS_CONSTRAINED` - Sioux Falls problem after filtering with plausibility checks

### 2. `default_values`
Dictionary of default values for columns not included in the problem, but that you still want included. The script checks each column and adds it only if it doesn't already exist. See examples in `problem_definitions.py`

### 3. `N_per_var`
Number of samples per variable. The total number of samples will be:
```
n_samples = N_per_var × num_vars
```

**Example:**
```python
N_per_var = 50  # Generates 50 × 15 = 750 samples for BERLIN_CONSTRAINED
```

### 4. `out_csv`
Output CSV file path. The directory will be created automatically if it doesn't exist.

### 5. `random_seed`
Random seed for reproducibility. Use `None` for a random seed.

### 6. `integer_columns`
List of columns to round to integers.

**Example:**
```python
integer_columns = ['maxAgentPlanMemorySize', 'numberOfIterations', 'timeStepSize']
```

## Configuration Examples

### Berlin Constrained
```python
problem_scenario = problem_definitions.BERLIN_CONSTRAINED
default_values = problem_definitions.BERLIN_DEFAULT_VALUES
N_per_var = 50  # 50 × 15 = 750 samples
out_csv = "cache/LHS/lhs_samples_berlin_constrained.csv"
random_seed = 42
integer_columns = ['maxAgentPlanMemorySize', 'numberOfIterations', 'timeStepSize']
```

## Utility Functions

### `generate_lhs_samples()`
Main function that generates LHS samples.

**Parameters:**
- `problem`: Problem definition
- `n_samples`: Number of samples
- `default_values`: Default values
- `out_csv`: Output path
- `seed`: Random seed (optional)
- `normalize_replanning`: Normalize replanning columns (default: True)
- `round_decimals`: Number of decimals (default: 5)
- `integer_columns`: Columns to round to integers (optional)

### `normalize_replanning_columns()`
Normalizes replanning strategy columns so their sum equals 1.

**Affected columns:**
- TimeAllocationMutator
- ReRoute
- SubtourModeChoice
- ChangeExpBeta

### `add_default_values()`
Adds default values for columns not present in the DataFrame.

**Behavior:**
- Checks if the column already exists
- Adds the column only if it doesn't exist
- Prints a message for each added column

### `round_integer_columns()`
Rounds specified columns to integers.

## Output Format

The generated CSV file contains all problem columns plus the added default columns.

**Example structure for BERLIN_CONSTRAINED:**
```
ASC_car,ASC_bike,ASC_ride,ASC_freight,ASC_pt,waitingPt_util,...,ASC_walk,maxAgentPlanMemorySize,timeStepSize,numberOfIterations
0.544,-0.231,1.234,-0.789,2.345,-8.765,...,0.0,5,1,200
...
```

## Important Notes

1. **Replanning Normalization**: Replanning columns are automatically normalized so their sum equals 1
2. **Directory Creation**: The script automatically creates necessary directories
3. **Default Values**: Default values are added ONLY if columns don't already exist
4. **Seed**: Use the same seed to generate identical samples
5. **Sample Count**: Total number is always `N_per_var × num_vars`
