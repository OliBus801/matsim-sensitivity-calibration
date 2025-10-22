# TrainGP_LOO.py - Documentation

## Description
This script evaluates the performance of a Gaussian Process Regressor model using Leave-One-Out Cross-Validation (LOO-CV) on the entire dataset.

## Main Changes
- **Simplified**: Removed nested loops for different sample sizes and seeds
- **Optimized**: All configurable parameters are now CLI arguments
- **Full dataset**: Uses all X and Y data (no more subsampling)
- **Flexible**: Full configuration via ArgumentParser

## Usage

### Basic usage (with default values)
```bash
python TrainGP_LOO.py \
    --x_path path/to/features.csv \
    --y_path path/to/targets.csv
```

### Full usage with all arguments
```bash
python TrainGP_LOO.py \
    --x_path path/to/features.csv \
    --y_path path/to/targets.csv \
    --output_path results/my_experiment.csv \
    --seed 42 \
    --lr 0.1 \
    --n_iter 50
```

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--x_path` | str | ✓ | - | Path to the features (X) CSV file |
| `--y_path` | str | ✓ | - | Path to the targets (Y) CSV file |
| `--output_path` | str | ✗ | `gp_results.csv` | Path to save the results |
| `--seed` | int | ✗ | `42` | Random seed for reproducibility |
| `--lr` | float | ✗ | `0.1` | Learning rate for GP training |
| `--n_iter` | int | ✗ | `50` | Number of training iterations |

## Usage Examples

### Evaluation with different seeds
```bash
for seed in 22 33 44 55 66; do
    python TrainGP_LOO.py \
        --x_path data/X.csv \
        --y_path data/Y.csv \
        --output_path results/results_seed_${seed}.csv \
        --seed $seed
done
```

### Evaluation with different hyperparameters
```bash
# Test with different learning rates
python TrainGP_LOO.py --x_path X.csv --y_path Y.csv --lr 0.01 --output_path results_lr001.csv
python TrainGP_LOO.py --x_path X.csv --y_path Y.csv --lr 0.1 --output_path results_lr01.csv
python TrainGP_LOO.py --x_path X.csv --y_path Y.csv --lr 0.5 --output_path results_lr05.csv

# Test with different numbers of iterations
python TrainGP_LOO.py --x_path X.csv --y_path Y.csv --n_iter 25 --output_path results_iter25.csv
python TrainGP_LOO.py --x_path X.csv --y_path Y.csv --n_iter 50 --output_path results_iter50.csv
python TrainGP_LOO.py --x_path X.csv --y_path Y.csv --n_iter 100 --output_path results_iter100.csv
```

## Output Format

The results CSV file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `target` | str | Name of the target variable |
| `n_samples` | int | Number of samples used |
| `seed` | int | Seed used for the experiment |
| `RMSE` | float | Root Mean Squared Error |
| `R2` | float | R² coefficient of determination |

## Notes
- The script automatically normalizes the data (StandardScaler)
- The "Experiment Number" column in Y is automatically removed if present
- The output file is overwritten if it already exists
- LOO-CV can be slow for large datasets (N iterations × N samples)
