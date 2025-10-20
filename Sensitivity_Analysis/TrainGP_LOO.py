import pandas as pd
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score
import time
import os
import argparse


def load_and_prepare_data(x_path, y_path):
    """
    Loads data from CSV files, applies standard scaling, 
    and returns normalized tensors, column names, and scalers.
    
    Args:
        x_path (str): Path to the input features CSV file.
        y_path (str): Path to the target outputs CSV file.
    
    Returns:
        tuple: (X_norm, Y_norm, y_columns, x_scaler, y_scaler)
    """
    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)
    
    if "Experiment Number" in Y.columns:
        Y = Y.drop(columns=["Experiment Number"])
    
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(Y)
    
    X_norm = torch.from_numpy(x_scaler.transform(X)).float()
    Y_norm = torch.from_numpy(y_scaler.transform(Y)).float()
    
    print(f"Finished loading and normalizing data | X shape: {X_norm.shape}, Y shape: {Y_norm.shape}")
    return X_norm, Y_norm, Y.columns.tolist(), x_scaler, y_scaler


class GPModel(ExactGP):
    """
    Gaussian Process Model for regression using ExactGP.

    Args:
        train_x (Tensor): Training input data.
        train_y (Tensor): Training target values.
        likelihood (Likelihood): Gaussian process likelihood module.

    Attributes:
        mean_module (ConstantMean): Mean function of the GP.
        covar_module (ScaleKernel): Covariance function (kernel) of the GP.

    Methods:
        forward(x):
            Computes the multivariate normal distribution for input x.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def train_gp_model(train_x, train_y, lr=0.1, n_iter=50):
    """
    Trains an exact Gaussian Process (GP) model using Maximum Likelihood Loss (MLL).

    Args:
        train_x (torch.Tensor): Training input data.
        train_y (torch.Tensor): Training target values.
        lr (float): Learning rate for the optimizer.
        n_iter (int): Number of training iterations.

    Returns:
        tuple: Trained GP model and its likelihood.
    """
    likelihood = GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    
    return model, likelihood


def loo_cv(train_x, train_y, lr=0.1, n_iter=50):
    """
    Leave-One-Out Cross-Validation.
    
    Args:
        train_x (torch.Tensor): Input features.
        train_y (torch.Tensor): Target values.
        lr (float): Learning rate for GP training.
        n_iter (int): Number of training iterations.
    
    Returns:
        tuple: (rmse, r2) evaluation metrics.
    """
    preds, trues = [], []
    n = train_x.size(0)
    start_time = time.time()
    
    for i in range(n):
        mask = torch.ones(n, dtype=torch.bool)
        mask[i] = False
        x_tr, y_tr = train_x[mask], train_y[mask]
        x_val, y_val = train_x[~mask].unsqueeze(0), train_y[~mask]
        
        model, lik = train_gp_model(x_tr, y_tr, lr=lr, n_iter=n_iter)
        model.eval()
        lik.eval()
        
        with torch.no_grad():
            post = lik(model(x_val))
            preds.append(post.mean.item())
            trues.append(y_val.item())

    elapsed = time.time() - start_time
    print(f"  LOO-CV completed in {elapsed:.2f} seconds.")
    
    rmse = root_mean_squared_error(trues, preds)
    r2 = r2_score(trues, preds)
    
    return rmse, r2


def run_loo_experiments(X, Y, y_columns, seed, lr, n_iter, output_path):
    """
    Runs Leave-One-Out Cross-Validation experiments on the entire dataset.

    Args:
        X (torch.Tensor): Feature matrix (normalized).
        Y (torch.Tensor): Target matrix (normalized).
        y_columns (list of str): Names of target variables (columns in Y).
        seed (int): Random seed for reproducibility.
        lr (float): Learning rate for GP training.
        n_iter (int): Number of training iterations for GP.
        output_path (str): Path where results CSV will be saved.

    Returns:
        pandas.DataFrame: DataFrame containing RMSE and R2 scores for each target.
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Si le fichier existe déjà, on supprime pour éviter les doublons
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file: {output_path}")
    
    # Initialize results CSV
    pd.DataFrame(columns=["target", "n_samples", "seed", "n_iters", "lr", "RMSE", "R2"]) \
        .to_csv(output_path, index=False)

    n_samples = X.size(0)
    print(f"\nRunning LOO-CV on full dataset with {n_samples} samples")
    print(f"Configuration: seed={seed}, lr={lr}, n_iter={n_iter}\n")

    for i, name in enumerate(y_columns):
        print(f"[{i+1}/{len(y_columns)}] Running LOO-CV for target: {name}")
        y_vec = Y[:, i]
        rmse, r2 = loo_cv(X, y_vec, lr=lr, n_iter=n_iter)
        
        pd.DataFrame([{
            "target": name,
            "n_samples": n_samples,
            "seed": seed,
            "n_iters": n_iter,
            "lr": lr,
            "RMSE": rmse,
            "R2": r2
        }]).to_csv(output_path, mode='a', header=False, index=False)
        
        print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f}\n")
    
    return pd.read_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate GP model performance with Leave-One-Out Cross-Validation."
    )
    parser.add_argument('--x_path', type=str, required=True, 
                        help='Path to the input features CSV file.')
    parser.add_argument('--y_path', type=str, required=True, 
                        help='Path to the target outputs CSV file.')
    parser.add_argument('--output_path', type=str, default='gp_results.csv',
                        help='Path where the results CSV will be saved (default: gp_results.csv).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for GP training (default: 0.1).')
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Number of training iterations for GP (default: 100).')
    
    args = parser.parse_args()

    print("=" * 80)
    print("GP Model LOO-CV Evaluation")
    print("=" * 80)
    print(f"Input features:  {args.x_path}")
    print(f"Target outputs:  {args.y_path}")
    print(f"Output results:  {args.output_path}")
    print(f"Random seed:     {args.seed}")
    print(f"Learning rate:   {args.lr}")
    print(f"Iterations:      {args.n_iter}")
    print("=" * 80)

    # Load and prepare data
    X, Y, y_cols, x_scaler, y_scaler = load_and_prepare_data(args.x_path, args.y_path)
    
    # Run LOO-CV experiments
    results_df = run_loo_experiments(
        X=X, 
        Y=Y, 
        y_columns=y_cols,
        seed=args.seed,
        lr=args.lr,
        n_iter=args.n_iter,
        output_path=args.output_path
    )
    
    results_path = os.path.abspath(args.output_path)
    print("=" * 80)
    print(f"✓ Results saved to {results_path}")
    print("=" * 80)
    print("\nSummary of results:")
    print(results_df.to_string(index=False))
