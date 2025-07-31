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
    """
    # Charge et normalise X et Y (standard scaling)
    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)
    if "Experiment Number" in Y.columns:
        Y = Y.drop(columns=["Experiment Number"])
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(Y)
    X_norm = torch.from_numpy(x_scaler.transform(X)).float()
    Y_norm = torch.from_numpy(y_scaler.transform(Y)).float()
    print(f"Finished loading and normalizing data | X shape : {X_norm.shape}, Y shape : {Y_norm.shape}")
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
        lr (float, optional): Learning rate for the optimizer. Default is 0.1.
        n_iter (int, optional): Number of training iterations. Default is 50.

    Returns:
        tuple: Trained GP model and its likelihood.
    """
    likelihood = GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.train(); likelihood.train()
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
    Returns (rmse, r2).
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
        model.eval(); lik.eval()
        with torch.no_grad():
            post = lik(model(x_val))
            preds.append(post.mean.item())
            trues.append(y_val.item())

    elapsed = time.time() - start_time
    print(f"LOO-CV completed in {elapsed:.2f} seconds.")
    rmse = root_mean_squared_error(trues, preds)
    r2   = r2_score(trues, preds)
    return rmse, r2

def run_loo_experiments(X, Y, y_columns,
                    seeds=[42, 24, 44],
                    sizes=[450],
                    output_path="gp_results.csv"):
    """
    Runs experiments by performing leave-one-out cross-validation (LOO-CV) on subsets of the data for different target variables, sample sizes, and random seeds.

    Args:
        X (torch.Tensor): Feature matrix.
        Y (torch.Tensor): Target matrix.
        y_columns (list of str): Names of target variables (columns in Y).
        seeds (list of int, optional): Random seeds for reproducibility. Defaults to [42, 43, 44].
        sizes (list of int, optional): List of sample sizes to use for experiments. Defaults to [160, 320, 640].

    Returns:
        pandas.DataFrame: DataFrame containing RMSE and R2 scores for each target, sample size, and seed.
    """
    # Si le fichier existe déjà, on le supprime pour repartir à zéro
    if os.path.exists(output_path):
        os.remove(output_path)

    # Initialisation du CSV (écriture de l’en-tête)
    pd.DataFrame(columns=["target","n_samples","seed","RMSE","R2"]) \
      .to_csv(output_path, index=False)

    for size in sizes:

        print(f"Running experiments for sample size: {size}")

        for seed in seeds:

            print(f"Using seed: {seed}")

            # Set random seed for reproducibility
            torch.manual_seed(seed)
            idx = torch.randperm(X.size(0))[:size]
            X_sub = X[idx]
            Y_sub = Y[idx]
            for i, name in enumerate(y_columns):
                print(f"Running LOO-CV for target: {name}")
                y_vec = Y_sub[:, i]
                rmse, r2 = loo_cv(X_sub, y_vec)
                pd.DataFrame([{
                    "target": name,
                    "n_samples": size,
                    "seed": seed,
                    "RMSE": rmse,
                    "R2": r2
                }]).to_csv(output_path, mode='a', header=False, index=False)
    return pd.read_csv(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the training of a GP model with Leave-One-Out Validation. Inputs and Outputs with specified data paths.")
    parser.add_argument('--x_path', type=str, required=True, help='Path to the input features CSV file.')
    parser.add_argument('--y_path', type=str, required=True, help='Path to the target outputs CSV file.')
    args = parser.parse_args()

    x_path = args.x_path
    y_path = args.y_path
    X, Y, y_cols, x_scaler, y_scaler = load_and_prepare_data(x_path, y_path)
    results_df = run_loo_experiments(X, Y, y_cols)
    results_df.to_csv("gp_results.csv", index=False)
    results_path = os.path.abspath("gp_results.csv")
    print(f"Results saved to {results_path}")