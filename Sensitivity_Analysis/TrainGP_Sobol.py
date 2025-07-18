import pandas as pd
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from sklearn.preprocessing import StandardScaler
import time
import argparse


def load_and_prepare_data(data_path):
    """
    Loads data from a CSV file, applies standard scaling, 
    and returns normalized tensor and scaler.
    """
    X = pd.read_csv(data_path)
    x_scaler = StandardScaler().fit(X)
    X_scaled = torch.tensor(x_scaler.transform(X), dtype=torch.float)
    print(f"Finished loading and normalizing data | X shape : {X_scaled.shape}")
    return X_scaled, x_scaler


class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))

def train_gp(X_train, y_train, lr=0.1, n_iter=50):
    likelihood = GaussianLikelihood()
    model = GPModel(X_train, y_train, likelihood)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    return model.eval(), likelihood.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Gaussian Process model for Sobol sensitivity analysis using specified input and output data paths.")
    parser.add_argument('--x_path', type=str, required=True, help='Path to the input features CSV file.')
    parser.add_argument('--y_path', type=str, required=True, help='Path to the target outputs CSV file.')
    parser.add_argument('--sobol_path', type=str, required=True, help='Path to the Sobol samples CSV file.')
    args = parser.parse_args()

    x_path = args.x_path
    sobol_path = args.sobol_path

    # Normalize and load data from CSV files
    X_train_scaled, x_scaler = load_and_prepare_data(x_path)
    X_sobol_scaled, _ = load_and_prepare_data(sobol_path)

    Y_train = pd.read_csv(args.y_path)
    Y_train = Y_train.drop(columns=["Experiment Number"]) if "Experiment Number" in Y_train.columns else Y_train

    # Dictionnary to hold all predictions
    all_predictions = {}

    for col in Y_train.columns:
        print(f"Training GP for output column: {col}")
        start_time = time.time()
        
        y_scaler = StandardScaler().fit(Y_train[[col]])
        y_train_scaled = torch.tensor(y_scaler.transform(Y_train[[col]]), dtype=torch.float).squeeze()

        model, likelihood = train_gp(X_train_scaled, y_train_scaled)

        with torch.no_grad():
            preds = model(X_sobol_scaled)
            means = preds.mean.cpu().numpy()
            unscaled = y_scaler.inverse_transform(means.reshape(-1, 1)).flatten()
            all_predictions[col] = unscaled

        elapsed_time = time.time() - start_time
        print(f"Elapsed time for {col}: {elapsed_time:.2f} seconds")


    df_preds = pd.DataFrame(all_predictions)
    df_preds.to_csv("sobol_predictions.csv", index=False)