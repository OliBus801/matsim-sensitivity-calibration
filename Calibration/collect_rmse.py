import os
import pandas as pd
import numpy as np
import math
import tqdm
import argparse

def calculate_counts_rmse(sim_path):
    iters_dir = os.path.join(sim_path, 'ITERS')
    it_folders = [d for d in os.listdir(iters_dir) if d.startswith('it.')]
    last_it = max(it_folders, key=lambda x: int(x.split('.')[-1]))
    it_number = last_it.replace('it.', '')
    countscompare_path = os.path.join(iters_dir, last_it, f"{it_number}.countscompare.txt")

    df = pd.read_csv(countscompare_path, sep='\t', engine='python')
    diff = df['MATSIM volumes'] - df['Count volumes']
    mse = np.mean(diff ** 2)
    return math.sqrt(mse)

def collect_best_rmse_per_seed(seed_dir):
    sub_dirs = next(os.walk(seed_dir))[1]
    rmses = []

    for sim_dir in sub_dirs:
        sim_path = os.path.join(seed_dir, sim_dir)
        try:
            rmse = calculate_counts_rmse(sim_path)
            rmses.append(rmse)
        except Exception as e:
            print(f"Erreur dans {sim_dir} ({seed_dir}): {e}")

    if len(rmses) > 0:
        best_rmse = np.nanmin(rmses)
        mean_rmse = np.nanmean(rmses)
        std_rmse = np.nanstd(rmses)
        return best_rmse, mean_rmse, std_rmse
    else:
        return np.nan, np.nan, np.nan

def collect_summary_stats(base_dir):
    method_dirs = next(os.walk(base_dir))[1]
    summary = {}

    for method in tqdm.tqdm(method_dirs, desc="Traitement des méthodes"):
        method_path = os.path.join(base_dir, method)
        seed_dirs = next(os.walk(method_path))[1]
        best_rmses = []

        for seed in seed_dirs:
            seed_path = os.path.join(method_path, seed)
            best_rmse, mean_rmse, std_rmse = collect_best_rmse_per_seed(seed_path)
            best_rmses.append((seed, best_rmse, mean_rmse, std_rmse))

        # Enregistrement détaillé par méthode
        method_df = pd.DataFrame(best_rmses, columns=["Seed", "best_rmse", "mean_rmse", "std_rmse"]).set_index("Seed")
        method_df.loc["best"] = {
            "best_rmse": method_df["best_rmse"].min(),
            "mean_rmse": method_df["best_rmse"].mean(),
            "std_rmse": method_df["best_rmse"].std()
        }
        method_df.to_csv(os.path.join(method_path, f"best_rmse_{method}.csv"))
        print(f"\nStatistiques pour {method} sauvegardées dans : {os.path.join(method_path, f'best_rmse_{method}.csv')}")

        summary[method] = {
            "mean_best_rmse": method_df.loc["best", "mean_rmse"],
            "std_best_rmse": method_df.loc["best", "std_rmse"]
        }

    # Résumé global
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.index.name = "Method"
    output_path = os.path.join(base_dir, "best_rmse_summary.csv")
    summary_df.to_csv(output_path)
    print(f"\nRésumé global sauvegardé dans : {output_path}")

    return summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_directory", type=str, help="Chemin du dossier contenant les méthodes.")
    args = parser.parse_args()

    collect_summary_stats(args.base_directory)
