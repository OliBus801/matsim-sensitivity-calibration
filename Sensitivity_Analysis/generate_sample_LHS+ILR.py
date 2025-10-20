# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import problem_definitions
from SALib.sample import latin
from skbio.stats.composition import ilr_inv

# ----------------------------- utils -----------------------------

REPLANNING_COLS = [
    'TimeAllocationMutator',
    'ReRoute',
    'SubtourModeChoice',
    'ChangeExpBeta'
]

def closure(x, axis=-1, eps=0.0):
    """Fermeture (normalise pour sommer à 1)."""
    x = np.asarray(x, dtype=float)
    if eps > 0:
        x = np.clip(x, eps, None)
    s = np.sum(x, axis=axis, keepdims=True)
    s = np.where(s == 0.0, 1.0, s)
    return x / s

# ----------------------------- core ------------------------------

def build_ilr_problem(problem_scenario, z_bounds=[-5, 5]):
    """Remplace les 4 colonnes de replanning par 3 coords ILR avec bornes [-5,5]."""
    names = list(problem_scenario["names"])
    bounds = np.array(problem_scenario["bounds"], dtype=float)

    # indices replanning
    idx_r = [names.index(c) for c in REPLANNING_COLS]
    idx_other = [i for i in range(len(names)) if i not in idx_r]

    other_names = [names[i] for i in idx_other]
    other_bounds = bounds[idx_other, :]

    ilr_names = ["ILR1", "ILR2", "ILR3"]
    ilr_bounds = np.array([z_bounds, z_bounds, z_bounds], dtype=float)

    problem_ilr = {
        "num_vars": len(other_names) + 3,
        "names": other_names + ilr_names,
        "bounds": np.vstack([other_bounds, ilr_bounds])
    }
    return problem_ilr, other_names, ilr_names, idx_other, idx_r

def lhs_in_box(problem_box, n_samples, seed=None):
    return latin.sample(problem_box, n_samples, seed=seed)

def ilr_to_parts_matrix(Z):
    """Vectorise ilr_inv sur un batch (Z shape: [n,3]) → parts shape: [n,4]."""
    # skbio.ilr_inv attend un vecteur 1D; on boucle proprement
    parts = np.vstack([ilr_inv(z) for z in Z])
    # sécurité numérique
    return closure(parts, eps=1e-15)

def apply_ceb_clip_and_close(P, ceb_min=0.01):
    """Clip CEB (colonne 3) puis fermeture pour rester sur le simplex."""
    P = np.asarray(P, dtype=float)
    P[:, 3] = np.maximum(P[:, 3], ceb_min)
    P = closure(P, eps=1e-15)
    return P

def generate_lhs_ilr_csvs(problem_scenario,
                          default_values={},
                          N_per_var=50,
                          out_ilr_csv="ilr_features.csv",
                          out_sim_csv="simulator_params.csv",
                          z_bounds=[-5, 5],
                          ceb_min=0.01,
                          seed=123):
    """
    Génère N_per_var * problem_scenario['num_vars'] échantillons LHS dans l'espace
    (autres paramètres) × (ILR1..ILR3), puis inverse ILR → (TAM, ReRoute, SMC, CEB),
    applique un clip(Ceb, 0.01) + fermeture, et écrit deux CSV.
    """
    rng = np.random.default_rng(seed)
    names = list(problem_scenario["names"])
    bounds = np.array(problem_scenario["bounds"], dtype=float)
    n_total = N_per_var * problem_scenario["num_vars"]

    # Construire le problème ILR
    problem_ilr, other_names, ilr_names, idx_other, idx_r = build_ilr_problem(
        problem_scenario, z_bounds=z_bounds
    )

    # LHS (autres + ILR)
    X = lhs_in_box(problem_ilr, n_total, seed=seed)
    X_other = X[:, :len(other_names)]
    Z = X[:, len(other_names):]  # (n,3) → ILR

    # ILR inverse → parts (TAM, ReRoute, SMC, CEB)
    P = ilr_to_parts_matrix(Z)
    P = apply_ceb_clip_and_close(P, ceb_min=ceb_min)

    # --- CSV 1: ILR features (pour GPR) ---
    ilr_df = pd.DataFrame(np.hstack([X_other, Z]), columns=other_names + ilr_names)

    # For default values in DEFAULT_VALUES, check if column exists, else add it with default values
    for col, default_val in default_values.items():
        if col not in ilr_df.columns:
            ilr_df[col] = default_val

    # Arrondir les valeurs à 6 décimales
    ilr_df = ilr_df.round(6)

    # Arrondir à l'entier les colonnes "maxAgentPlanMemorySize" et "numberOfIterations"
    ilr_df["maxAgentPlanMemorySize"] = ilr_df["maxAgentPlanMemorySize"].round(0).astype(int)
    ilr_df["numberOfIterations"] = ilr_df["numberOfIterations"].round(0).astype(int)

    ilr_df.to_csv(out_ilr_csv, index=False)

    # --- CSV 2: paramètres pour le simulateur (ordre original) ---
    sim_df = pd.DataFrame(np.zeros((n_total, len(names))), columns=names)
    # remplir les autres colonnes
    sim_df[other_names] = X_other
    # remplir les 4 colonnes replanning dans leur ordre d'origine
    for k, col in enumerate(REPLANNING_COLS):
        sim_df[col] = P[:, k]

    # For default values in DEFAULT_VALUES, check if column exists, else add it with default values
    for col, default_val in default_values.items():
        if col not in sim_df.columns:
            sim_df[col] = default_val

    # Arrondir les valeurs à 6 décimales
    sim_df = sim_df.round(6)

    # Arrondir à l'entier les colonnes "maxAgentPlanMemorySize" et "numberOfIterations"
    sim_df["maxAgentPlanMemorySize"] = sim_df["maxAgentPlanMemorySize"].round(0).astype(int)
    sim_df["numberOfIterations"] = sim_df["numberOfIterations"].round(0).astype(int)

    sim_df.to_csv(out_sim_csv, index=False)

    return ilr_df, sim_df

# ----------------------------- exemple ---------------------------

if __name__ == "__main__":

    # N_per_var = 50 ⇒ n_samples = 50 * 16 = 800
    ilr_df, sim_df = generate_lhs_ilr_csvs(
        problem_scenario=problem_definitions.BERLIN_CONSTRAINED,
        default_values=problem_definitions.BERLIN_DEFAULT_VALUES,
        N_per_var=50,
        out_ilr_csv="Sensitivity_Analysis/cache/GP_Training/Berlin/constrained_space/ilr_features.csv",
        out_sim_csv="Sensitivity_Analysis/cache/GP_Training/Berlin/constrained_space/simulator_params.csv",
        z_bounds=[-5, 5],   # simple et robuste
        ceb_min=0.01,
        seed=42
    )

    print("ILR head:")
    print(ilr_df.head())
    print("\nSimulator replanning head:")
    print(sim_df[REPLANNING_COLS].head())