import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === Paramètres de configuration ===
scenarios = {
    "Sioux Falls": {
        "samples": "Bounds_Selection/cache/Siouxfalls/lhs_samples_800.csv",
        "flags": "Bounds_Selection/cache/Siouxfalls/broken_sims_LHS_800.csv"
    },
    "Berlin": {
        "samples": "Bounds_Selection/cache/Berlin/lhs_samples_95.csv",
        "flags": "Bounds_Selection/cache/Berlin/broken_sims_LHS_original_95.csv"
    }
}

# === Fonction d’entraînement et extraction des importances ===
def compute_feature_importances(samples_path, flags_path):
    df_sims = pd.read_csv(samples_path)
    df_broken = pd.read_csv(flags_path)
    df_sims["is_broken"] = df_broken["is_broken"].astype(int)

    X = df_sims.drop(columns=["is_broken"])
    y = df_sims["is_broken"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    return X.columns, importances

# === Création du graphe comparatif ===
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# Seuils pour les couleurs
high_thr = 0.08
medium_thr = 0.04

for ax, (name, paths) in zip(axes, scenarios.items()):
    feature_names, importances = compute_feature_importances(paths["samples"], paths["flags"])
    sorted_idx = importances.argsort()[::-1]
    
    # Génération des couleurs
    colors = []
    for imp in importances[sorted_idx]:
        if imp >= high_thr:
            colors.append("#d73027")  # high
        elif imp >= medium_thr:
            colors.append("#fee08b")  # medium
        else:
            colors.append("#4575b4")  # low

    sns.barplot(
        x=feature_names[sorted_idx],
        y=importances[sorted_idx],
        palette=colors,
        ax=ax
    )
    
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xticklabels(feature_names[sorted_idx], rotation=45, ha="right", fontsize=6, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Importance" if ax == axes[0] else "")
    ax.set_ylim(0.00, 0.42)
    
    # Ajout des valeurs numériques sur les barres
    for idx, imp in enumerate(importances[sorted_idx]):
        ax.text(
            idx, imp,
            f"{imp:.3f}",
            ha="center", va="bottom",
            fontsize=6, color="black", fontweight="normal", rotation=0,
            clip_on=True
        )

plt.tight_layout()
plt.savefig("feature_importances_comparison.pdf")
plt.show()
