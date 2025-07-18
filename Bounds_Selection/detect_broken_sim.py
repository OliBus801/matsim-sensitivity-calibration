import os
import pandas as pd
import argparse
from tqdm import tqdm
import re

def detect_broken_simulation(df, population_total, timestep=300):
    en_route = df["en-route_all"]
    stuck = df["stuck_all"]

    final_en_route = df.loc[df["time.1"] == 108000, "en-route_all"].iloc[0]
    final_stuck = df.loc[df["time.1"] == 108000, "stuck_all"].iloc[0]
    final = final_en_route + final_stuck
    max_val = (en_route + stuck).max()
    severe_cong_duration = (en_route > 0.5 * population_total).sum() * timestep / 3600
    total_area = en_route.sum()

    is_broken = (
        final > 0.01 * population_total or
        max_val > 0.7 * population_total or
        severe_cong_duration > 2 or
        total_area > 0.01 * population_total * 3600
    )

    return is_broken, {
        "final_en_route+stuck": final,
        "max_en_route+stuck": max_val,
        "severe_cong_duration_h": severe_cong_duration,
        "total_en_route_area": total_area
    }

def find_last_iteration(iters_path, prefix=None):
    subdirs = [d for d in os.listdir(iters_path) if d.startswith("it.")]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda x: int(x.split(".")[1]))
    iter_num = latest.split(".")[1]
    filename = f"{prefix}.{iter_num}.legHistogram.txt" if prefix is not None else f"{iter_num}.legHistogram.txt"
    return os.path.join(iters_path, latest, filename)

def analyze_simulations(base_path, population_total, output_csv="broken_simulations_report.csv", prefix=None):
    sim_dirs = [os.path.join(dp, d) for dp, dn, _ in os.walk(base_path) for d in dn if d == "ITERS"]
    results = []

    for iters_path in tqdm(sim_dirs, desc="Analyse des simulations"):
        root = os.path.dirname(iters_path)
        leghist_path = find_last_iteration(iters_path, prefix=prefix)
        if leghist_path and os.path.exists(leghist_path):
            try:
                df = pd.read_csv(leghist_path, sep="\t")
                if not df.empty:
                    broken, metrics = detect_broken_simulation(df, population_total)
                    match = re.search(r"simulation_(\d+)", root)
                    sim_num = int(match.group(1)) if match else None
                    results.append({
                        "sim_num": sim_num,
                        "is_broken": broken,
                        **metrics
                    })
                else:
                    print(f"[Warning] Fichier vide : {leghist_path}")
            except Exception as e:
                print(f"[Error] Problème avec {leghist_path} : {e}")
        else:
            print(f"[Skip] Fichier non trouvé : {leghist_path}")

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values("sim_num")
        output_path = os.path.join(base_path, output_csv)
        df_results.to_csv(output_path, index=False)
        print(f"[OK] Rapport sauvegardé : {output_path}")
    else:
        print("[Aucun résultat] Aucun fichier analysable n’a été trouvé ou toutes les analyses ont échoué.")

# === USAGE ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection automatique de simulations MATSim brisées.")
    parser.add_argument("base_path", type=str, help="Répertoire de base des simulations")
    parser.add_argument("--population_total", type=int, default=84110, help="Population totale simulée (défaut: 84110)")
    parser.add_argument("--output_csv", type=str, default="broken_simulations_report.csv", help="Nom du fichier CSV de sortie")
    parser.add_argument("--prefix", type=str, default=None, help="Préfixe des noms de simulation pour l'extraction du numéro de simulation")

    args = parser.parse_args()

    analyze_simulations(
        base_path=args.base_path,
        population_total=args.population_total,
        output_csv=args.output_csv,
        prefix=args.prefix
    )
