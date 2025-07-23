import argparse, csv, subprocess, sys, pathlib, os

"""
Lance une expérience MATSim à partir d'un CSV.
Usage :
    python run_experiment.py <id_experience> <fichier_csv> <config.xml>

- <id_experience> : numéro de ligne à lire (1 = première ligne après l’en‑tête)
- <fichier_csv>   : fichier contenant les hyper‑paramètres
- <config.xml>    : chemin du fichier de configuration MATSim
"""

# ******************************************************************
# ********** BESOIN D'AJUSTER LE CHEMIN MATSim MANUELLEMENT ********
# ******************************************************************
MATSIM_HOME = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, "matsim_project")
print(f"📂 MATSim Home : {MATSIM_HOME}")


def build_param_string(row: dict) -> str:
    """Convertit {'col1':'v1', 'col2':'v2'} → 'col1=v1,col2=v2'."""
    return ",".join([f"{k}={v}" for k, v in row.items()])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=int, help="ID d'expérience (1‑N)")
    parser.add_argument("csv_path", type=pathlib.Path, help="Fichier CSV contenant les hyper‑paramètres")
    parser.add_argument("config_path", type=pathlib.Path, help="Chemin du fichier de configuration MATSim")
    parser.add_argument("output_path", type=pathlib.Path, help="Répertoire de sortie pour les résultats")
    parser.add_argument("--firstIteration", type=int, required=False, default=None, help="Itération de début (optionnel)")
    parser.add_argument("--lastIteration", type=int, required=False, default=None, help="Itération de fin (optionnel)")
    parser.add_argument("--blockID", type=int, required=False, default=None, help="ID du block (optionnel)")
    args = parser.parse_args()

    # --- Lecture du CSV ------------------------------------------------------
    with args.csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    idx = args.exp_id - 1   # index Python (0‑based)
    if idx < 0 or idx >= len(rows):
        sys.exit(f"❌ ID {args.exp_id} hors limites (1‑{len(rows)})")

    param_string = build_param_string(rows[idx])

    # --- Commande Java -------------------------------------------------------
    os.chdir(MATSIM_HOME)

    # Extraire le nom du scénario après "scenarios"
    config_path_str = str(args.config_path)
    try:
        scenarios_index = config_path_str.split(os.sep).index("scenarios")
        scenario_name = config_path_str.split(os.sep)[scenarios_index + 1]
        print(f"📂 Scénario détecté automatiquement : {scenario_name}")
    except (ValueError, IndexError):
        sys.exit("❌ Impossible de détecter le sous-dossier après 'scenarios' dans le chemin du fichier de configuration.")

    # Template de switch-case pour scenario_name
    match scenario_name:
        case "siouxfalls-2014":
            print("🔧 Configuration spécifique pour siouxfalls-2014 : RunMatsimHP")
            java_class = "org.matsim.project.RunMatsimHP"
            command = ""
        case "kyoto":
            print("🔧 Configuration spécifique pour kyoto : KyotoScenarioHP")
            java_class = "org.matsim.project.KyotoScenarioHP"
            command = "run"
        case "berlin":
            print("🔧 Configuration spécifique pour berlin : BerlinScenarioHP")
            java_class = "org.matsim.project.BerlinScenarioHP"
            command = "run"
        case _:
            print("⚠ Scénario non reconnu, aucune configuration spécifique appliquée")
            java_class = "org.matsim.project.RunMatsimHP"
            command = ""

    # Vérification de l'existence de command 
    if command != "":
        if args.firstIteration is not None and args.lastIteration is not None and args.blockID is not None:
            print(f"🔧 Itérations personnalisées : {args.firstIteration} à {args.lastIteration} | block ID : {args.blockID}")
            java_cmd = [
                "java",
                "-cp", "matsim-example-project-0.0.1-SNAPSHOT.jar",
                java_class, command,
                param_string,
                str(args.exp_id),                # simulation_number
                str(args.config_path),            # chemin du config.xml
                str(args.output_path),         # répertoire de sortie
                "--firstIteration", str(args.firstIteration),            # itération de début
                "--lastIteration", str(args.lastIteration),               # itération de fin
                "--blockID", str(args.blockID)                             # block ID
            ]
        else :
            java_cmd = [
                "java",
                "-cp", "matsim-example-project-0.0.1-SNAPSHOT.jar",
                java_class, command,
                param_string,
                str(args.exp_id),                # simulation_number
                str(args.config_path),            # chemin du config.xml
                str(args.output_path)         # répertoire de sortie    
            ]
    else:
        java_cmd = [
            "java",
            "-cp", "matsim-example-project-0.0.1-SNAPSHOT.jar",
            java_class,
            param_string,
            str(args.exp_id),                # simulation_number
            str(args.config_path),            # chemin du config.xml
            str(args.output_path)         # répertoire de sortie    
        ]

    print("▶ Lancement :", " ".join(java_cmd))
    completed = subprocess.run(java_cmd, check=False)
    sys.exit(completed.returncode)

if __name__ == "__main__":
    main()