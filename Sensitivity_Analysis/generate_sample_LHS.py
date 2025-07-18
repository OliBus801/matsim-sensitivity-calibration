from SALib.sample import latin
import pandas as pd

# Fonction pour normaliser les colonnes de stratégie de replanning
def normalize_replanning_columns(df):
    replanning_columns = [
        'TimeAllocationMutator',
        'ReRoute', 
        'SubtourModeChoice',
        'ChangeExpBeta'
    ]
    # Calculer la somme des colonnes pour chaque ligne
    row_sums = df[replanning_columns].sum(axis=1)
    # Éviter la division par zéro
    row_sums = row_sums.replace(0, 1)
    # Diviser chaque colonne par la somme de la ligne correspondante
    df[replanning_columns] = df[replanning_columns].div(row_sums, axis=0)

    # Arrondir les valeurs à 2 décimales
    df[replanning_columns] = df[replanning_columns].round(2)

    return df


# Définir le problème avec des bornes similaire à Zhuge et al. (2019)
problem_siouxfalls = {
    "num_vars": 16,
    "names": [
        'ASC_car',
        'ASC_pt',
        'ASC_walk',
        'waitingPt_util',
        'lateArrival_util',
        'earlyDeparture_util',
        'performing_util',
        'money_util',
        'TimeAllocationMutator',
        'ReRoute', 
        'SubtourModeChoice',
        'ChangeExpBeta', 
        'mutationRange',
        'maxAgentPlanMemorySize', 
        'timeStepSize',
        'numberOfIterations'
        ],
    "bounds": [
        [-1, 3],        #'ASC_car',
        [-1, 3],        #'ASC_pt',
        [-1, 3],        #'ASC_walk',
        [-12, 0],       #'waitingPt_util',
        [-36, 0],       #'lateArrival_util',
        [-12, 0],       #'earlyDeparture_util',
        [0, 12],        #'performing_util',
        [0.25, 2.0],    #'money_util',
        [0, 1],         #'TimeAllocationMutator',
        [0, 1],         #'ReRoute', 
        [0, 1],         #'SubtourModeChoice',
        [0.01, 1],         #'ChangeExpBeta', 
        [900, 3600],    #'mutationRange',
        [1, 10],        #'maxAgentPlanMemorySize', 
        [1, 120],       #'timeStepSize',
        [1, 100],       #'numberOfIterations'
                ]
}

# Définir le problème avec des variables restreintes
# On se concentre uniquement sur les six variables avec le plus de feature importances 
# Bornes justifiées par le partial dependence plot
problem_siouxfalls_final = {
    "num_vars": 12,
    "names": [
        'ASC_car',
        'ASC_pt',
        'waitingPt_util',
        'lateArrival_util',
        'earlyDeparture_util',
        'performing_util',
        'money_util',
        'TimeAllocationMutator',
        'ReRoute', 
        'SubtourModeChoice',
        'ChangeExpBeta', 
        'mutationRange',
        ],
    "bounds": [
        [-1, 0.444],    #'ASC_car',
        [-1, 3],        #'ASC_pt',
        [-12, 0],       #'waitingPt_util',
        [-36, 0],       #'lateArrival_util',
        [-12, 0],       #'earlyDeparture_util',
        [0, 5],        #'performing_util',
        [1.6, 2.0],    #'money_util',
        [0, 1],         #'TimeAllocationMutator',
        [0, 1],         #'ReRoute', 
        [0, 1],         #'SubtourModeChoice',
        [0.01, 1],         #'ChangeExpBeta', 
        [900, 3600],    #'mutationRange',
                ]
}

problem_berlin = {
    "num_vars": 19,
    "names": [
        'ASC_car',
        'ASC_bike',
        'ASC_ride',
        'ASC_freight',
        'ASC_pt',
        'ASC_walk',
        'waitingPt_util',
        'lateArrival_util',
        'earlyDeparture_util',
        'performing_util',
        'money_util',
        'TimeAllocationMutator',
        'ReRoute', 
        'SubtourModeChoice',
        'ChangeExpBeta',
        'mutationRange', 
        'maxAgentPlanMemorySize', 
        'timeStepSize',
        'numberOfIterations'
        ],
    "bounds": [
        [-1, 3],        #'ASC_car',
        [-1, 3],        #'ASC_bike',
        [-1, 3],        #'ASC_ride',
        [-1, 3],        #'ASC_freight',
        [-1, 3],        #'ASC_pt',
        [-1, 3],        #'ASC_walk',
        [-12, 0],       #'waitingPt_util',
        [-36, 0],       #'lateArrival_util',
        [-12, 0],       #'earlyDeparture_util',
        [0, 12],        #'performing_util',
        [0.25, 2.0],    #'money_util',
        [0, 1],         #'TimeAllocationMutator',
        [0, 1],         #'ReRoute',
        [0, 1],         #'SubtourModeChoice',
        [0, 1],         #'ChangeExpBeta',
        [900, 3600],    #'mutationRange',
        [1, 10],        #'maxAgentPlanMemorySize',
        [1, 120],       #'timeStepSize',
        [1, 100]        #'numberOfIterations'
                ]
}

# On multiplie chaque élément de N par num_vars individuellement
N = [5]  # nombre d'échantillons souhaité
n_samples = [n * problem_berlin["num_vars"] for n in N]

problem = problem_berlin  # Choisir le problème à utiliser

# Générer les échantillons LHS
for n_sample in n_samples:
    print(f"Generating {n_sample} samples...")
    param_values = latin.sample(problem, n_sample)

    # Sauvegarder les échantillons dans un fichier CSV
    df_samples = pd.DataFrame(param_values, columns=problem["names"])

    # Arrondir les valeurs à 5 décimales
    df_samples = df_samples.round(5)

    # Arrondir à l'entier les colonnes "maxAgentPlanMemorySize" et "numberOfIterations"
    df_samples["maxAgentPlanMemorySize"] = df_samples["maxAgentPlanMemorySize"].round(0).astype(int)
    df_samples["numberOfIterations"] = df_samples["numberOfIterations"].round(0).astype(int)

    # Normaliser les colonnes de stratégie de replanning
    normalize_replanning_columns(df_samples)

    # Ajouter une colonne 'numberOfIterations' avec des valeurs égales à 50
    #df_samples['numberOfIterations'] = 50

    # Ajouter une colonne 'maxAgentPlanMemorySize' avec des valeurs égales à 5
    #df_samples['maxAgentPlanMemorySize'] = 5
    
    # Ajouter une colonne 'ASC_walk' avec des valeurs nulles
    #df_samples['ASC_walk'] = 0.0

    # Ajouter une colonne 'timeStepSize' avec des valeurs égales à 1
    #df_samples['timeStepSize'] = 1

    # Réorganiser les colonnes pour correspondre à l'ordre du problème
    #df_samples = df_samples[problem_siouxfalls["names"]]

    # Sauvegarder le DataFrame dans un fichier CSV
    df_samples.to_csv(f"Sensitivity_Analysis/cache/LHS/Berlin/lhs_samples_{n_sample}.csv", index=False)