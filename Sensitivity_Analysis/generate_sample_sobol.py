import pandas as pd
import numpy as np
from SALib.sample import sobol
from skbio.stats.composition import  ilr_inv

# Définir le nombre d'échantillons de base
N = 1024

var_names = [
        'ASC_car',
        'ASC_pt',
        #'ASC_walk',
        'waitingPt_util',
        'lateArrival_util',
        'earlyDeparture_util',
        'performing_util',
        'money_util',
        'mutationRange',
        #'maxAgentPlanMemorySize', 
        #'timeStepSize',
        #'numberOfIterations',
        'TimeAllocationMutator',
        'ReRoute', 
        'SubtourModeChoice',
        'ChangeExpBeta' 
        ]

# Définir les bornes pour les 12 variables libres
bounds_free = [
    [-1, 3], [-1, 3], [-1, 3],
    [-12, 0], [-36, 0], [-12, 0],
    [0, 12], [0.25, 2.0],
    [900, 3600], [1, 10], [1, 120], [1, 100]
]

bounds_constrained = [
    [-1, 0.444], [-1, 3],
    [-12, 0], [-36, 0], [-12, 0],
    [0, 5], [1.6, 2.0],
    [900, 3600], [1, 10], [1, 18], [2, 12]
]

bounds_constrained_notimestep = [
    [-1, 0.444], [-1, 3],
    [-12, 0], [-36, 0], [-12, 0],
    [0, 5], [1.6, 2.0],
    [900, 3600], [1, 10], [2, 12]
]

bounds_constrained_final = [
    [-1, 0.444], [-1, 3],
    [-12, 0], [-36, 0], [-12, 0],
    [0, 5], [1.6, 2.0],
    [900, 3600],
]

# Définir les bornes pour les 3 variables ILR (approximées ici entre -5 et 5)
bounds_ilr = [[-5, 5]] * 3

# Créer le dictionnaire 'problem' pour SALib
problem = {
    'num_vars': 11,
    'names': [
        'ASC_car', 'ASC_pt', 
        #'ASC_walk',
        'waitingPt_util', 'lateArrival_util', 'earlyDeparture_util',
        'performing_util', 'money_util',
        'mutationRange',
        #'maxAgentPlanMemorySize', 
        #'timeStepSize',
        #'numberOfIterations',
        'ILR1', 'ILR2', 'ILR3'
    ],
    'bounds': bounds_constrained_final + bounds_ilr
}

# Générer les échantillons avec la méthode de Saltelli
param_values = sobol.sample(problem, N, calc_second_order=False)

# Sauvegarder les échantillons dans un fichier CSV
param_df = pd.DataFrame(param_values, columns=problem['names'])
param_df.to_csv('Sensitivity_Analysis/cache/Sobol/Final/sobol_samples_final.csv', index=False)

# Séparer les échantillons des variables libres et des variables ILR
samples_free = param_values[:, :8]
samples_ilr = param_values[:, 8:]

# Appliquer la rétro-transformation ILR pour obtenir les 4 variables contraintes
weights = np.array([ilr_inv(sample) for sample in samples_ilr])

# Combiner les variables libres et contraintes pour obtenir l'échantillon complet
full_samples = np.hstack((samples_free, weights))

# Mettre dans un DataFrame
df_samples = pd.DataFrame(full_samples, columns=var_names)

# Arrondir les valeurs à 5 décimales
df_samples = df_samples.round(5)

# Arrondir "maxAgentPlanMemorySize" et "numberOfIterations" à l'entier supérieur
#df_samples['maxAgentPlanMemorySize'] = np.ceil(df_samples['maxAgentPlanMemorySize']).astype(int)
#df_samples['numberOfIterations'] = np.ceil(df_samples['numberOfIterations']).astype(int)

# Ajouter une colonne 'numberOfIterations' avec des valeurs égales à 50
df_samples['numberOfIterations'] = 50

# Ajouter une colonne 'maxAgentPlanMemorySize' avec des valeurs égales à 5
df_samples['maxAgentPlanMemorySize'] = 5

# Ajouter une colonne 'ASC_walk' avec des valeurs nulles
df_samples['ASC_walk'] = 0.0

# Ajouter une colonne 'timeStepSize' avec des valeurs égales à 1
df_samples['timeStepSize'] = 1

# Arranger l'ordre des colonnes pour correspondre à l'ordre souhaité
desired_order = [
    'ASC_car', 'ASC_pt', 
    'ASC_walk',
    'waitingPt_util', 'lateArrival_util', 'earlyDeparture_util',
    'performing_util', 'money_util', 'TimeAllocationMutator', 'ReRoute', 'SubtourModeChoice',
    'ChangeExpBeta', 'mutationRange', 'maxAgentPlanMemorySize', 
    'timeStepSize',
    'numberOfIterations'
]
df_samples = df_samples[desired_order]

# Enregistrer le DataFrame dans un fichier CSV
df_samples.to_csv('Sensitivity_Analysis/cache/Sobol/Final/TEST.csv', index=False)
