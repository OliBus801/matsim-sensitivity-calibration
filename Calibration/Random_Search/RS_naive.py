import pandas as pd
import numpy as np
import os

# Define hyperparameter intervals (to be modified by the user)
param_bounds_sobol_s1 = {
    'ASC_car': (-1, 0.444),
    'performing_util': (0, 5),
    'SubtourModeChoice': (0, 1),
}

param_bounds_sobol_st = {
    'ASC_car': (-1, 0.444),
    'ASC_pt': (0.75, 2.25),
    'performing_util': (0, 5),
    'SubtourModeChoice': (0, 1),
    'money_util': (0.25, 2.0),
}

param_bounds_morris_mu = {
    'ASC_car': (-1, 0.444),
    'ASC_pt': (0.75, 2.25),
    'lateArrival_util': (-36, 0),
    'performing_util': (0, 5),
    'SubtourModeChoice': (0, 1),
}

param_bounds_morris_sigma = {
    'ASC_car': (-1, 0.444),
    'ASC_pt': (0.75, 2.25),
    'waitingPt_util': (-1, 3),
    'lateArrival_util': (-36, 0),
    'money_util': (0.25, 2.0),
    'mutationRange': (900, 3600),
    'performing_util': (0, 5),
    'SubtourModeChoice': (0, 1),
    'ChangeExpBeta': (0.01, 1),
}

default_param_values = {
    'ASC_walk': 0,
    'ASC_pt': 1.5,
    'lateArrival_util': -18,
    'waitingPt_util': 0,
    'earlyDeparture_util': 0,
    'money_util': 1,
    'TimeAllocationMutator': 0.1,
    'ReRoute': 0.1,
    'ChangeExpBeta': 0.7,
    'mutationRange': 1800,
    'maxAgentPlanMemorySize': 5,
    'timeStepSize': 1,
    'numberOfIterations': 50,
}

original_param_bounds = {
    'ASC_car': (-1, 3),
    'ASC_pt': (-1, 3),
    'ASC_walk': (-1, 3),
    'waitingPt_util': (-12, 0),
    'lateArrival_util': (-36, 0),
    'earlyDeparture_util': (-12, 0),
    'performing_util': (0, 12),
    'money_util': (0.25, 2.0),
    'TimeAllocationMutator': (0, 1),
    'ReRoute': (0, 1),
    'SubtourModeChoice': (0, 1),
    'ChangeExpBeta': (0.01, 1),
    'mutationRange': (900, 3600),
    'maxAgentPlanMemorySize': (1, 10),
    'timeStepSize': (1, 120),
    'numberOfIterations': (1, 100),
}

refined_param_bounds = {
    'ASC_car': (-1, 0.444),
    'ASC_pt': (0.75, 2.25),
    'waitingPt_util': (-12, 0),
    'lateArrival_util': (-36, 0),
    'earlyDeparture_util': (-12, 0),
    'performing_util': (0, 5),
    'money_util': (1.6, 2.0),
    'TimeAllocationMutator': (0, 1),
    'ReRoute': (0, 1),
    'SubtourModeChoice': (0, 1),
    'ChangeExpBeta': (0.01, 1),
    'mutationRange': (900, 3600)
}

PARAM_NAMES = list(original_param_bounds.keys())

# Budget configuration
total_samples = 100

def random_search_naive(bounds, n_samples):
    result = {}
    for param in PARAM_NAMES:
        if param in bounds:
            low, high = bounds[param]
            result[param] = np.random.uniform(low, high, n_samples)
        else:
            default_value = default_param_values[param]
            result[param] = np.full(n_samples, default_value)
    result['maxAgentPlanMemorySize'] = np.round(result['maxAgentPlanMemorySize']).astype(int)
    result['numberOfIterations'] = np.round(result['numberOfIterations']).astype(int)
    total_replanning = result['TimeAllocationMutator'] + result['ReRoute'] + result['SubtourModeChoice'] + result['ChangeExpBeta']
    result['TimeAllocationMutator'] /= total_replanning
    result['ReRoute'] /= total_replanning
    result['SubtourModeChoice'] /= total_replanning
    result['ChangeExpBeta'] /= total_replanning
    return pd.DataFrame(result)

# List of seeds
seeds = [256, 123, 42]

for seed in seeds:
    np.random.seed(seed)
    RS_naive = random_search_naive(original_param_bounds, total_samples)
    RS_refined_naive = random_search_naive(original_param_bounds, total_samples)
    RS_Morris_mu = random_search_naive(param_bounds_morris_mu, total_samples)
    RS_Morris_sigma = random_search_naive(param_bounds_morris_sigma, total_samples)
    RS_Sobol_s1 = random_search_naive(param_bounds_sobol_s1, total_samples)
    RS_Sobol_st = random_search_naive(param_bounds_sobol_st, total_samples)
    outdir = f"Calibration/cache/Naive_V2"
    os.makedirs(outdir, exist_ok=True)
    RS_naive.to_csv(f"{outdir}/RS_naive_samples_seed{seed}.csv", index=False)
    RS_refined_naive.to_csv(f"{outdir}/RS_refined_naive_samples_seed{seed}.csv", index=False)
    RS_Morris_mu.to_csv(f"{outdir}/RS_Morris_mu_samples_seed{seed}.csv", index=False)
    RS_Morris_sigma.to_csv(f"{outdir}/RS_Morris_sigma_samples_seed{seed}.csv", index=False)
    RS_Sobol_s1.to_csv(f"{outdir}/RS_Sobol_s1_samples_seed{seed}.csv", index=False)
    RS_Sobol_st.to_csv(f"{outdir}/RS_Sobol_st_samples_seed{seed}.csv", index=False)
