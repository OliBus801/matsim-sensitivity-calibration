import csv

#from SALib.analyze import morris
from SALib.sample import morris
import pandas as pd
import numpy as np


def write_sample_to_csv(problem, sample, filename="morris_method_sample.csv"):
    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(problem['names'])
        for row in sample:
            csvwriter.writerow([round(value, 5) for value in row])


# Group 1 (Scoring Parameters) : ASC_{mode}, waitingPt_util, lateArrival_util, earlyDeparture_util, performing_util, money_util
# Group 2 (Replanning Modules) : TimeAllocationMutator, mutationRange, ReRoute, SubtourModeChoice, ChangeExpBeta, maxAgentPlanMemorySize
# Group 4 (Global Parameters) : timeStepSize, numberOfIterations

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
        [0, 1],         #'ChangeExpBeta', 
        [900, 3600],    #'mutationRange',
        [1, 10],        #'maxAgentPlanMemorySize', 
        [1, 120],       #'timeStepSize',
        [1, 100],       #'numberOfIterations'
                ]
}

# Définir le problème avec des variables restreintes
# On se concentre uniquement sur les six variables avec le plus de feature importances 
# Bornes justifiées par le partial dependence plot
problem_siouxfalls_constrained = {
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
        [-1, 0.444],    #'ASC_car',
        [-1, 3],        #'ASC_pt',
        [0, 0.001],         #'ASC_walk',
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
        [1, 10],        #'maxAgentPlanMemorySize', 
        [1, 18],       #'timeStepSize',
        [2, 12],       #'numberOfIterations'
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

problem_kyoto = {
    "num_vars": 20,
    "names": [
        'ASC_car',
        'ASC_bike',
        'ASC_ride',
        'ASC_truck',
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
        [-1, 3],        #'ASC_truck',
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



def preformat(input_csv, output_csv):
    """
    Preformat a CSV file by applying various transformations, such as normalizing specified columns.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to save the preformatted CSV file.
        columns_to_normalize (list, optional): List of column names to normalize. Defaults to None.

    Returns:
        None
    """
    #columns_to_normalize=["TimeAllocationMutator", "ReRoute", "SubtourModeChoice", "ChangeExpBeta"]

    # Load the CSV file
    df = pd.read_csv(input_csv)

    """
    # Normalize the specified columns row-wise if provided
    if columns_to_normalize:
        df[columns_to_normalize] = df[columns_to_normalize].div(
            df[columns_to_normalize].sum(axis=1), axis=0
        ).round(5)
    """

    # Add a new column 'numberOfIterations' with a default value of 50
    if 'numberOfIterations' not in df.columns:
        df['numberOfIterations'] = 50
    
    # Add a new column 'maxAgentPlanMemorySize' with a default value of 5
    if 'maxAgentPlanMemorySize' not in df.columns:
        df['maxAgentPlanMemorySize'] = 5
    
    # Add a new column 'ASC_walk' with a default value of 0.0
    if 'ASC_walk' not in df.columns:
        df['ASC_walk'] = 0.0

    # Add a new column 'timeStepSize' with a default value of 1
    if 'timeStepSize' not in df.columns:
        df['timeStepSize'] = 1

    # Round the columns 'numberOfIterations' and 'maxAgentPlanMemorySize' to an integer
    #df['numberOfIterations'] = df['numberOfIterations'].round(0).astype(int)
    #df['maxAgentPlanMemorySize'] = df['maxAgentPlanMemorySize'].round(0).astype(int)

    df.to_csv("FOR_RUNS_" + output_csv, index=False)

    # Order the columns by 'numberOfIterations'
    #df = df.sort_values(by='numberOfIterations')

    # Save the preformatted DataFrame to a new CSV file
    #df.to_csv("ordered_" + output_csv, index=False)


sample = morris.sample(problem_siouxfalls_final, N=10, num_levels=4)
write_sample_to_csv(problem_siouxfalls_final, sample, "morris_sample_final.csv")
preformat(input_csv="morris_sample_final.csv", output_csv="morris_sample_final.csv")