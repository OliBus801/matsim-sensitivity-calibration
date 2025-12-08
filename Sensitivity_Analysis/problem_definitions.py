SIOUXFALLS = {
    "num_vars": 16,
    "names": [
        'ASC_car','ASC_pt','ASC_walk',
        'waitingPt_util','lateArrival_util','earlyDeparture_util',
        'performing_util','money_util',
        'TimeAllocationMutator','ReRoute','SubtourModeChoice','ChangeExpBeta',
        'mutationRange','maxAgentPlanMemorySize','timeStepSize','numberOfIterations'
    ],
    "bounds": [
        [-1, 3],[-1, 3],[-1, 3],
        [-12, 0],[-36, 0],[-12, 0],
        [0, 12],[0.25, 2.0],
        [0, 1],[0, 1],[0, 1],[0.01, 1],
        [900, 3600],[1, 10],[1, 120],[1, 100]
    ]
    }

SIOUXFALLS_CONSTRAINED = {
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

SIOUXFALLS_DEFAULT_VALUES = {
    'ASC_walk': 0.0,
    'maxAgentPlanMemorySize': 5,
    'timeStepSize': 1,
    'numberOfIterations': 100
}


BERLIN = {
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

BERLIN_CONSTRAINED = {
    "num_vars": 15,
    "names": [
        'ASC_car',
        'ASC_bike',
        'ASC_ride',
        'ASC_freight',
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
        [-1, 3],        #'ASC_car',
        [-1, 3],        #'ASC_bike',
        [0, 3],         #'ASC_ride',
        [-1, 3],        #'ASC_freight',
        [-1, 3],        #'ASC_pt',
        [-12, 0],       #'waitingPt_util',
        [-36, 0],       #'lateArrival_util',
        [-9.33, -5.21], #'earlyDeparture_util',
        [0, 12],        #'performing_util',
        [0.46, 1.59],   #'money_util',
        [0.06, 0.35],   #'TimeAllocationMutator',
        [0, 1],         #'ReRoute',
        [0, 1],         #'SubtourModeChoice',
        [0, 1],         #'ChangeExpBeta',
        [900, 3600],    #'mutationRange',
                ]
}

BERLIN_SOBOL = {
    "num_vars": 6,
    "names": [
        'ASC_car',
        'ASC_bike',
        'ASC_ride',
        'performing_util',
        'money_util',
        'SubtourModeChoice',
        ],
    "bounds": [
        [-1, 3],        #'ASC_car',
        [-1, 3],        #'ASC_bike',
        [0, 3],         #'ASC_ride',
        [0, 12],        #'performing_util',
        [0.46, 1.59],   #'money_util',
        [0, 1],         #'SubtourModeChoice',
                ]
}

BERLIN_DEFAULT_VALUES = {
    'ASC_walk': 0.0,
    'ASC_car': 1.5,
    'ASC_pt': 1.5,
    'ASC_bike': 1.5,
    'ASC_ride': 1.5,
    'ASC_freight': 1.5,
    'waitingPt_util': 0,
    'lateArrival_util': -18,
    'earlyDeparture_util': 0,
    'money_util': 1,
    'TimeAllocationMutator': 0.1,
    'ReRoute': 0.1,
    'SubtourModeChoice': 0.1,
    'ChangeExpBeta': 0.7,
    'mutationRange': 1800,
    'maxAgentPlanMemorySize': 5,
    'timeStepSize': 1,
    'numberOfIterations': 200
}