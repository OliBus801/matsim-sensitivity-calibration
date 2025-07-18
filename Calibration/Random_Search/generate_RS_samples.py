import pandas as pd
import numpy as np

# Define hyperparameter intervals (to be modified by the user)
param_bounds = {
    'ASC_car': (-1, 0.444),
    'ASC_pt': (-1, 3),
    'waitingPt_util': (-1, 3), 
    'performing_util': (0, 5),
    'SubtourModeChoice': (0, 1),
}

# Budget configuration
total_samples = 150
adaptive_split = [1/3, 1/3, 1/3]  # fractions for adaptive search
early_stopping_fraction = 0.5  # proportion of budget used in early-stopping phase
top_fraction_to_continue = 0.3  # top % to continue

# 1. Naive Random Search
def random_search_naive(bounds, n_samples):
    return pd.DataFrame({
        param: np.random.uniform(low, high, n_samples)
        for param, (low, high) in bounds.items()
    })

# 2. Adaptive Random Search (3 phases with bounds refinement)
def random_search_adaptive(bounds, split, total_samples, score_func, top_fraction=0.2):
    """
    Adaptive Random Search: at each phase, select the best candidates
    and refine the bounds around their parameters.
    """
    samples = []
    current_bounds = bounds.copy()
    prev_df = None

    for i, frac in enumerate(split):
        n = int(total_samples * frac)
        df = random_search_naive(current_bounds, n)

        # If not the first phase, concatenate with previous candidates
        if prev_df is not None:
            df = pd.concat([prev_df, df], ignore_index=True)

        # Evaluate candidates
        scores = score_func(df)
        df = df.copy()
        df["score"] = scores

        # Select the top X% candidates
        n_top = max(1, int(len(df) * top_fraction))
        top_candidates = df.nsmallest(n_top, "score").drop(columns=["score"])

        # Refine bounds around the best candidates
        for param in current_bounds:
            low = top_candidates[param].min()
            high = top_candidates[param].max()
            # Optionally: widen the interval a bit to avoid being too restrictive
            margin = (high - low) * 0.1
            current_bounds[param] = (
                max(bounds[param][0], low - margin),
                min(bounds[param][1], high + margin)
            )

        samples.append(df.drop(columns=["score"]))
        prev_df = top_candidates  # For the next phase

    return pd.concat(samples, ignore_index=True)

# 3. Random Search with Early Stopping (only simulates initial selection here)
def random_search_early_stopping(bounds, total_samples, early_fraction, top_fraction):
    early_phase = int(total_samples * early_fraction)
    remaining = total_samples - early_phase

    early_candidates = random_search_naive(bounds, early_phase)

    # Placeholder for selection score: here, generate a dummy score
    early_candidates["score"] = np.random.uniform(0.5, 1.5, size=early_phase)
    top_candidates = early_candidates.nsmallest(int(early_phase * top_fraction), "score").drop(columns=["score"])

    late_candidates = top_candidates.sample(n=remaining, replace=True).reset_index(drop=True)

    return pd.concat([early_candidates.drop(columns=["score"]), late_candidates], ignore_index=True)

# Generate candidate sets
df_naive = random_search_naive(param_bounds, total_samples)
# For adaptive search, you need to provide a scoring function. Here is a dummy one:
dummy_score_func = lambda df: np.random.uniform(0.5, 1.5, size=len(df))
df_adaptive = random_search_adaptive(param_bounds, adaptive_split, total_samples, dummy_score_func)
df_earlystop = random_search_early_stopping(param_bounds, total_samples, early_stopping_fraction, top_fraction_to_continue)

# Save to CSV
df_naive.to_csv("Calibration/cache/random_search_naive.csv", index=False)
df_adaptive.to_csv("Calibration/cache/random_search_adaptive.csv", index=False)
df_earlystop.to_csv("/mnt/data/random_search_earlystop.csv", index=False)
