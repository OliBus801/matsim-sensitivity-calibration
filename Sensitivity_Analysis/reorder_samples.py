import pandas as pd
import sys

# Charger les fichiers
ordered_df = pd.read_csv("cache/ordered_siouxfalls_SA.csv")
unordered_df = pd.read_csv("cache/unordered_siouxfalls_SA.csv")
output_df = pd.read_csv("cache/unordered_siouxfalls_output.csv")

# Créer une clé unique pour chaque ligne (tu peux adapter selon les colonnes pertinentes)
def create_key(df):
    return df.astype(str).agg('-'.join, axis=1)

ordered_keys = create_key(ordered_df)
unordered_keys = create_key(unordered_df)


# Créer un mapping clé -> output
output_df.index = ordered_keys
output_reordered = output_df.loc[unordered_keys].reset_index(drop=True)

# Sauvegarder le fichier réordonné
output_reordered.to_csv("ordered_siouxfalls_output.csv", index=False)
