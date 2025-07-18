import pandas as pd
from lxml import etree

# Charger les flux horaires à partir du fichier CSV
msa_df = pd.read_csv("/mnt/data/msa_results.csv")

# Créer l'arbre XML de base
root = etree.Element("counts", name="from_msa_assignment")

# Grouper par lien
grouped = msa_df.groupby("link_id")

for link_id, group in grouped:
    count = etree.SubElement(root, "count", locId=str(link_id), linkId=str(link_id))
    for _, row in group.iterrows():
        hour = int(row["hour"])
        flow = int(round(row["flow"]))
        if flow > 0:
            etree.SubElement(count, "volume", h=str(hour), val=str(flow))

# Écrire dans un fichier XML
output_file = "/mnt/data/msa_counts.xml"
tree = etree.ElementTree(root)
tree.write(output_file, pretty_print=True, xml_declaration=True, encoding="UTF-8")

output_file
