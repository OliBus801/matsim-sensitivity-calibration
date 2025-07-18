from lxml import etree
from collections import defaultdict
import pandas as pd
import gzip
import math

# Fonction de parsing de temps (hh:mm:ss -> secondes)
def parse_time_to_sec(time_str):
    h, m, s = map(int, time_str.strip().split(":"))
    return h * 3600 + m * 60 + s

# Lecture du fichier XML compressé
plans_file = "/media/olibuss/OliPassport/University/Master/MATSIM/MATSIM_Source/Baselines/No_Replan/output_plans.xml.gz"  # À adapter si besoin

# Structure de sortie : od_counts[hour][(start_link, end_link)] = count
od_counts = defaultdict(lambda: defaultdict(int))

# Lire XML incrémentalement
with gzip.open(plans_file, 'rb') as f:
    context = etree.iterparse(f, events=('end',), tag='person')

    for event, person in context:
        plans = person.findall('.//plan[@selected="yes"]')
        if not plans:
            continue
        plan = plans[0]

        legs = plan.findall('leg')
        for leg in legs:
            mode = leg.attrib.get("mode")
            if mode != "car":
                continue

            dep_time_str = leg.attrib.get("dep_time")
            if dep_time_str is None:
                continue
            try:
                dep_time = parse_time_to_sec(dep_time_str)
            except:
                continue

            route = leg.find('route')
            if route is None:
                continue

            start_link = route.attrib.get("start_link")
            end_link = route.attrib.get("end_link")
            if not start_link or not end_link:
                continue

            hour = math.ceil(dep_time / 3600)
            od_counts[hour][(start_link, end_link)] += 1

        person.clear()

# Convertir en DataFrame pour inspection
records = []
for hour, od_pairs in od_counts.items():
    for (start, end), count in od_pairs.items():
        records.append({
            "hour": int(hour),
            "start_link": start,
            "end_link": end,
            "count": count
        })

od_df = pd.DataFrame.from_records(records)

# Sauvegarder en CSV
od_df.to_csv('od_counts.csv', index=False)
