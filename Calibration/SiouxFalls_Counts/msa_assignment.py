import xml.etree.ElementTree as ET
import networkx as nx
import pandas as pd
from collections import defaultdict

def parse_network_to_graph(network_file):
    tree = ET.parse(network_file)
    root = tree.getroot()

    G = nx.DiGraph()
    link_to_nodes = {}

    for link in root.find('links'):
        lid = link.attrib['id']
        from_id = link.attrib['from']
        to_id = link.attrib['to']
        length = float(link.attrib.get('length', 1))
        freespeed = float(link.attrib.get('freespeed', 13.9))
        capacity = float(link.attrib.get('capacity', 1000.0))

        travel_time = length / freespeed

        G.add_edge(from_id, to_id, id=lid, t0=travel_time, capacity=capacity, flow=0.0, time=travel_time)
        link_to_nodes[lid] = (from_id, to_id)

    return G, link_to_nodes

def update_link_travel_times(G):
    for u, v, data in G.edges(data=True):
        v_a = data['flow']
        t0 = data['t0']
        c = data['capacity']
        data['time'] = t0 * (1 + 0.15 * (v_a / c) ** 4)

def msa_assignment(G, link_to_nodes, od_df, max_iter=200, tol=1e-3):
    results = []

    for hour, od_group in od_df.groupby('hour'):
        # initialise flows
        for u, v in G.edges():
            G[u][v]['flow'] = 0.0

        demands = [(row["start_link"], row["end_link"], row["count"]) for _, row in od_group.iterrows()]
        for k in range(1, max_iter + 1):
            update_link_travel_times(G)
            aux_flow = defaultdict(float)

            for start_link, end_link, trips in demands:
                if start_link not in link_to_nodes or end_link not in link_to_nodes:
                    continue
                source_node = link_to_nodes[start_link][0]
                target_node = link_to_nodes[end_link][1]

                try:
                    path = nx.shortest_path(G, source=source_node, target=target_node, weight='time')
                except nx.NetworkXNoPath:
                    continue

                for u, v in zip(path[:-1], path[1:]):
                    aux_flow[(u, v)] += trips

            # update flows
            step = 1 / k
            max_delta = 0
            for (u, v), x_star in aux_flow.items():
                xk = G[u][v]['flow']
                new_flow = xk + step * (x_star - xk)
                max_delta = max(max_delta, abs(new_flow - xk))
                G[u][v]['flow'] = new_flow

            if max_delta < tol:
                print(f"[Hour {hour}] Converged in {k} iters, max_delta={max_delta:.6f}")
                break

        # collect results
        for u, v, data in G.edges(data=True):
            results.append({
                "hour": hour,
                "link_id": data["id"],
                "from": u,
                "to": v,
                "flow": round(data["flow"])
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Chemin vers le fichier XML du réseau MATSim
    network_xml = "/media/olibuss/OliPassport/University/Master/MATSIM/matsim_project/scenarios/siouxfalls-2014/Siouxfalls_network_PT.xml"  # À adapter si besoin

    # Charger le graphe
    G, link_to_nodes = parse_network_to_graph(network_xml)

    # Charger les OD counts à partir d'un fichier CSV
    od_df = pd.read_csv('Calibration/cache/od_counts.csv')

    # Exécuter l'algorithme MSA
    df_results = msa_assignment(G, link_to_nodes, od_df)
    print(df_results.head())

    # Sauvegarder les résultats dans un fichier CSV
    df_results.to_csv('Calibration/cache/msa_results.csv', index=False)