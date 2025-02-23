import torch_geometric as pyg
import torch
def read_graph_file(file_path):
    vertices = {}
    edges = []
    filtered_vertices = {}
    node_to_old = {}
    node_to_new = {}
    new_node_id = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        is_vertices_section = False
        is_edges_section = False

        for line in lines:
            line = line.strip()

            if line.startswith("*Vertices"):
                is_vertices_section = True
                is_edges_section = False
                continue
            elif line.startswith("*Edges"):
                is_vertices_section = False
                is_edges_section = True
                continue

            if is_vertices_section:
                parts = line.split('\t')
                node_id = int(parts[0])
                node_name = parts[1].strip('"')
                node_weight = int(parts[2])
                node_type = parts[3]
                categories = [cat.strip() for cat in parts[4].split(';') if cat.strip()]

                vertices[node_id] = {
                    'node_name': node_name,
                    'node_weight': node_weight,
                    'node_type': node_type,
                    'categories': categories
                }

                # Check if the node's categories match any of the target patterns and if node_type is "starring"
                match_type = None

                if node_type == "starring":
                    if ("American film actors" in categories and
                            "American television actors" not in categories and
                            "American stage actors" not in categories):
                        match_type = 0

                    elif ("American film actors" in categories and
                          "American television actors" in categories and
                          "American stage actors" not in categories):
                        match_type = 1

                    elif ("American television actors" in categories and
                          "American stage actors" in categories and
                          "American film actors" not in categories):
                        match_type = 2

                    elif any(cat.startswith("English") for cat in categories):
                        match_type = 3

                    elif any(cat.startswith("Canadian") for cat in categories):
                        match_type = 4

                categories = [cat for cat in categories if not any(cat.startswith(prefix) for prefix in
                                                                   ["American film actors",
                                                                    "American television actors",
                                                                    "American stage actors", "English", "Canadian"])]

                if match_type is not None:
                    filtered_vertices[new_node_id] = {
                        'original_node_id': node_id,
                        'node_name': node_name,
                        'node_weight': node_weight,
                        'node_type': node_type,
                        'categories': categories,
                        'type': match_type
                    }
                    node_to_old[new_node_id] = node_id
                    node_to_new[node_id] = new_node_id
                    new_node_id += 1

            if is_edges_section:
                parts = line.split()
                node_id1 = int(parts[0])
                node_id2 = int(parts[1])
                edges.append((node_id1, node_id2))

    # Filter edges to include only those that map to the filtered nodes
    new_edges = [
        (node_to_new[node_id1], node_to_new[node_id2])
        for node_id1, node_id2 in edges
        if node_id1 in node_to_new and node_id2 in node_to_new
    ]

    edge_index = [[e[0] for e in new_edges], [e[1] for e in new_edges]]

    return vertices, edges, filtered_vertices, node_to_old, edge_index

import numpy as np
label_texts = np.array(["American film actors (only)", "American film actors and American television actors", "American television actors and American stage actors", "English actors", "Canadian actors"])
file_path = './Actor/newmovies.txt'
vertices, edges, filtered_vertices, node_to_old, edge_index = read_graph_file(file_path)
types = [vertex['type'] for vertex in filtered_vertices.values()]
non_isolated_nodes = set(edge_index[0]) | set(edge_index[1])
oldtonew = {old_id: new_id for new_id, old_id in enumerate(sorted(list(non_isolated_nodes)))}
newtoold = {v: k for k, v in oldtonew.items()}
filtered_edges = [(oldtonew[edge[0]], oldtonew[edge[1]]) for edge in zip(edge_index[0], edge_index[1]) if
                  edge[0] in non_isolated_nodes and edge[1] in non_isolated_nodes]
data = pyg.data.data.Data(edge_index=torch.tensor(edges))
nx_g = pyg.utils.to_networkx(data, to_undirected=True)
edge_index = torch.tensor(list(nx_g.edges())).T
edges = edge_index.numpy()
node_labels = np.array([types[newtoold[node_id]] for node_id in range(len(non_isolated_nodes))])
text = []
node_text = "node name: {}; the number of words introducing the node on Wikipedia: {}; node type: {}; Some key words on Wikipedia pages: {}"
for i in range(len(non_isolated_nodes)):
    textpart = node_text.format(
        filtered_vertices[newtoold[i]]['node_name'],
        filtered_vertices[newtoold[i]]['node_weight'],
        filtered_vertices[newtoold[i]]['node_type'],
        ','.join(filtered_vertices[newtoold[i]]['categories'])
    )
    text.append(textpart)
node_texts = np.array(text)
np.savez(
    'dataset.npz',
    edges=edges,
    node_labels=node_labels,
    node_texts=node_texts,
    label_texts=label_texts
)
