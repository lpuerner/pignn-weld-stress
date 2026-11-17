"""
Code für das Preprocessing von FEM-Simulationen in Graphform.
Eingabe:
    Tabellen mit Knotendaten,
    Tabelle mit Elementdatei
Ausgabe:
    Knotenfeatures: [N, 5],
    Kantenfetures: [E, 4],
    Kantenliste: [2,E],
    Name der FEM-Simulation,
    Tabelle mit Element - Knoten beziehung,
"""
import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

work_dir = './'  # Arbeitsverzeichniss
dir_path = os.path.join(work_dir, 'solfiles')  # Ordner mit FEM-Simulationen (node_*.csv / elem_*.csv)
save_path = os.path.join(work_dir, 'KI', 't_weld_v1')  # Zielordner/-name für den fertigen Datensatz

do_element_list_to_dataset = True  # Fügt die Element - Knotenbeziehung dem Datensatz hinzu. !Wichtig fürs ploten!


def cubes_to_edges(df):
    """
    Wandelt die Hexaederelement-Knoten-Tabelle in eine Adjazenzliste um.
    :param df: Hexaederelement-Knoten-Tabelle aus einer FEM-Simulation [E, 8 Knoten-IDs]
    :return: Adjazenzliste für Kanten [2, E]
    """
    edges = []
    for _, row in df.iterrows():
        n1, n2, n3, n4, n5, n6, n7, n8 = (row['node1'], row['node2'], row['node3'], row['node4'],
                                          row['node5'], row['node6'], row['node7'], row['node8'])
        edge_list = [(n1, n2), (n2, n3), (n3, n4), (n4, n1), (n5, n6), (n6, n7), (n7, n8), (n8, n5), (n2, n6),
                     (n3, n7), (n4, n8), (n1, n5)]

        # Sortiere jede Kante intern, um Richtung zu ignorieren
        sorted_edges = [(max(a, b), min(a, b)) for a, b in edge_list]
        edges.extend(sorted_edges)

    # Entferne Duplikate
    unique_edges = list(set(edges))
    edge_tensor = torch.tensor(unique_edges, dtype=torch.int32)
    return edge_tensor.T - 1


if __name__ == '__main__':
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data_list = []
    file_name_list = []
    file_list = os.listdir(dir_path)
    iter_max = len(file_list) / 2
    j = 1
    # Iteriert für alle FEM-Simulationen in dir_path
    for item in file_list:
        print(f'{j}/{iter_max}')
        try:
            design_name = item.partition('_')[2]
            if design_name not in file_name_list and design_name != ".DS":
                info = design_name.split("_")
                file_name_list.append(design_name)

                # Erzeugt string für das einlesen der Daten
                design_node = dir_path + '/' + 'node_' + design_name
                design_element = dir_path + '/' + 'elem_' + design_name
                # Liest Knoten- und Elementdatein aus einer FEM-Simulatioen
                df = pd.read_csv(design_node, header=None)
                df.columns = ['node_id', 'x', 'y', 'z', 'ux', 'uy', 'uz', 'fx', 'fy', 'fz', 'sxx',
                              'syy', 'szz', 'sxy', 'syz', 'sxz', 'mises', 'nodetype']
                df_node_indices_four = pd.read_csv(design_element, header=None)
                df_node_indices_four = df_node_indices_four.iloc[:, :9].astype(int)
                df_node_indices_four.columns = ['elem_id', 'node1', 'node2', 'node3', 'node4',
                                                'node5', 'node6', 'node7', 'node8']

                # Knoten positionen und verschiebungen in Tensoren einlesen
                node_position = np.vstack((df["x"].to_numpy(), df["y"].to_numpy(),
                                           df["z"].to_numpy())).T.astype('float32')
                mesh_pos = torch.tensor(node_position)
                node_u = torch.tensor(np.vstack((df["ux"].to_numpy(), df["uy"].to_numpy(),
                                                 df["uz"].to_numpy())).T.astype('float32'))

                # Spannungen in Tensor einlesen
                stress = torch.tensor(np.vstack((df["sxx"].to_numpy(), df["syy"].to_numpy(), df["szz"].to_numpy(),
                                                 df["sxy"].to_numpy(), df["sxz"].to_numpy(),
                                                 df["syz"].to_numpy())).T.astype('float32'))
                stress_mises = torch.unsqueeze(torch.tensor(df["mises"].to_numpy().T.astype('float32')), 1)
                stressSXX = torch.unsqueeze(torch.tensor(df["sxx"].to_numpy().T.astype('float32')), 1)
                stressSYY = torch.unsqueeze(torch.tensor(df["syy"].to_numpy().T.astype('float32')), 1)
                stressSZZ = torch.unsqueeze(torch.tensor(df["szz"].to_numpy().T.astype('float32')), 1)
                stressSXY = torch.unsqueeze(torch.tensor(df["sxy"].to_numpy().T.astype('float32')), 1)
                stressSXZ = torch.unsqueeze(torch.tensor(df["sxz"].to_numpy().T.astype('float32')), 1)
                stressSYZ = torch.unsqueeze(torch.tensor(df["syz"].to_numpy().T.astype('float32')), 1)

                # Erzeugt Adjazenliste aus Elementtabelle
                edges = cubes_to_edges(df_node_indices_four)

                # Erzeugt Kantenfeatures
                node_position_tensor = torch.from_numpy(node_position)
                u_i = node_position_tensor[edges[0] - 1]
                u_j = node_position_tensor[edges[1] - 1]
                u_ij = u_i - u_j
                u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
                edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).type(torch.float)

                # Erzeugt Knotenfeatures
                node_type_info = df["nodetype"].to_numpy().T.astype('int32')
                node_type = nn.functional.one_hot(torch.tensor(node_type_info).to(torch.long), num_classes=2)
                x = torch.cat((mesh_pos, node_type), dim=-1).type(torch.float)

                # Liest Daten in Liste ein aus welcher der Datensatz erstellt wird
                if do_element_list_to_dataset:
                    elist = torch.tensor(df_node_indices_four.iloc[:, 1:9].values)
                    data_list.append(Data(x=x, node_u=node_u,
                                          edge_index=edges, edge_attr=edge_attr, stress=stress, elist=elist,
                                          mesh_pos=mesh_pos, stress11=stressSXX,
                                          stress22=stressSYY, stress33=stressSZZ, stress12=stressSXY,
                                          stress13=stressSXZ, stress23=stressSYZ, stress_mises=stress_mises,
                                          name=design_name))
                else:
                    data_list.append(Data(x=x, node_u=node_u,
                                          edge_index=edges, edge_attr=edge_attr, stress=stress,
                                          mesh_pos=mesh_pos, stress11=stressSXX,
                                          stress22=stressSYY, stress33=stressSZZ, stress12=stressSXY,
                                          stress13=stressSXZ, stress23=stressSYZ, stress_mises=stress_mises,
                                          name=design_name))
            else:
                pass
            j = j + 1
        except Exception as e:
            print(e)

    print("Wuhu. Fertig!")
    torch.save(data_list, os.path.join(save_path + '.pt'))
