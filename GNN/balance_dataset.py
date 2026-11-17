"""
Preprocessing Skript um eine Gleichverteilung der maximalen Von-Mises-Spannung
im Nahtübergang zu erreichen.
"""
import os
import torch
import numpy as np
import pandas as pd
import utils as u

work_dir = './'  # Arbeitsverzeichniss
input_dataset = os.path.join(work_dir, 'KI', 'butt_konst_r1_v1.pt')   # Datensatz aus fem_to_graph.py
output_dataset = os.path.join(work_dir, 'KI', 'butt_konst_r1_v2.pt')  # reduzierter Datensatz


def limit_bins(df: pd.DataFrame,
               value_column: str = "fem_mean",
               bin_size: float = 0.3,
               max_per_bin: int = 15,
               bin_range: tuple = (0, 4)):
    """
    Reduziert den Datensatz um eine gleichverteilung zu erreichen.
    Die Reduzierung erfolgt wie bei einem Histogram.
    """
    df = df.copy()
    # Bins erstellen nach fem_mean
    bins = np.arange(bin_range[0], bin_range[1] + bin_size, bin_size)
    df['bin'] = pd.cut(df[value_column], bins=bins, right=False)

    # Pro bin maximal max_per_bin FEM-Simulationen behalten
    df_limited = df.groupby('bin', group_keys=False).apply(lambda g: g.head(max_per_bin)).reset_index(drop=True)
    return df_limited


if __name__ == '__main__':
    # Lädt Dataset und bestimmt für jede Simulation die maximalen Von-Mises-Spannungen
    # im Nahtübergang und speichert diese
    ds = torch.load(input_dataset, weights_only=False)
    extreme_stresses = []
    i = 0
    for data in ds:
        print(f'{i+1}/{len(ds)}')
        mises_r2, mises_fem, mises_fem = u.get_mises(data.stress, data.stress)
        extrema_vals_array, max_stresses = u.find_extreme_nodes_by_abs_and_box_v2(data, mises_fem, mises_fem,
                                                                                  0.05, 0.05)
        df_temp = pd.DataFrame({
            'fem': max_stresses[0],
            'name': data.name
        })
        extreme_stresses.append(df_temp)
        i = i + 1
    df_max_s = pd.concat(extreme_stresses, ignore_index=True)

    # Sortiert DataFrame
    df_max_s = df_max_s.copy()
    df_max_s["fem_mean"] = df_max_s.groupby("name")["fem"].transform("mean")
    df_max_s = df_max_s[["name", "fem_mean"]].drop_duplicates(subset="name")

    # Erzeugt Blaupause für den reduzierten Datensatz
    bin_size = 0.05
    bin_range = (1.5, 3.5)
    max_per_bin = 12
    samples = limit_bins(df_max_s, bin_size=bin_size, bin_range=bin_range, max_per_bin=max_per_bin)

    # Erzeugt reduzierten Datensatz
    allowed_names = set(samples['name'])
    data_list = []
    for data in ds:
        if data.name in allowed_names:
            data_list.append(data)

    torch.save(data_list, output_dataset)
