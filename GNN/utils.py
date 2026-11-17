"""
Hilfsfunktionen für Preprocessing, Evaluationsmetriken und die normierte Darstellung.
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# Allgemeine Metriken
# ---------------------------------------------------------------------------

def get_R22(data, difference):
    """Bestimmtheitsmaß R² aus Label und Differenz (Label - Vorhersage)."""
    numerator = np.sum(difference**2)
    diff = data - np.mean(data)
    denominator = np.sum(diff**2)
    r2 = 1 - numerator / denominator
    return r2


def rae(actual, predicted):
    """Relative Absolute Error."""
    numerator = np.sum(np.abs(predicted - actual))
    denominator = np.sum(np.abs(np.mean(actual) - actual))
    return numerator / denominator


def get_mae(real_stress, pred_stress):
    """Mean Absolute Error."""
    MAE = 1 / pred_stress.shape[0] * np.sum(np.abs(pred_stress - real_stress))
    return MAE


def get_mises(truth, pred):
    """
    Berechnet die Von-Mises-Spannung aus den 6 Spannungskomponenten
    [s11, s22, s33, s12, s23, s13] für FEM-Ergebnis und GNN-Vorhersage.
    Rückgabe: r2, mises_fem, mises_gnn
    """
    truth = truth.numpy() if isinstance(truth, torch.Tensor) else truth
    pred = pred.numpy() if isinstance(pred, torch.Tensor) else pred

    df1 = pd.DataFrame(truth, columns=['s11', 's22', 's33', 's12', 's23', 's13'])
    df1["mises"] = df1.apply(lambda x: (((x['s11']-x['s22'])**2+(x['s22']-x['s33'])**2+(x['s11']-x['s33'])**2
                                         + 6*(x['s12']**2+x['s23']**2+x['s13']**2))/2)**0.5, axis=1)

    df2 = pd.DataFrame(pred, columns=['s11', 's22', 's33', 's12', 's23', 's13'])
    df2["mises"] = df2.apply(lambda x: (((x['s11']-x['s22'])**2+(x['s22']-x['s33'])**2+(x['s11']-x['s33'])**2
                                         + 6*(x['s12']**2+x['s23']**2+x['s13']**2))/2)**0.5, axis=1)
    return r2_score(df1['mises'].to_numpy(), df2['mises'].to_numpy()), df1['mises'].to_numpy(), df2['mises'].to_numpy()


def write_df_to_excel_sheet(df, excel_path, sheet_name):
    """
    Schreibt DataFrame in Ziel-xlsx auf entsprechendem sheet.
    """
    if os.path.exists(excel_path):
        # Lade bestehende Excel-Datei
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Erstelle neue Excel-Datei
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# ---------------------------------------------------------------------------
# Auswahl der maximalen Spannungen im Nahtübergang
# ---------------------------------------------------------------------------

def find_extreme_nodes_by_abs_and_box_v2(data, stress, pred_stress, boundary_min, boundary_max, plot_mode=False):
    """
    Bestimmt die Knoten mit minimaler/maximaler Spannung in einer Box um die
    Schweißnaht (Stumpfstoß). Wird in balance_dataset.py verwendet.
    """
    coords = data.mesh_pos
    coords = coords.numpy()
    if not isinstance(stress, np.ndarray):
        stress = stress.numpy()
    if not isinstance(pred_stress, np.ndarray):
        pred_stress = pred_stress.numpy()

    node_ids = np.arange(1, len(coords) + 1)
    x_half = max(coords[:, 0]) / 2
    x_min = x_half - 20
    x_max = x_half + 20
    z_min = 0
    z_max = max(coords[:, 2])
    y_min = 0
    y_max = max(coords[:, 1]) + 5

    in_box_mask = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max) &
        (coords[:, 2] >= z_min) & (coords[:, 2] <= z_max)
    )

    # Wende Maske an
    filtered_ids = node_ids[in_box_mask]
    filtered_stresses = stress[in_box_mask]
    # Minimum und Maximum berechnen
    min_val = np.min(filtered_stresses)
    max_val = np.max(filtered_stresses)
    # Neue Grenzen definieren
    lower_bound = min_val + boundary_min
    upper_bound = max_val - boundary_max
    # IDs entsprechend den Grenzen filtern
    lower_ids = filtered_ids[filtered_stresses <= lower_bound] - 1
    upper_ids = filtered_ids[filtered_stresses >= upper_bound] - 1

    eps = 1e-8  # Schutz gegen Division durch 0
    real_low = stress[lower_ids]
    real_high = stress[upper_ids]
    gnn_low = pred_stress[lower_ids]
    gnn_high = pred_stress[upper_ids]
    diff_low = np.abs(real_low - gnn_low)
    diff_high = np.abs(real_high - gnn_high)
    mre_min = np.mean((diff_low / (np.abs(real_low) + eps)) * 100)
    mre_max = np.mean((diff_high / (np.abs(real_high) + eps)) * 100)
    mae_min = np.mean(diff_low)
    mae_max = np.mean(diff_high)
    val_min = np.mean(real_low)
    val_max = np.mean(real_high)
    val_std_min = np.std(real_low)
    val_std_max = np.std(real_high)
    res_array = [mre_max, mre_min, mae_max, mae_min, val_max, val_min, val_std_max, val_std_min]

    if plot_mode:
        # Nur für visualisierung
        labels = np.zeros(len(coords), dtype=int)
        labels[lower_ids - 1] = -1
        labels[upper_ids - 1] = +1
        return labels
    return res_array, [real_high, gnn_high]


def find_extreme_nodes_by_abs_and_two_boxes(data, vm_label_g, vm_pred_g, boundary_min=0.05, boundary_max=0.05,
                                            plot_mode=False, box_T=False):
    """
    Berechnet die Evaluationsmetrik MRE_max für den linken und rechten Nahtübergang
    (Stumpfstoß und T-Stoß).
    """
    # Input wird in Tensoren umgewandelt
    mesh_pos_g = data.mesh_pos
    if not isinstance(vm_label_g, torch.Tensor):
        vm_label_g = torch.as_tensor(vm_label_g)
    if not isinstance(vm_pred_g, torch.Tensor):
        vm_pred_g = torch.as_tensor(vm_pred_g)

    # Erstellt die Masken zur Knotenauswahl im Bereich der Nahtübergängen
    if mesh_pos_g[:, 1].max() > 50:  # Maske für den T-Stoß
        h = mesh_pos_g[:, 1].max() - 50  # idx:achse- 0=x, 1=z, 2=y
        y_min, y_max = 0, 20
        z_min, z_max = h - 3, h + 5

        yz_mask = (
            (mesh_pos_g[:, 2] >= y_min) & (mesh_pos_g[:, 2] <= y_max) &
            (mesh_pos_g[:, 1] >= z_min) & (mesh_pos_g[:, 1] <= z_max)
        )

        x_mask_left = (mesh_pos_g[:, 0] >= -30) & (mesh_pos_g[:, 0] <= 5)
        x_mask_right = (mesh_pos_g[:, 0] > 5) & (mesh_pos_g[:, 0] <= 30)

        final_left = yz_mask & x_mask_left
        final_right = yz_mask & x_mask_right
    else:  # Maske für die Stumpfschweißnaht
        x_half = mesh_pos_g[:, 0].max() / 2
        x_min, x_max = x_half - 20, x_half + 20
        z_min, z_max = 0, mesh_pos_g[:, 1].max()
        y_min, y_max = 0, mesh_pos_g[:, 2].max() + 5

        yz_mask = (
            (mesh_pos_g[:, 2] >= y_min) & (mesh_pos_g[:, 2] <= y_max) &
            (mesh_pos_g[:, 1] >= z_min) & (mesh_pos_g[:, 1] <= z_max)
        )
        x_mask_left = (mesh_pos_g[:, 0] >= x_min) & (mesh_pos_g[:, 0] <= x_half)
        x_mask_right = (mesh_pos_g[:, 0] > x_half) & (mesh_pos_g[:, 0] <= x_max)

        final_left = yz_mask & x_mask_left
        final_right = yz_mask & x_mask_right

    # Lokale Bereiche für linken und rechten Nahtübergang
    vm_label_left = vm_label_g[final_left]
    vm_label_right = vm_label_g[final_right]
    vm_pred_left = vm_pred_g[final_left]
    vm_pred_right = vm_pred_g[final_right]
    # Maxwerte und Threashold berechnen
    MAX_vm_label_left = vm_label_left.max()
    MAX_vm_label_right = vm_label_right.max()
    threshold_left = MAX_vm_label_left - boundary_max
    threshold_right = MAX_vm_label_right - boundary_max
    # Masken für die Auswahl der Knoten welche die PINN-Bedingung erfüllen
    max_stresses_mask_left = vm_label_left >= threshold_left
    max_stresses_mask_right = vm_label_right >= threshold_right
    # Knoten mit den maximalen VM-Spannungen im Nahtübergang
    vm_max_label_left = vm_label_left[max_stresses_mask_left]
    vm_max_label_right = vm_label_right[max_stresses_mask_right]
    vm_max_pred_left = vm_pred_left[max_stresses_mask_left]
    vm_max_pred_right = vm_pred_right[max_stresses_mask_right]
    diff_max_left = (vm_max_label_left - vm_max_pred_left)
    diff_max_right = (vm_max_label_right - vm_max_pred_right)
    # Berechnet betragsweisen MRE
    mre_max_left = (diff_max_left.abs() / vm_max_label_left.abs().mean() * 100).mean()
    mre_max_right = (diff_max_right.abs() / vm_max_label_right.abs().mean() * 100).mean()
    mre_max = 0.5 * (mre_max_right + mre_max_left)
    mre_max_std = 0.5 * (diff_max_right.std() + diff_max_left.std())
    if plot_mode == 'MRE_max':  # Ausgabe für regression plot
        return {
            "left": [vm_max_label_left, vm_max_pred_left],
            "right": [vm_max_label_right, vm_max_pred_right],
        }
    elif plot_mode == 'selection':  # Ausgabe für Knotenselektions plot
        max_stresses_mask_l = vm_label_g >= threshold_left
        max_stresses_mask_r = vm_label_g >= threshold_right
        return max_stresses_mask_l, max_stresses_mask_r
    else:  # Ausgabe für Modellevaluierung
        return [mre_max.item(), mre_max_std.item()]


# ---------------------------------------------------------------------------
# Normierte Darstellung (21x11 Matrix der Nahtübergänge)
# ---------------------------------------------------------------------------

def strip_outer_layers_with_masks(pos, edge_index, stress, gnn_stress,
                                  layers=1, neighbors_expected=6):
    """
    Entfernt äußere Knoten der FEM-Simulatioen
    Rückgabe:
        Daten ohne die äußeren Knoten und updated edge_index
            P, S, G, edge_index_new
        masks = Liste von Bool-Masken, je eine pro Iteration (bezogen auf P vor dem Entfernen)
    """
    P = pos.copy()
    S = stress.copy()
    G = gnn_stress.copy()
    send = edge_index[0].astype(np.int64).copy()
    recv = edge_index[1].astype(np.int64).copy()
    masks = []  # speichert outer_mask je Iteration

    for _ in range(layers):
        N = P.shape[0]
        if N == 0:
            break

        # --- aktuelle gültige Kanten filtern
        edges = np.column_stack((send, recv))
        valid = (
            (edges[:, 0] >= 0) & (edges[:, 0] < N) &
            (edges[:, 1] >= 0) & (edges[:, 1] < N) &
            (edges[:, 0] != edges[:, 1])
        )
        edges = edges[valid]

        # ungerichtet & eindeutig
        e_sorted = np.sort(edges, axis=1)
        e_uniq = np.unique(e_sorted, axis=0)

        # Anzahl der Nachbarnknoten bestimmen
        neighbors = [set() for _ in range(N)]
        for a, b in e_uniq:
            neighbors[a].add(b)
            neighbors[b].add(a)
        neighbor_counts = np.array([len(s) for s in neighbors], dtype=np.int64)

        # Äußerer Knoten wenn weniger als 6 Nachbarsknoten
        outer_mask = neighbor_counts < neighbors_expected
        masks.append(outer_mask.copy())
        keep_mask = ~outer_mask

        # Knotenindezes remappen, damit die Knotenindezes wieder stimmen
        new_index = -np.ones(N, dtype=np.int64)
        new_index[keep_mask] = np.arange(keep_mask.sum(), dtype=np.int64)

        # Ungenutze Kanten löschen und Knotenindezes anpassen
        keep_edge = keep_mask[edges[:, 0]] & keep_mask[edges[:, 1]]
        edges_kept = edges[keep_edge]
        if edges_kept.size == 0:
            P = P[keep_mask]
            S = S[keep_mask]
            G = G[keep_mask]
            send = recv = np.array([], dtype=np.int64)
            break

        edges_remap = np.column_stack([new_index[edges_kept[:, 0]], new_index[edges_kept[:, 1]]])
        dir_edges = np.vstack([edges_remap, edges_remap[:, ::-1]])
        dir_edges = np.unique(dir_edges, axis=0)
        send = dir_edges[:, 0]
        recv = dir_edges[:, 1]

        # Knoten reduzieren
        P = P[keep_mask]
        S = S[keep_mask]
        G = G[keep_mask]

    edge_index_new = np.vstack([send, recv]).astype(np.int64)
    return P, S, G, edge_index_new, masks


def norm_plot(ds, gnn_stress, side_switch=False, tiefe=0, c_mode=None):
    """
    Erstellt die 21x11 Matrix für die normierte Darstellung im Nahtübergang.
    ds: FEM-Simulation im Datensatz
    gnn_stress: Vorhersage des GNNs (Nx6)
    side_switch: False = rechter Nahtübergang
    tiefe: Tiefer der Knotenebene
    c_mode: 'MREu'=return MRE | 'mises_real'=return VM-Spannungen
    """
    # Daten einlesen und ggf. umwandeln
    pos = ds.mesh_pos[:, [0, 2, 1]]
    node_flags = ds.x
    stress = ds.stress
    if gnn_stress is None:
        gnn_stress = stress

    pos = pos.numpy() if isinstance(pos, torch.Tensor) else pos
    pos_old = pos.copy()
    stress = stress.numpy() if isinstance(stress, torch.Tensor) else stress
    stress_old = stress.copy()
    gnn_stress = gnn_stress.numpy() if isinstance(gnn_stress, torch.Tensor) else gnn_stress
    node_flags = node_flags.numpy() if isinstance(node_flags, torch.Tensor) else node_flags

    try:
        if tiefe != 0:
            # Entfernt die äußeren Knoten
            pos, stress, _, edge_index, _ = strip_outer_layers_with_masks(
                pos=pos, edge_index=ds.edge_index.numpy(), stress=stress,
                gnn_stress=gnn_stress, layers=tiefe, neighbors_expected=6)
        _, mises_stress, _ = get_mises(stress, stress)

        # Erzeugt bedingung für die Außenknoten. Äußere Knoten haben <= 5 Kanten,
        # innere Knoten haben 6 Kanten
        if tiefe == 0:
            corner_flags = ds.x[:, 4] == 1
            corner_flags = corner_flags.numpy()
        else:
            send = edge_index[0]
            recv = edge_index[1]
            num_nodes = len(pos)

            # Für jeden Knoten alle Nachbarn sammeln
            neighbors = [[] for _ in range(num_nodes)]
            for s, r in zip(send, recv):
                neighbors[s].append(r)
                neighbors[r].append(s)  # falls ungerichtet, Mesh-Adjazenz ist bidirektional

            # Anzahl **eindeutiger** Nachbarn je Knoten
            neighbor_counts = np.array([len(set(nbrs)) for nbrs in neighbors])
            corner_flags = neighbor_counts < 6

        # Fehler wenn die FEM-Simulationen mehr als 21 Knoten in der breite haben
        if pos[:, 1].max() > 20:
            raise ValueError("Mehr als 20 mm")
        len_x = pos[:, 0].max()

        # 1. Auswahl der oberen Oberflächenknoten auf Ebene y = 10-+0.5
        if side_switch:
            mask_x = (pos[:, 0] > len_x * 0.5) & (pos[:, 0] < (len_x * 0.5 + 20))
        else:
            mask_x = (pos[:, 0] < len_x * 0.5) & (pos[:, 0] > (len_x * 0.5 - 20))
        mask_z = pos[:, 2] > 5
        temp_mask = corner_flags & mask_x & mask_z
        idx_max_stress = mises_stress[temp_mask].argmax()
        pos_filtered_1 = pos[temp_mask]
        [max_x, _, max_z] = pos_filtered_1[idx_max_stress]
        mask_y = (pos[:, 1] < 10.5) & (pos[:, 1] > 9.5)
        mask_x_2 = (pos[:, 0] < max_x + 5) & (pos[:, 0] > max_x - 5)
        pos_sel_y_plane = pos[mask_y & mask_x_2 & corner_flags & mask_z]
        arr_sorted = pos_sel_y_plane[np.argsort(pos_sel_y_plane[:, 0])]
        pos = pos_old
        stress = stress_old
        _, mises_stress, mises_gnn = get_mises(stress, gnn_stress)

        if c_mode == 'MREu':
            diff_mises = ((mises_stress - mises_gnn) / mises_stress) * 100
        elif c_mode == 'mises_real':
            diff_mises = mises_stress

        # 2. Feinere Knotenauswahl nach Schema: 1 grobes Element - 8 feines E. - 1 grobes E.
        dx = np.diff(arr_sorted[:, 0])
        threshold = dx.min() * 2
        is_small = dx < threshold
        pads = np.r_[False, is_small, False]
        edges = np.flatnonzero(pads[1:] != pads[:-1])
        runs = edges.reshape(-1, 2)
        lengths = runs[:, 1] - runs[:, 0]
        best = np.argmax(lengths)
        s, e = runs[best]
        middle = arr_sorted[s: e + 1]
        left_idx = max(s - 1, 0)
        right_idx = min(e + 1, arr_sorted.shape[0] - 1)
        first_one = arr_sorted[left_idx:left_idx + 1]
        last_one = arr_sorted[right_idx:right_idx + 1]
        selected_x_and_z_vals = np.vstack([first_one, middle, last_one])  # [1,11]

        # 3. extrudiert den Pfad entlang der y-Achse
        tol = 1e-4
        mask_old = np.zeros(len(pos), dtype=bool)
        for (tx, _, tz) in selected_x_and_z_vals:
            mask_new = (
                (np.abs(pos[:, 0] - tx) < tol) &
                (np.abs(pos[:, 2] - tz) < tol)
            )
            mask_old = mask_old | mask_new

        # 4. Finale Knotenmaske für die 21x11 Matrix
        selected_pos_final = pos[mask_old]
        selected_stress_final = diff_mises[mask_old]  # len(21*11)

        # Listenform -> 21x11 Matrix
        order = np.lexsort((selected_pos_final[:, 0], selected_pos_final[:, 1]))
        pos_sorted = selected_pos_final[order]
        stress_sorted = selected_stress_final[order]
        xs = np.unique(pos_sorted[:, 0])
        ys = np.unique(pos_sorted[:, 1])
        nx = len(xs)
        ny = len(ys)
        stress_grid = np.full((ny, nx), np.nan)
        # Werte ins Raster legen
        for p, s in zip(pos_sorted, stress_sorted):
            ix = np.where(xs == p[0])[0][0]
            iy = np.where(ys == p[1])[0][0]
            stress_grid[iy, ix] = s
        # Dreht für den linken Nahtübergang die Matrix
        if side_switch:
            stress_grid = np.fliplr(stress_grid)
            stress_grid = np.flipud(stress_grid)

        return stress_grid
    except ValueError as e:
        print('Mehr als 20mm')
        return np.zeros((2, 2))
    except Exception as e:
        print(f'Fehler: {e}. In: {ds.name}')
