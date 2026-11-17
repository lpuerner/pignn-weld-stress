"""
Berechnung der Evaluationsmetriken (R², RAE, MAE, MRE_max) für trainierte GNN-Modelle.
Die Ergebnisse werden in die jeweilige <ID>_model_results.xlsx geschrieben (Sheets 'eval_2' und 'extrema').
"""
import pandas as pd
from types import SimpleNamespace
import utils as u
import json
import os
import torch
import random
import numpy as np
import utils_MG8 as uMG

root_dir = './'
KI_root = 'KI/'                  # Ordner mit den Datensätzen
best_model_root = 'best_model/'  # Ordner mit den trainierten Modellen

# Liest Datenbank ein und wählt Modelle aus die Evaluiert werden sollen.
# Alternativ: id_list = ['d85b5493', ...]
df_db = pd.read_excel(root_dir + 'DB/df_11-16.xlsx')
df_db['date_parsed'] = pd.to_datetime(df_db['date'] + '/2024', format='%m/%d - %H:%M:%S/%Y')
id_list = df_db[(df_db['date_parsed'].dt.month == 11) & (df_db['date_parsed'].dt.day >= 15)]['id']
print(len(id_list))

# Wenn das Modell nicht auf dem eigentlichen test_datenset getestet werden soll
costum_test_dataset = False
costum_test_dataset_path = root_dir + 'KI/t_weld_v1.pt'

step_1 = 1
# Iteriert über alles zu evaluierenden Modelle
for id in id_list:
    try:
        print(f'{step_1}/{len(id_list)}: {id}')
        ID = id

        # Überspring Modelle die bereits evaluiert wurden oder die nicht lang genug trainiert haben
        try:
            df_eval = pd.read_excel(root_dir + best_model_root + ID + '_model_results.xlsx', sheet_name="eval")
            if "mre_max_std" in df_eval.columns:
                print(f'Skips: {ID}. Reason: Already calculated.')
                step_1 += 1
                continue

            df_metrics = pd.read_excel(root_dir + best_model_root + ID + '_model_results.xlsx', sheet_name="metrics")
            if df_metrics.shape[0] < 10:
                print(f'Skips: {ID}. Reason: Not enough epochs (<10).')
                step_1 += 1
                continue
        except Exception:
            pass

        try:
            df_eval = pd.read_excel(root_dir + best_model_root + ID + '_model_results.xlsx', sheet_name="eval_2")
            if "mre_max_2_mean" in df_eval.columns:
                print(f'Skips: {ID}. Reason: Already calculated.')
                step_1 += 1
                continue
        except Exception:
            pass

        PATH = os.path.join(root_dir, best_model_root + ID + '_model.pt')
        if not os.path.isfile(PATH):
            print(f'Skips: {id}. Reason: No best model.')
            step_1 += 1
            continue

        # Läd Modell config
        with open(root_dir + best_model_root + ID + '_model_config.json') as f:
            config = json.load(f)
        args = SimpleNamespace(**config[0]["args"])
        test_loader_list = config[0]["test_loader"]
        hash_id = config[0]["id"]

        # Läd Datensatz und seperiert diese in Traing und Testing
        try:
            path_ds = root_dir + KI_root + args.dataset
            dataset = torch.load(path_ds, weights_only=False)
        except Exception:
            path_ds = root_dir + KI_root + args.dataset + '.pt'
            dataset = torch.load(path_ds, weights_only=False)

        test_sets = []
        for i in range(len(dataset)):
            if dataset[i].name in test_loader_list:
                test_sets.append(i)

        train_sets = [i for i in range(len(dataset)) if i not in test_sets]
        test_dataset = [dataset[i] for i in test_sets]
        train_dataset = [dataset[i] for i in train_sets]
        stats_list = uMG.get_stats(dataset)

        # Initialisiert speziellen Datensatz
        if costum_test_dataset:
            ds = torch.load(costum_test_dataset_path, weights_only=False)
            test_dataset = ds[:50]
            stats_list = uMG.get_stats(dataset)

        # Läd Modell
        args.device = torch.device('cpu')
        num_node_features = dataset[0].x.shape[1]
        num_edge_features = dataset[0].edge_attr.shape[1]
        num_classes = 6
        model = uMG.GNN(num_node_features, num_edge_features, args.hidden_dim, num_classes, args).to(args.device)
        print(PATH)
        model.load_state_dict(torch.load(PATH, map_location=args.device))
        model.eval()
        device = args.device

        torch.manual_seed(args.seeds)
        random.seed(args.seeds)
        np.random.seed(args.seeds)

        [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
        (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
            mean_vec_x.to(device), std_vec_x.to(device), mean_vec_edge.to(device),
            std_vec_edge.to(device), mean_vec_y.to(device), std_vec_y.to(device))

        # Initalisiert die DataFrames zum speichern
        cols = ['sxx', 'syy', 'szz', 'sxy', 'syz', 'szx', 'mises']
        R2 = pd.DataFrame(columns=cols)
        Error = pd.DataFrame(columns=cols)
        MAE = pd.DataFrame(columns=cols)
        step = 1
        cols_extrema = ['mre_max_2', 'mre_max_std_2']
        DF_extrema = pd.DataFrame(columns=cols_extrema)

        # Evaluiert jede Simulation in den Testdaten
        for data in test_dataset:
            with torch.no_grad():
                pred_stress_normalized = model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
                pred_stress = uMG.unnormalize(pred_stress_normalized, mean_vec_y, std_vec_y)
                diff_stress = data.stress - pred_stress

                # Iteriert über Spannungskomponenten
                r2_vals_stresses = []
                error_stresses = []
                mae_stresses = []

                for i in range(6):
                    r2_vals_stresses.append(u.get_R22(data.stress[:, i].numpy(), diff_stress[:, i].numpy()))
                    error_stresses.append(u.rae(data.stress[:, i].numpy(), pred_stress[:, i].numpy()))
                    mae_stresses.append(u.get_mae(data.stress[:, i].numpy(), pred_stress[:, i].numpy()))

                # Berechnet Von-Mises-Spannung und hängt Ergebnisse an
                mises_r2, mises_fem, mises_gnn = u.get_mises(data.stress, pred_stress)
                error_mises = u.rae(mises_gnn, mises_fem)
                r2_vals_stresses.append(mises_r2)
                error_stresses.append(error_mises)
                mae_stresses.append(u.get_mae(mises_fem, mises_gnn))
                # Berechnet die MRE_max Metrik für einen einstellung von 0,05 MPa
                mre_results = u.find_extreme_nodes_by_abs_and_two_boxes(data, mises_fem, mises_gnn, 0.05, 0.05)

                R_append = pd.DataFrame([r2_vals_stresses], columns=cols)
                error_append = pd.DataFrame([error_stresses], columns=cols)
                MAE_append = pd.DataFrame([mae_stresses], columns=cols)
                extrema_vals_append = pd.DataFrame([mre_results], columns=cols_extrema)

                R2 = pd.concat([df for df in [R2, R_append] if not df.empty])
                Error = pd.concat([df for df in [Error, error_append] if not df.empty])
                MAE = pd.concat([df for df in [MAE, MAE_append] if not df.empty])
                DF_extrema = pd.concat([df for df in [DF_extrema, extrema_vals_append] if not df.empty])

            step += 1

        # Finales DataFrame welches gespeichert wird
        stats_df = pd.DataFrame({
            'Werte': cols,
            'r2_mean': R2.mean(),
            'r2_std': R2.std(),
            'rae_mean': Error.mean(),
            'rae_std': Error.std(),
            'mae_mean': MAE.mean(),
            'mae_std': MAE.std(),
            'mre_max_2_mean': DF_extrema["mre_max_2"].mean(),
            'mre_max_std_2_mean': DF_extrema["mre_max_std_2"].mean(),
            'mre_max_2_std': DF_extrema["mre_max_2"].std(),
            'mre_max_std_2_std': DF_extrema["mre_max_std_2"].std(),
        })
        xls_path = os.path.join(root_dir, best_model_root + ID + '_model_results.xlsx')

        # Speichert DataFrame mit den seperaten Ergebnissen der Simulationen
        name_list = []
        for data in test_dataset:
            name_list.append(data.name)
        DF_extrema["name"] = name_list
        DF_extrema['r2'] = R2['mises'].values

        # Speichert DataFrames als .xlsx im jeweiligen Modell-result file
        if costum_test_dataset:
            pass
        else:
            u.write_df_to_excel_sheet(stats_df, xls_path, 'eval_2')
            u.write_df_to_excel_sheet(DF_extrema, xls_path, 'extrema')
        step_1 += 1

    except Exception as e:
        print(f'Error in {id}: {e}')
        step_1 += 1
