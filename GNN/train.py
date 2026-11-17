"""
Minimal Code um das Training eines GNNs zu starten.
"""
import random
import torch
import numpy as np
import os
from datetime import datetime
import utils_MG8 as uMG  # importieren des GNNs

root_dir = './'  # Arbeitsverzeichniss (enthält KI/ und best_model/)
checkpoint_dir = os.path.join(root_dir, 'best_model')
os.makedirs(checkpoint_dir, exist_ok=True)

dataset_str = "3D_data_butt_3D_v13_1.pt"
dataset = torch.load(os.path.join(root_dir, 'KI', dataset_str), weights_only=False)
train_size = round(len(dataset) * 0.8)
test_size = len(dataset) - train_size


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


# Hyperparameter der KI
seedss = 5
for args in [
        {'model_type': 'v8',             # Version der GNN Architektur
         'num_layers': 20,               # Anzahl GN-Blöcke
         'batch_size': 5,                # Batchgröße
         'hidden_dim': 32,               # hidden dimension h
         'epochs': 600,                  # max. Epochs
         'seeds': seedss,                # Seed für Einstellung der Zufälligkeiten
         'opt': 'adam',                  # Ausgewähler Optimierer
         'weight_decay': 5e-4,           # weight decay des Optimierers
         'lr': 0.001,                    # Lernrate des Optimierers
         'train_size': train_size,
         'test_size': test_size,
         'device': 'cuda',               # GNN wird auf cpu oder nvidia gpu trainiert
         'shuffle': True,                # Wenn True: Datensat wird gemischt
         'save_best_model': True,        # Wenn True: Beste KI-Modelle werden gespeichert
         'checkpoint_dir': checkpoint_dir + '/',  # Speichertort der Ergebnisse
         'dataset': dataset_str,         # Name des Datensatzes
         'loss_function': 'MSE',         # Loss-Funktion des Basis-GNNs
         'mode_loss': 'beides',          # Loss-Funktion Einstellung für die PINN-Terme
         'drop_mlp': 0.0,                # Normales Dropoutlayer 0%
         'drop_enc': 0.0,                # Normales Dropoutlayer 0%
         'drop_edge': 0.05,              # Kantendropout 5%
         'alpha_ki': 5,                  # Verstärkungsfaktor der PINN-Terme
         },
]:
    args = objectview(args)

    # Seed festlegen und Datensatz mischen
    torch.manual_seed(seedss)
    random.seed(seedss)
    np.random.seed(seedss)
    if args.shuffle:
        random.seed(args.seeds)
        random.shuffle(dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    uMG.clear_cache()
    args.device = str(device)
    stats_list = uMG.get_stats(dataset)

    now = datetime.now()
    print(f'Simulation start |Device: {args.device} |Date: {now.strftime("%m-%d %H:%M:%S")}')
    try:
        test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader, ID = uMG.train(
            dataset, device, stats_list, args)

        now2 = datetime.now()
        elapsed_time = now2 - now
        print(f'Min test loss: {min(test_losses):.3f} |Min train loss: {min(losses):.3f} '
              f'|Epochs: {len(losses)} |Elapsed time: {elapsed_time}\n')
    except Exception as e:
        print(f'Error: {e} \n')
