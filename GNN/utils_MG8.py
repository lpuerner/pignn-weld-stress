import torch_scatter
import os
import torch
import gc
import random
import torch
import pandas as pd
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader
import numpy as np
import time
import torch.optim as optim
from tqdm import trange
import pandas as pd
import copy
import os
import hashlib
import re
from datetime import datetime
from torch.nn import Dropout
from torch_geometric.utils import dropout_edge
import json

def normalize(to_normalize,mean_vec,std_vec):
    return (to_normalize-mean_vec)/std_vec

def unnormalize(to_unnormalize,mean_vec,std_vec):
    return to_unnormalize*std_vec+mean_vec

def get_stats(data_list):
    '''
    Ermittelt die Werte für die Z-Normalisierung vom Datensatz
    '''
    # Mittelwert und STD der Knotenfeatures
    mean_vec_x=torch.zeros(data_list[0].x.shape[1:])
    std_vec_x=torch.zeros(data_list[0].x.shape[1:])

    # Mittelwert und STD der Kantenfeatures
    mean_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])

    # Mittelwert und STD der Spannungen
    mean_vec_y=torch.zeros(data_list[0].stress.shape[1:])
    std_vec_y=torch.zeros(data_list[0].stress.shape[1:])

    # Zähler und kleine zahl definieren
    eps=torch.tensor(1e-8)
    num_accs_x = 0
    num_accs_edge=0
    num_accs_y=0

    # Durch den Datensatz iterieren
    for dp in data_list:
        mean_vec_x+=torch.sum(dp.x,dim=0)
        std_vec_x+=torch.sum(dp.x**2,dim=0)
        num_accs_x+=dp.x.shape[0]

        mean_vec_edge+=torch.sum(dp.edge_attr,dim=0)
        std_vec_edge+=torch.sum(dp.edge_attr**2,dim=0)
        num_accs_edge+=dp.edge_attr.shape[0]

        mean_vec_y+=torch.sum(dp.stress,dim=0)
        std_vec_y+=torch.sum(dp.stress**2,dim=0)
        num_accs_y+=dp.stress.shape[0]

    mean_vec_x = mean_vec_x/num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x/num_accs_x - mean_vec_x**2),eps)
    mean_vec_edge = mean_vec_edge/num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge/num_accs_edge - mean_vec_edge**2),eps)
    mean_vec_y = mean_vec_y/num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y/num_accs_y - mean_vec_y**2),eps)
    mean_std_list=[mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y]
    return mean_std_list

def clean_cuda_gpu():
    """
    Löscht cuda cache
    """
    gc.collect()
    torch.cuda.empty_cache()

# Alias: train.py ruft uMG.clear_cache() auf
clear_cache = clean_cuda_gpu

class GNN(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False):
        super(GNN, self).__init__()
        """
        Diese Klasse implementiert ein Graph Neural Network nach dem Aufbau von 
        DeepMind (2021)[3] und Jin (2023)[14]. Das Modell besteht aus:

        1) Encodern für Knoten- und Kantenfeatures
        2) Mehreren Message-Passing-Schichten (GN-Blöcke)
        3) Einem Decoder zur Vorhersage der Knotenoutputgrößen

        Zusätzlich unterstützt das Modell Normalisierung der Eingabefeatures
        sowie optionales Kantendropout während des Trainings.
        """
        self.num_layers = args.num_layers
        self.drop_mlp = args.drop_mlp
        self.drop_edge = args.drop_edge

        # Knoten- und Kantenencoder
        # Bsp. für Knoten: Input [N x 5] -> Output: [N x hidden_dim]
        self.node_encoder = Sequential(Linear(input_dim_node , hidden_dim),
                              ReLU(),
                              Dropout(p=args.drop_enc),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear( input_dim_edge , hidden_dim),
                              ReLU(),
                              Dropout(p=args.drop_enc),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim)
                              )
        #baut GN-Blocks
        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))

        # Knotendecoder
        self.decoder = Sequential(Linear( hidden_dim , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, output_dim)
                              )

    def build_processor_model(self):
        def layer_fn(in_ch, out_ch):
            return ProcessorLayer(in_ch, out_ch, drop_mlp=self.args.drop_mlp)
        return ProcessorLayer

    def forward(self,data,mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = normalize(x,mean_vec_x,std_vec_x)
        edge_attr=normalize(edge_attr,mean_vec_edge,std_vec_edge)

        # 1: Features encoden
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # 2: Kantendropout
        if self.training and self.drop_edge > 0:
                edge_index, edge_mask = dropout_edge(edge_index,p=self.drop_edge, force_undirected=False,training=True)
                edge_attr = edge_attr[edge_mask]
                del edge_mask

        # 3: Erstellt GN-Blöcke
        for i in range(self.num_layers):
            x,edge_attr = self.processor[i](x,edge_index,edge_attr)

        # 4: Decoding der Knotenembeddings
        return self.decoder(x)

    def loss(self,pred,  inputs,  mean_vec_stress, std_vec_stress, loss_type, alpha_ki: float = 1.0,  mode_loss = 'beides'):
        """
        Berechnet die Loss-Werte für das GNN.
        Die Funktion umfasst zwei Teile:
        1) Standard-Loss (MSE/RMSE) für alle Knoten.
        2) Optional: PINN-Terme zur verbesserten Vorhersage der Spannungen.
            a) PINN-Term der die maximalen Von-Mises-Spannungen im Nahtübergang gewichtet
            b) PINN-Term der die Differenz zweier symmetrischer Nahtübergänge minimiert
        """
        device = pred.device
        # Maske für relevante Knoten
        cls = torch.argmax(inputs.x[:, 3:], dim=1)
        loss_mask = torch.logical_or((cls == 0), (cls == 1))
        stress_labels = normalize(inputs.stress, mean_vec_stress, std_vec_stress)

        # Unterschiedliche Loss-Funktionen für das Basis-GNN
        if loss_type == 'RMSE':
            stress_error = torch.sum((stress_labels - pred) ** 2, dim=1)
            stress_loss = torch.sqrt(torch.mean(stress_error[loss_mask]))
        elif loss_type == 'MSE':
            stress_error = torch.sum((stress_labels - pred) ** 2, dim=1)
            stress_loss = torch.mean(stress_error[loss_mask])
        elif loss_type == 'RMSE3':
            stress_error = torch.sum(torch.abs(stress_labels - pred) ** 3, dim=1)
            stress_loss = torch.sqrt(torch.mean(stress_error[loss_mask]))
        else:
            raise ValueError(f"Unbekannter loss_type: {loss_type}")

        if mode_loss == 'normal':
            total_loss = stress_loss
            ki_loss = 0
            return total_loss, [total_loss.detach().cpu().item(), 0]
        else:
            # Berechnung der Loss-Werte aus den PINN-Termen
            pred_phys = unnormalize(pred, mean_vec_stress, std_vec_stress)
            labels_phys = unnormalize(stress_labels, mean_vec_stress, std_vec_stress)

            # wahre und vorhergesagte Von-Mises-Spannung
            vm_pred_nodes = von_mises_from_components(pred_phys)  # [N]
            vm_label_nodes = von_mises_from_components(labels_phys)  # [N]

            ki_losses = []
            gnn_max_diffs = []

            #Iteriert pro Graph im Dataloader
            ptr = inputs.ptr
            num_graphs = int(ptr.numel()) - 1
            for g in range(num_graphs):
                start, end = int(ptr[g]), int(ptr[g + 1])

                mesh_pos_g = inputs.mesh_pos[start:end]  # _g = globaler index
                vm_pred_g = vm_pred_nodes[start:end]
                vm_label_g = vm_label_nodes[start:end]
                mask_g = loss_mask[start:end]

                # T-Stoß maske für die Knotenauswahl im Nahtübergang
                if mesh_pos_g[:, 1].max() > 50:
                    h = mesh_pos_g[:, 1].max() - 50

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
                else:
                    # Stumpfstoß maske für die Knotenauswahl im Nahtübergang
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
                threshold_left = MAX_vm_label_left - 0.15
                threshold_right = MAX_vm_label_right - 0.15
                # Masken für die Auswahl der Knoten welche die PINN-Bedingung erfüllen
                max_stresses_mask_left = vm_label_left >= threshold_left
                max_stresses_mask_right = vm_label_right >= threshold_right
                vm_max_label_left = vm_label_left[max_stresses_mask_left]
                vm_max_label_right = vm_label_right[max_stresses_mask_right]
                vm_max_pred_left = vm_pred_left[max_stresses_mask_left]
                vm_max_pred_right = vm_pred_right[max_stresses_mask_right]

                # MSE berechnung beider Nahtübergänge
                loss_left = torch.mean((vm_max_pred_left - vm_max_label_left) ** 2)
                loss_right = torch.mean((vm_max_pred_right - vm_max_label_right) ** 2)
                ki_losses.append(0.5 * (loss_left + loss_right))

                # Differenz linker und rechter Nahtübergang
                gnn_max_diffs.append(((torch.mean(vm_max_pred_right) - torch.mean(vm_max_pred_left)) ** 2).detach())

            # Loss-Mitteltwert für den Batch
            if len(ki_losses) > 0:
                ki_loss = torch.stack(ki_losses).mean()
            else:
                ki_loss = torch.tensor(0.0, device=device, dtype=pred.dtype)

            # Modus für die Loss-Funktion
            if mode_loss == 'beides':
                ki_loss = alpha_ki * (gnn_max_diffs[-1] + ki_loss)
            elif mode_loss == 'nur_diff_LR':
                ki_loss = alpha_ki * (gnn_max_diffs[-1])
            else: #diff_gnn-fem
                ki_loss = alpha_ki * ki_loss

            ki_loss = torch.clamp(ki_loss, max=1.0)
            total_loss = stress_loss + ki_loss #Finaler-Loss
            return total_loss, [stress_loss.detach().cpu().item(), ki_loss.detach().cpu().item()]

def von_mises_from_components(stresses):
    """
    Berechnet die Von-Mises_Spannung.
    Eingabe in der Spaltenform N x [sx, sy, sz, sxy, syx, sxz]
    """
    s11, s22, s33, s12, s13, s23 = stresses.unbind(dim=1)
    vm = torch.sqrt(0.5*((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3*(s12**2 + s13**2 + s23**2))
    return vm

class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, drop_mlp=0.0,  **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs )
        """
        Message-Passing-Layer/GN-Block des GNNs.
        1) Update der Kanten
        2) Aggregieren der Knoten über die Kanten
        3) Update der Knoten
        """
        # 3*h weil input = [x_i, x_j, edge_attr] (x_i: Senderknoten features | x_j: Empfängerknoten features | edge_attr: Kantenembedings)
        self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   ReLU(),
                                   Dropout(p=drop_mlp),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        # 2*h, weil input = [x,node_aggr] (x="Alte Knotenembeddings", node_aggr = Updated Knotenembeddings)
        self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                                   ReLU(),
                                   Dropout(p=drop_mlp),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Setzt die Parameter für die Update MLPs zurück
        """
        for layer in self.edge_mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()

        for layer in self.node_mlp:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Startet den GN-Block:
        1. propagate()-> message() -> aggregate()
        2. Knoten-Update durch Knoten_mlp und Residuum.

        Eingaben:
        - x: Knotenembeddings [N, in_channels]
        - edge_index: Kantenliste [2, E]
        - edge_attr: Kantenfeatures [E, in_channels]
        """
        dim_size = x.size(0) #Saftey: Incase the highes edge_idx gets droped -> x =/= node_aggr
        node_aggr, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = (dim_size,dim_size))
        h = torch.cat([x,node_aggr],dim=1)        # x = Verkettung, node_aggr = Updated Knoten
        updated_nodes = x + self.node_mlp(h) # Residuum + Updated Knoten
        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        Definiert die Nachricht, die über jede Kante gesendet wird.
        Die Nachricht besteht aus:
        [x_i, x_j, edge_attr] → durch edge_mlp verarbeitet → Residual zu edge_attr.
        edge_attr: [E, in_channels]
        x_i: Features Senderknoten | x_j: Features Empfängerknoten [E, in_channels]
        """
        updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) # [E, 3 * in_channels]
        updated_edges=self.edge_mlp(updated_edges)+edge_attr #Updated Kanten + Verkettung
        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size):
        """
        Aggregiert alle eingehenden Nachrichten pro Knoten (Summation).
        Gibt aggregierte Knotennachrichten und aktualisierte Kanten zurück.
        """
        node_dim = 0
        index = edge_index[0,:].long()
        out = torch_scatter.scatter(updated_edges, edge_index[0,:].long(), dim=node_dim ,dim_size=dim_size, reduce = 'sum')
        return out, updated_edges

def train(dataset, device, stats_list, args):
    """
    Führt den kompletten Trainingsablauf des GNNs aus.
    """
    # Einstellungen early stopping durch moving average window
    moving_avg_window = 40
    min_delta_percent = 0.01

    df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss', 'velo_val_loss', 'times'])
    loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)

    # Generiert Modell-ID
    now = datetime.now()
    args.date = now.strftime("%m/%d - %H:%M:%S")
    params = args.__dict__
    exclude_keys = ['device']
    params_filtered = {k: v for k, v in params.items() if k not in exclude_keys}
    params_serializable = {
        k: (str(v) if isinstance(v, list) else v) for k, v in params_filtered.items()
    }
    param_str = json.dumps(params_serializable, sort_keys=True)
    id_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    model_name = id_hash + '_model'
    print(f'ID: {id_hash}')
    save_config(args, test_loader, id_hash)

    # Statistiken des Datensatzes werden eingelesen
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (mean_vec_x.to(device),
                                                                            std_vec_x.to(device),
                                                                            mean_vec_edge.to(device),
                                                                            std_vec_edge.to(device),
                                                                            mean_vec_y.to(device),
                                                                            std_vec_y.to(device))

    # Initialisiert das GNN
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    num_classes = 6  # GNN sagt 6 Spannungen vorher
    model = GNN(num_node_features, num_edge_features, args.hidden_dim, num_classes,args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # Trainingsschleife
    losses = []
    GNN_losses = []
    sKI_losses = []
    times = []
    test_losses = []
    velo_val_losses = []
    best_test_loss = np.inf
    best_model = None
    sim_start = time.time()
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        start = time.time()
        total_loss = 0
        total_GNN_loss = 0
        total_sKI_loss = 0
        model.train()
        num_loops = 0
        # Iteriert über die mini-batches im Trainingsloader
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()  # setzt Gradienten auf 0
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss, spliced_loss = model.loss(pred, batch, mean_vec_y, std_vec_y, args.loss_function, alpha_ki =  args.alpha_ki, mode_loss = args.mode_loss)
            [GNN_loss, sKI_loss] = spliced_loss
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_GNN_loss += GNN_loss
            total_sKI_loss += sKI_loss
            num_loops += 1
        total_loss /= num_loops
        total_GNN_loss /= num_loops
        total_sKI_loss /= num_loops
        losses.append(total_loss)
        GNN_losses.append(total_GNN_loss)
        sKI_losses.append(total_sKI_loss)
        times.append(time.time() - start)

        test_loss, _ = test(test_loader, device, model, mean_vec_x, std_vec_x, mean_vec_edge,std_vec_edge, mean_vec_y, std_vec_y, args)
        test_losses.append(test_loss.item())

        # Speichern der Ergebnisse pro Epoche
        new_row = pd.DataFrame([{'epoch': epoch,
                                 'train_loss': losses[-1],
                                 'GNN_loss': GNN_losses[-1],
                                 'sKI_loss': sKI_losses[-1],
                                 'test_loss': test_losses[-1],
                                 'times': times[-1]}])
        df = pd.concat([df, new_row], ignore_index=True)

        if (epoch % 10 == 0):
            # Speichert Modell wenn besseres vorliegt
            if df["test_loss"].min() < best_test_loss:
                best_test_loss = df["test_loss"].min()
                best_model = copy.deepcopy(model)

            # Speichert die results xlsx mit den Loss-Werten etc.
            write_df_to_excel_sheet(df, args.checkpoint_dir + model_name +'_results.xlsx' ,'metrics')
            if (args.save_best_model):
                PATH = os.path.join(args.checkpoint_dir, model_name + '.pt')
                torch.save(best_model.state_dict(), PATH)

            # check early stopping Bedingungen
            if (epoch > 250):
                # moving average window
                recent_window = losses[-moving_avg_window:]
                recent_avg = np.mean(recent_window)

                prev_window = losses[-2*moving_avg_window:-moving_avg_window]
                prev_avg = np.mean(prev_window)
                change = abs(recent_avg - prev_avg) / prev_avg

                if change < min_delta_percent:
                    print(f'Training stopped at due to convergence: Epoch {epoch}')
                    write_df_to_excel_sheet(df, args.checkpoint_dir + model_name +'_results.xlsx' ,'metrics')
                    break
            if (time.time() - sim_start)/3600 > 4:
                # zeitbedingung
                print(f'Training stopped at due to 5h limit: Epoch {epoch}')
                write_df_to_excel_sheet(df, args.checkpoint_dir + model_name +'_results.xlsx' ,'metrics')
                break

    PATH = os.path.join(args.checkpoint_dir, model_name + '_best.pt')
    torch.save(best_model.state_dict(), PATH)
    return test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader, id_hash

def test(loader, device, test_model, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, args):
    """
    Berechnet den Loss-Wert für alle FEM-Simulationen im Testdatensatz.
    Iteriert über die Testdaten und berechnet die Loss-Werte anschließend durch Mitteltwertbildung.
    """
    loss = 0
    velo_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss_new, _ = test_model.loss(pred, data, mean_vec_y, std_vec_y, args.loss_function, alpha_ki =  args.alpha_ki, mode_loss = args.mode_loss)
            loss += loss_new
        num_loops += 1
    return loss / num_loops, velo_rmse / num_loops

def write_df_to_excel_sheet(df, excel_path, sheet_name):
    """
    Schreibt DataFrame in Ziel-xlsx auf entsprechendem sheet.
    """
    if os.path.exists(excel_path):
        # Lade bestehende Excel-Datei
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def build_optimizer(args, params):
    """
    Einstellungen für den Optimierer
    """
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    scheduler = None  # Es wird kein Lernraten-Scheduler verwendet
    return scheduler, optimizer

def save_config(args, test_loader, id_hash):
    """
    Speichert für ein GNN folgende Daten in einer .json Datei:
        args bzw. Modellparameter etc
        FEM-Namen im test_loader
        Modell-ID
    """
    config_path = args.checkpoint_dir + id_hash + '_model_config.json'
    args_dict = args.__dict__
    test_loader_list = [i.name for i in test_loader.dataset]

    new_entry = {
        "args":args_dict,
        "test_loader": test_loader_list,
        "id":id_hash
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            data.append(new_entry)
        else:
            data = [data, new_entry]
    else:
        data = [new_entry]

    with open(config_path, "w") as f:
        json.dump(data, f, indent=4)
