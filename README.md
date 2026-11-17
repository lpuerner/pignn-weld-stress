# Physics-Informed Graph Neural Networks zur Spannungsvorhersage in geschweißten Strukturen

Code zur Masterarbeit *„Physics-Informed Graph Neural Networks zur Spannungsvorhersage in geschweißten Strukturen“*
(Hochschule München, Fakultät 03, Wintersemester 2025/26) und zum Paper
*„Physics-Informed Graph Neural Networks for Stress Prediction in Welded Structures“* (ESREL 2026).

Ein encoder-process-decoder-GNN sagt die Spannungen (6 Komponenten) an den Knoten eines FE-Netzes voraus.
Zusätzliche physikinformierte Loss-Terme gewichten die maximale Von-Mises-Spannung im Nahtübergang
und die Symmetrie zwischen linkem und rechtem Nahtübergang. Die Trainingsdaten stammen aus automatisierten
ANSYS-APDL-Simulationen einer Stumpfnaht und einer Kehlnaht (T-Stoß) mit variierter Nahtgeometrie.

## Struktur

```
APDL/
  butt_weld_master_3D.mac    APDL-Masterskript Stumpfstoß (Geometrie, Vernetzung, RB, Lösung, Export)
  t_weld_master.mac          APDL-Masterskript T-Stoß / Kehlnaht
  write_results_3D.inp       Schreibt Knoten- und Elementmatrix als CSV (muss im tmp-Ordner liegen)
  FEM_automatisierung.py     Latin-Hypercube-Sampling der Geometrieparameter und Batch-Aufruf von MAPDL
GNN/
  fem_to_graph.py            FEM-Ergebnisse (node_*.csv / elem_*.csv) -> PyTorch-Geometric-Graphen (.pt)
  balance_dataset.py         Reduziert den Datensatz auf eine Gleichverteilung der max. Kerbspannung
  utils_MG8.py               GNN-Architektur, PINN-Loss, Training und Test
  utils.py                   Metriken (R², RAE, MAE, MRE_max) und normierte 21x11-Darstellung
  train.py                   Minimalbeispiel zum Starten eines Trainings
  evaluate.py                Evaluationsmetriken für trainierte Modelle
```

## Ablauf

1. **FEM-Daten erzeugen** (Windows, ANSYS MAPDL 2023 R2):
   `APDL/write_results_3D.inp` in den Ordner `tmp/` kopieren, Pfad zu `MAPDL.exe` in
   `APDL/FEM_automatisierung.py` prüfen, dann `python APDL/FEM_automatisierung.py`.
   Platzhalter mit der Endung `_new` (z. B. `r_new`, `t_new`) und `dir_path` im APDL-Masterskript
   werden vom Skript ersetzt. Die Ergebnisse landen in `solfiles/`.
2. **Graphen erzeugen:** `cd gnn && python fem_to_graph.py` (Pfade oben im Skript anpassen).
3. **Optional Datensatz balancieren:** `python balance_dataset.py`
4. **Training:** `python train.py`. Modelle, Loss-Verläufe (`*_results.xlsx`) und Konfiguration
   (`*_model_config.json`) werden mit einer 8-stelligen Modell-ID unter `best_model/` gespeichert.
5. **Evaluation:** `python evaluate.py`
6. **Hyperparameteroptimierung:** `python bayes_opt.py`

## Installation

```bash
pip install -r requirements.txt
```

`torch_scatter` muss passend zur installierten PyTorch-/CUDA-Version installiert werden
(siehe https://data.pyg.org/whl/).

## Hinweise zum Code

Der Code entspricht dem Anhang der Masterarbeit. Für die Veröffentlichung wurden nur kleine Anpassungen gemacht:

- Absolute Pfade (Workstation, Rechencluster) durch relative Pfade ersetzt..
- `evaluate.py` lädt die Modelle über `utils_MG8.GNN` (im Anhang `utils_MeshGraphnetv7.MeshGraphNet`).

Die Datensätze und trainierten Modelle sind wegen ihrer Größe nicht im Repository enthalten.

## Lizenz

Der Code steht unter der [MIT-Lizenz](LICENSE).
