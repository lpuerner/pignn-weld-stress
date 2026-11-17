import subprocess
import numpy as np
import pandas as pd
import os
import shutil
from scipy.stats import qmc
import time

ansyspath = r'C:\Program Files\ANSYS Inc\v232\ansys\bin\winx64\MAPDL.exe' # ANSYS exe Pfad
# Projektverzeichnis (bei Bedarf anpassen). Standard: Root des Repositories
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sol_directory = os.path.join(directory, 'solfiles') # Speicherort der FEM-Lösungen
tmp_directory = os.path.join(directory, 'tmp') # Arbeitsverzeichniss in dem APDL arbeitet
jobname = 'file2'
memory = '10096'
reserve = '1024'
inputfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'butt_weld_master_3D.mac') # APDL Masterskript
"""
Ordnerstruktur:
APDL-Masterfile.mac
/tmp
    write_results_3d.inp
    temporäres APDL-Skript
/solfiles
    "Sammlung aller FEM-Lösungsdatein als .csv"
"""


outputfile  = os.path.join(directory, 'outputfile.txt')

def delete_all_files_in_folder(folder_path,exclude_filename):
    """
    Löscht alles außer write_results_3d.inp im tmp_directory
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if filename in exclude_filename:
                continue
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Fehler beim Löschen von {file_path}: {e}")

def sample_generator(parameter_bounardies, sample_number, df_list):
    """
    Erzeugt einen gleichverteilten Parameterraum.
    """
    if len(df_list) == parameter_bounardies.shape[0]:
        bounds = parameter_bounardies
        n_samples = sample_number

        # LHS-Sampler initialisieren
        sampler = qmc.LatinHypercube(d=bounds.shape[0])
        sample_unit = sampler.random(n=n_samples)

        # Auf die Parametergrenzen skalieren
        samples_continuous = qmc.scale(sample_unit, bounds[:, 0], bounds[:, 1])
        # In DataFrame speichern
        df = pd.DataFrame(samples_continuous, columns=df_list)
        df['r_new'] = df['r_new'].round(3)
        df['a_new'] = df['a_new'].round(1)
        df['w_new'] = df['w_new'].round(1)
        df['t_new'] = df['t_new'].round(1)
        df['h_new'] = df['h_new'].round(3)
        return df
    else:
        print('Parameter size not similar')
        return False

def execute_Ansys2(master_txt, variable_series):
    """
    Ändert das APDL-Masterskript und führt die APDL Simulation aus.
    Speichert und verwaltet anschließend die Lösungsdatein.
    """
    try:
        # Bereitet /tmp für neue Simulation vor
        delete_all_files_in_folder(tmp_directory,['write_results_3D.inp','write_results.inp'])

        # Liest APDL-Masterfile
        with open(master_txt, 'r') as file:
            content = file.read()

        # Ersetzt Parameterstrings im APDL-Masterfile
        header_list = variable_series.index.tolist()
        name_str = ''
        for item in header_list:
            content = content.replace(item, str(variable_series[item]))
            name_str = name_str + item[0] + '_' + str(variable_series[item]) + '__'
        content = content.replace('dir_path', "'" + tmp_directory + "'")

        # Erzeugt neues APDL skript mit ausgetauschten variablen
        formated_inputfile = tmp_directory + ('/butt_weld_' + name_str +'.inp')
        with open(formated_inputfile, 'w') as file:
            file.write(content)

        # Ruft ANSYS auf
        callstring = ('\"{}\" -p ansys -dir \"{}\" -j \"{}\" -s read'
                    ' -m {} -db {} -t -b -i \"{}\" -o \"{}\"'
                    ).format(ansyspath,tmp_directory,jobname,memory,reserve,formated_inputfile,outputfile)
        result = subprocess.run(callstring, shell=True, capture_output=True, text=True)

        #Kopiert FEM-Lösungen vom tmp_directory ins sol_directory
        old_node_path = os.path.join(tmp_directory, 'node_result.csv')
        old_elem_path = os.path.join(tmp_directory, 'element_result.csv')
        new_node_path = os.path.join(sol_directory, 'node_' + name_str + '.csv')
        new_elem_path = os.path.join(sol_directory, 'elem_' + name_str + '.csv')
        shutil.copy(old_node_path, new_node_path)
        shutil.copy(old_elem_path, new_elem_path)
        print("Erfolgreiche Simulation: ")
        print(f'r:{variable_series[0]}| a:{variable_series[1]}| w:{variable_series[2]}| t:{variable_series[3]}| h:{variable_series[4]}')

    except:
        print("Fehlerhafte Simulation: ")
        print(f'r:{variable_series[0]}| a:{variable_series[1]}| w:{variable_series[2]}| t:{variable_series[3]}| h:{variable_series[4]} ')


if not os.path.exists(sol_directory):
    os.makedirs(sol_directory)
    print("Lösungsverzeichniss erstellt")

# Geometrieparameterbereiche
bounds = np.array([
    [0.2, 0.4],    #r_k
    [40, 60],   #alpha
    [30,60],  #weld_ang
    [10,30], #tiefe
    [0.1, 0.3] # Überhöhung
])

# Strings die im APDL-Masterskript geändert werden
df_list = ['r_new','a_new', 'w_new', 't_new', 'h_new']

sample_number = 50
sample_df = sample_generator(bounds, sample_number, df_list)
num_sim = sample_df.shape[0]

print("SIM START")
for i in range(num_sim):
    print("\n")
    start = time.time()
    execute_Ansys2(inputfile, sample_df.iloc[i])
    end = time.time()
    print(f"Simulation: {i+1:>2}/{num_sim} | Vergangene zeit: {end-start:.2f}s")
