import os
import re
import pandas as pd
import math

def get_files_and_labels(parent):
    rmses_files = []
    auc_files = []
    class_labels = []

    for root, dirs, files in os.walk(parent):
        for file in files:
            if file.endswith("_rmses"):
                rmses_files.append(os.path.join(root, file))
            if file.endswith("_auc"):
                auc_files.append(os.path.join(root, file))

    for directory in os.listdir(parent):
        if directory.startswith("class"):
            class_labels.append(directory.replace("class_", "class: "))

    class_labels.sort(key=lambda x: int(x.split(': ')[1].split('_')[0]))
    return rmses_files, auc_files, class_labels

def extract_image_data_info(rmses_all_file):
    parts = re.split(r'[/_-]', rmses_all_file)
    data_nb_tot = 0
    for part in parts:
        if '^' in part:
            data_nb_tot = int(part.split('^')[1])
            break
    data_nb = int((pd.read_csv(rmses_all_file, header=None).shape[0] - 1) / 5)
    return data_nb, data_nb_tot

def calculate_axes_limits(rmses_all_data, auc_data, max_x, max_y, min_y):
    max_rmses_all_data = rmses_all_data.iloc[1:].max().max()
    max_auc_data = auc_data.iloc[1:].max().max()
    min_auc_data = auc_data.iloc[1:].min().min()

    max_rmses_all_data = math.ceil((max_rmses_all_data + 5) / 5) * 5
    max_auc_data = math.ceil((max_auc_data + 0.015) / 0.015) * 0.015
    min_auc_data = math.floor((min_auc_data - 0.015) / 0.015) * 0.015

    return max(max_x, max_rmses_all_data), max(max_y, max_auc_data), min(min_y, min_auc_data)

def get_scatter_plot_data(parent_dir):
    rmses_all_files, auc_files, class_labels, datas_nb, datas_nb_tot = [], [], [], [], []
    max_x, max_y, min_y = 0, 0, 100

    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith("_rmses_all"):
                rmses_all_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith("_auc"):
                auc_files.append(os.path.join(root, file))
                for _ in range(len(rmses_all_files) - 1):
                    auc_files.append(os.path.join(root, file))

    for directory in os.listdir(parent_dir):
        if directory.startswith("class") and os.path.isdir(os.path.join(parent_dir, directory)) and os.listdir(os.path.join(parent_dir, directory)):
            class_labels.append(directory.replace("class_", "class: "))

    rmses_all_files.sort()
    class_labels.sort(key=lambda x: int(x.split(': ')[1].split('_')[0]))

    for l, rmses_all_file in enumerate(rmses_all_files):
        data_nb, data_nb_tot = extract_image_data_info(rmses_all_file)
        datas_nb.append(data_nb)
        datas_nb_tot.append(data_nb_tot)

        auc_data = pd.read_csv(auc_files[l], header=None)
        rmses_all_data = pd.read_csv(rmses_all_file, header=None)

        max_x, max_y, min_y = calculate_axes_limits(rmses_all_data, auc_data, max_x, max_y, min_y)

    return rmses_all_files, auc_files, class_labels, datas_nb, datas_nb_tot, max_x, max_y, min_y

def get_scatter_plot_2_data(parent_dir):
    return get_scatter_plot_data(parent_dir)  # Reusing the same logic as get_scatter_plot_data
