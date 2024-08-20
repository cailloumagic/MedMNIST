import os
import sys
import platform

from config import *

def set_directory_paths(directory_name):
    if platform.system() == 'Windows':
        directory_path_0 = r'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs'
    else:
        directory_path_0 = r'/home/ptreyer/Outputs'

    directory_path_1 = os.path.join(directory_path_0, directory_name)
    if not os.path.exists(directory_path_1):
        os.mkdir(directory_path_1)
    return directory_path_0, directory_path_1

def find_csv_files(directory_path_0, sev, augmented, predicted):
    parent_dirs_with_csv = []
    parent_dirs_with_csv_plot = []

    for root, dirs, files in os.walk(directory_path_0):
        for file in files:
            if file.endswith("_rmses"):
                parent_dir_plot = os.path.dirname(root)
                parent_dir = os.path.dirname(root)

                if match_conditions(file, sev, augmented, predicted):
                    if parent_dir_plot not in parent_dirs_with_csv_plot:
                        parent_dirs_with_csv_plot.append(parent_dir_plot)
                    if parent_dir not in parent_dirs_with_csv and sev == 1:
                        parent_dirs_with_csv.append(parent_dir)

    parent_dirs_with_csv.sort()
    parent_dirs_with_csv_plot.sort()

    return parent_dirs_with_csv, parent_dirs_with_csv_plot

def match_conditions(file, sev, augmented, predicted):
    if augmented and predicted and f"sev{sev}" in file and "augmented" in file and not "false" in file:
        return True
    if not augmented and predicted and f"sev{sev}" in file and "augmented" not in file and not "false" in file:
        return True
    if augmented and not predicted and f"sev{sev}" in file and "augmented" in file and "false" in file:
        return True
    if not augmented and not predicted and f"sev{sev}" in file and "augmented" not in file and "false" in file:
        return True
    return False
