import os
import re
import math
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


directory_name_0 = 'Plots'
if platform.system() == 'Windows':
    directory_path_0 = r'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs'
    directory_path_1 = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{directory_name_0}'

else:
    directory_path_0 = r'/home/ptreyer/Outputs'
    directory_path_1 = rf'/home/ptreyer/Outputs/{directory_name_0}'

if not os.path.exists(directory_path_1):
    path = os.path.join(directory_path_0, directory_name_0)
    os.mkdir(path)


title_label = ['gaussian', 'speckle', 'motion_blurred', 'contrast', 'brightened']
marker_label = ['o', 'x', '+', 's', 'D', '^', 'v', '<', '>']
color_label = ['b', 'g', 'r', 'y', 'k']
hatch_label = ['////', '\\\\\\', '|||', '-', '+', 'x', 'o', 'O', '.', '*']
augmentation = [False, True]


for augmented in augmentation:
    parent_dirs_with_csv = []

    for sev in range(1, 6):

        print("sev: ", sev)
        print("augmented: ", augmented)

        if augmented == True:
            augment = "_augmented"
        else:
            augment = ""

        # Find all parent directories containing CSV files
        parent_dirs_with_csv_plot = []

        for root, dirs, files in os.walk(directory_path_0):
            for file in files:
                if file.endswith("_rmses"):
                    parent_dir_plot = os.path.dirname(root)
                    parent_dir = os.path.dirname(root)

                    if augmented and f"sev{sev}" in file and "augmented" in file:
                        if parent_dir_plot not in parent_dirs_with_csv_plot:
                            parent_dirs_with_csv_plot.append(parent_dir_plot)
                        if parent_dir not in parent_dirs_with_csv and sev == 1:
                            parent_dirs_with_csv.append(parent_dir)

                    elif not augmented and f"sev{sev}" in file and "augmented" not in file:
                        if parent_dir_plot not in parent_dirs_with_csv_plot:
                            parent_dirs_with_csv_plot.append(parent_dir_plot)
                        if parent_dir not in parent_dirs_with_csv and sev == 1:
                            parent_dirs_with_csv.append(parent_dir)

        if not parent_dirs_with_csv:
            break
        parent_dirs_with_csv.sort()
        parent_dirs_with_csv_plot.sort()


        print("parent_dirs_with_csv_plot: ", parent_dirs_with_csv_plot)
        print("parent_dirs_with_csv: ", parent_dirs_with_csv)

        # RMSES plot
        fig, axs = plt.subplots(len(parent_dirs_with_csv_plot), 5, figsize=(30, 5 * len(parent_dirs_with_csv_plot)),
                                squeeze=False)  # Adjust hspace as needed

        plt.suptitle(f"Saliency maps RMSES\nseverity {sev} - Augmentation: {augmented}", fontsize=28)

        for dir_index, parent in enumerate(parent_dirs_with_csv_plot):  # Datasets

            print("parent: ", parent)

            rmses_files = []
            rmses_all_files = []
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

            # Process each matching file
            for rmses_file, auc_file in zip(rmses_files, auc_files):
                # Read the data from the CSV file
                rmses_data = pd.read_csv(rmses_file, header=None)
                auc_data = pd.read_csv(auc_file, header=None)
                auc_data_sev = auc_data.iloc[sev]

                class_nb = int((rmses_data.shape[0] - 1) / 5)

                # Generate target_labels dynamically based on the number of columns
                num_columns = rmses_data.shape[1]
                target_labels = [f'{i}_{j}' for i in range(1, num_columns // 2 + 1) for j in range(1, 3)]

                for k in range(class_nb):  # classes
                    min_range = 1 + (5 * k)
                    max_range = 6 + (5 * k)

                    for i in range(min_range, max_range):  # perturbations
                        axs[dir_index, i - min_range].plot(range(num_columns), rmses_data.iloc[i],
                                                           label=f'{class_labels[k]}')
                        if dir_index == 0:
                            axs[dir_index, i - min_range].set_title(
                                f"{title_label[i - min_range]}\nDelta AUC: {auc_data_sev[i - min_range]}", fontsize=22)
                        else:
                            axs[dir_index, i - min_range].set_title(f"Delta AUC: {auc_data_sev[i - min_range]}",
                                                                    fontsize=16)

                        axs[dir_index, i - min_range].set_xlabel('Target_layers', fontsize=18)
                        axs[dir_index, i - min_range].set_ylabel('RMSES', fontsize=18)
                        if i - min_range == 0:
                            axs[dir_index, i - min_range].legend()
                        axs[dir_index, i - min_range].set_xticks(range(num_columns))
                        axs[dir_index, i - min_range].set_xticklabels(target_labels)

                # Add text to the left of the first plot of each row
                axs[dir_index, 0].text(-0.15, 0.5, f'{os.path.basename(os.path.dirname(parent))}',
                                       va='center', ha='right', rotation=90, transform=axs[dir_index, 0].transAxes,
                                       fontsize=22)

                # Limit y-axis upper limit to maximum data of the row
                max_rmses_data = rmses_data.iloc[1:].max().max()  # Exclude the first row (class labels)
                max_rmses_data += 5
                max_rmses_data = math.ceil(max_rmses_data / 5) * 5

                for ax in axs[dir_index]:
                    ax.set_ylim(0, max_rmses_data)

        plt.tight_layout(rect=[0.03, 0, 0.97, 0.95])
        print(f"\nCombined RMSES plot sev{sev} generated")

        # Save the combined plot
        save_path = os.path.join(directory_path_1, f'RMSES_plot_combined_sev{sev}{augment}.png')
        plt.savefig(save_path)
        plt.close(fig)


    # Scatter plot
    for dir_index, parent_dir in enumerate(parent_dirs_with_csv):  # Datasets
        fig, axs = plt.subplots(5, 8, figsize=(45, 20), squeeze=False,
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.3})  # Adjust hspace as needed

        plt.suptitle(
            f"Correlation Analysis of Delta AUC and Delta Saliency\nDataset: {os.path.basename(os.path.dirname(parent_dir))} - Augmentation: {augmented}\nScatter plot",
            fontsize=22)

        print(f"\nScatter plot - Augmentation = {augmented}:")
        print("   Dataset: ", os.path.basename(os.path.dirname(parent_dir)))

        rmses_files = []
        rmses_all_files = []
        auc_files = []
        class_labels = []
        datas_nb = []
        datas_nb_tot = []
        max_x = 0
        max_y = 0
        min_y = 100

        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith("_rmses_all"):
                    rmses_all_files.append(os.path.join(root, file))

        class_nb = len(rmses_all_files)

        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith("_auc"):
                    auc_files.append(os.path.join(root, file))
                    for _ in range(class_nb - 1):  # Duplicate the auc file to match rmses_all files length
                        auc_files.append(os.path.join(root, file))

        for directory in os.listdir(parent_dir):
            class_dir_path = os.path.join(parent_dir, directory)
            if directory.startswith("class") and os.path.isdir(class_dir_path) and os.listdir(class_dir_path):
                class_labels.append(directory.replace("class_", "class: "))

        rmses_all_files.sort()
        class_labels.sort(key=lambda x: int(x.split(': ')[1].split('_')[0]))
        class_labels_2 = [label.split('_', 1)[-1].replace("class: ", "") for label in class_labels]

        # Process each matching file
        for l, (auc_file, rmses_all_file) in enumerate(zip(auc_files, rmses_all_files)):
            print("      Class: ", class_labels_2[l])

            auc_data = pd.read_csv(auc_file, header=None)
            rmses_all_data = pd.read_csv(rmses_all_file, header=None)

            # Extract the number of image by class from rmses_all_file name
            parts = re.split(r'[/_-]', rmses_all_file)

            for part in parts:
                if '|' in part:
                    data_nb_tot = int(part.split('|')[1])
                    break
            datas_nb_tot.append(data_nb_tot)

            data_nb = int((rmses_all_data.shape[0] - 1) / 5)
            datas_nb.append(data_nb)

            max_rmses_all_data = rmses_all_data.iloc[1:].max().max()  # Exclude the first row (class labels)
            max_auc_data = auc_data.iloc[1:].max().max()
            min_auc_data = auc_data.iloc[1:].min().min()

            max_rmses_all_data += 5
            max_auc_data += 0.015
            min_auc_data -= 0.015

            max_rmses_all_data = math.ceil(max_rmses_all_data / 5) * 5
            max_auc_data = math.ceil(max_auc_data / 0.015) * 0.015
            min_auc_data = math.floor(min_auc_data / 0.015) * 0.015

            if max_rmses_all_data > max_x:
                max_x = max_rmses_all_data
            if max_auc_data > max_y:
                max_y = max_auc_data
            if min_auc_data < min_y:
                min_y = min_auc_data

            target_labels = [f'{i}_{j}' for i in range(1, num_columns // 2 + 1) for j in range(1, 3)]

            for k in range(data_nb):  # Images
                if k != data_nb - 1:
                    print(f"\r         Image number: {k + 1}/{data_nb}", end='', flush=True)
                else:
                    print(f"\r         Image number: {k + 1}/{data_nb}\n", end='', flush=True)

                min_range = 1 + (5 * k)
                max_range = 6 + (5 * k)

                for j in range(8):  # layers
                    for i in range(min_range, max_range):  # severities
                        x_data_serie = rmses_all_data.iloc[i, j::8]
                        x_data_serie = x_data_serie.tolist()
                        y_data_serie = auc_data.iloc[((5 * l) + (i - min_range + 1))]
                        y_data_serie = y_data_serie.tolist()

                        for h in range(5):  # perturbations

                            x_data = x_data_serie[h]
                            y_data = y_data_serie[h]

                            axs[i - min_range, j].scatter(x_data, y_data, marker=marker_label[l],
                                                          color=color_label[h])

                        if i - min_range == 0:
                            axs[i - min_range, j].set_title(f'Layer {target_labels[j]}', fontsize=22)

                        if k == 0:
                            # Add text to the left of the first plot of each row
                            axs[i - min_range, 0].text(-0.3, 0.5, f'severity {i - min_range + 1}',
                                                       va='center', ha='right', rotation=90,
                                                       transform=axs[i - min_range, 0].transAxes,
                                                       fontsize=22)

                            axs[i - min_range, j].set_xlabel('RMSES')
                            axs[i - min_range, j].set_ylabel('Delta AUC')
                            axs[i - min_range, j].set_ylim(min_y, max_y)
                            axs[i - min_range, j].set_xlim(0, max_x)

        # Legend for class labels
        fig.legend(handles=[plt.Line2D([], [], linestyle='None', marker=marker_label[i], color='blue',
                                       label=f"{label} - {datas_nb[i]}/{datas_nb_tot[i]} images")
                            for i, label in enumerate(class_labels_2)],
                   loc='upper left', bbox_to_anchor=(0.01, 0.97), fontsize='xx-large', markerscale=2,
                   title="Classes", title_fontsize='xx-large')

        perturbation_legend_handles = [
            Patch(facecolor=color_label[h], edgecolor=color_label[h], label=title_label[h])
            for h in range(len(title_label))]

        # Legend for perturbations
        fig.legend(handles=perturbation_legend_handles, loc='upper left',
                   bbox_to_anchor=(0.01, 0.87 - (0.02 * class_nb)), fontsize='xx-large',
                   markerscale=2, title="Perturbations", title_fontsize='xx-large')

        plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.98])  # Adjust left and right margins

        print("Saving scatter plot...")
        # Save the combined plot
        save_path = os.path.join(directory_path_1,
                                 f'AUC_scatterplot_{os.path.basename(os.path.dirname(parent_dir))}{augment}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Scatter plot {os.path.basename(os.path.dirname(parent_dir))} generated")

    # Boxplot
    for dir_index, parent_dir in enumerate(parent_dirs_with_csv):  # Datasets
        fig, axs = plt.subplots(5, 8, figsize=(45, 20), squeeze=False,
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.3})  # Adjust hspace as needed

        plt.suptitle(
            f"Correlation Analysis of Delta AUC and Delta Saliency\nDataset: {os.path.basename(os.path.dirname(parent_dir))} - Augmentation: {augmented}\nBox plot",
            fontsize=22)

        print(f"\nBoxplot - Augmentation = {augmented}:")
        print("   Dataset: ", os.path.basename(os.path.dirname(parent_dir)))

        rmses_files = []
        rmses_all_files = []
        auc_files = []
        class_labels = []
        datas_nb = []
        datas_nb_tot = []
        max_x = 0
        max_y = 0
        min_y = 100

        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith("_rmses_all"):
                    rmses_all_files.append(os.path.join(root, file))

        class_nb = len(rmses_all_files)

        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith("_auc"):
                    auc_files.append(os.path.join(root, file))
                    for _ in range(class_nb - 1):  # Duplicate the auc file to match rmses_all files length
                        auc_files.append(os.path.join(root, file))

        for directory in os.listdir(parent_dir):
            class_dir_path = os.path.join(parent_dir, directory)
            if directory.startswith("class") and os.path.isdir(class_dir_path) and os.listdir(class_dir_path):
                class_labels.append(directory.replace("class_", "class: "))

        rmses_all_files.sort()
        class_labels.sort(key=lambda x: int(x.split(': ')[1].split('_')[0]))
        class_labels_2 = [label.split('_', 1)[-1].replace("class: ", "") for label in class_labels]

        # Process each matching file
        for l, (auc_file, rmses_all_file) in enumerate(zip(auc_files, rmses_all_files)):
            print("      Class: ", class_labels_2[l])

            rmses_all_data = pd.read_csv(rmses_all_file, header=None)
            auc_data = pd.read_csv(auc_file, header=None)  # Assuming auc_data needs to be read similarly

            # Extract the number of image by class from rmses_all_file name
            parts = re.split(r'[/_-]', rmses_all_file)

            for part in parts:
                if '|' in part:
                    data_nb_tot = int(part.split('|')[1])
                    break
            datas_nb_tot.append(data_nb_tot)

            data_nb = int((rmses_all_data.shape[0] - 1) / 5)
            datas_nb.append(data_nb)

            max_rmses_all_data = rmses_all_data.iloc[1:].max().max()  # Exclude the first row (class labels)
            max_auc_data = auc_data.iloc[1:].max().max()
            min_auc_data = auc_data.iloc[1:].min().min()

            max_rmses_all_data += 5
            max_auc_data += 0.015
            min_auc_data -= 0.015

            max_rmses_all_data = math.ceil(max_rmses_all_data / 5) * 5
            max_auc_data = math.ceil(max_auc_data / 0.015) * 0.015
            min_auc_data = math.floor(min_auc_data / 0.015) * 0.015

            if max_rmses_all_data > max_x:
                max_x = max_rmses_all_data
            if max_auc_data > max_y:
                max_y = max_auc_data
            if min_auc_data < min_y:
                min_y = min_auc_data

            target_labels = [f'{i}_{j}' for i in range(1, num_columns // 2 + 1) for j in range(1, 3)]

            count = 0

            for j in range(8):  # layers
                for i in range(5):  # perturbations
                    count += 1

                    if count != 40:
                        print(f"\r         Boxplot number: {count}/40", end='', flush=True)
                    else:
                        print(f"\r         Boxplot number: {count}/40\n", end='', flush=True)

                    for k in range(5):  # severities
                        x_data = []
                        y_data = []

                        auc_row = 1 + k + (l * 5)
                        auc_column = i

                        y_data = auc_data.iloc[auc_row, auc_column]

                        for h in range(data_nb):  # Images
                            rmses_all_row = 1 + (5 * h) + k
                            rmses_all_column = 8 * i + j

                            x_data_serie = rmses_all_data.iloc[rmses_all_row, rmses_all_column]
                            x_data.append(x_data_serie)

                        # Calculate the bottom position for the boxplot
                        bottom_position = y_data  # Use y_data directly for bottom position

                        # Customizing boxplot colors based on i (perturbations) and hatches based on l (classes)
                        box_color = color_label[i % len(color_label)]  # Using modulo to cycle through colors
                        hatch_pattern = hatch_label[
                            l % len(hatch_label)]  # Using modulo to cycle through hatch patterns

                        if hatch_pattern == 'none':
                            boxprops = dict(color=box_color, facecolor='none')
                        else:
                            boxprops = dict(color=box_color, facecolor='none', hatch=hatch_pattern)

                        whiskerprops = dict(color=box_color)
                        capprops = dict(color=box_color)
                        medianprops = dict(color='red', linewidth=2)
                        flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor=box_color,
                                          markersize=8,
                                          linestyle='none')

                        axs[k, j].boxplot(x_data, positions=[bottom_position], vert=False, widths=0.02,
                                          patch_artist=True, boxprops=boxprops, whiskerprops=whiskerprops,
                                          capprops=capprops, medianprops=medianprops, flierprops=flierprops)

                        axs[k, j].set_xlabel('RMSES')
                        axs[k, j].set_ylabel('Delta AUC')
                        axs[k, 0].text(-0.25, 0.5, f'severity {k + 1}', va='center', ha='right', rotation=90,
                                       transform=axs[k, 0].transAxes, fontsize=22)

                axs[0, j].set_title(f'Layer {target_labels[j]}', fontsize=22)

        for axe in axs:
            for ax in axe:
                ax.set_xlim(0, max_x)
                ax.set_ylim(min_y, max_y)

        plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.98])  # Adjust left and right margins

        # Legend for classes
        handles = [
            Patch(edgecolor='b', facecolor='none', hatch=hatch_label[i % len(hatch_label)])
            # Blue edge color for legend
            for i in range(len(class_labels_2))
        ]

        labels = [f"{label} - {datas_nb[i]}/{datas_nb_tot[i]} images" for i, label in enumerate(class_labels_2)]

        fig.legend(handles=handles,
                   labels=labels,
                   loc='upper left', bbox_to_anchor=(0.01, 0.97), fontsize='xx-large', markerscale=2,
                   title="Classes",
                   title_fontsize='xx-large')

        perturbation_legend_handles = [
            Patch(facecolor=color_label[h], edgecolor=color_label[h], label=title_label[h])
            for h in range(len(title_label))]

        # Legend for perturbations
        fig.legend(handles=perturbation_legend_handles, loc='upper left',
                   bbox_to_anchor=(0.01, 0.87 - (0.02 * class_nb)), fontsize='xx-large',
                   markerscale=2, title="Perturbations", title_fontsize='xx-large')

        save_path = os.path.join(directory_path_1,
                                 f'AUC_boxplot_{os.path.basename(os.path.dirname(parent_dir))}{augment}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"            Box plot {os.path.basename(os.path.dirname(parent_dir))} generated")

    # Average plot
    for dir_index, parent_dir in enumerate(parent_dirs_with_csv):
        fig, axs = plt.subplots(5, 8, figsize=(45, 20), squeeze=False,
                                gridspec_kw={'hspace': 0.2, 'wspace': 0.3})  # Adjust hspace as needed

        plt.suptitle(
            f"Correlation Analysis of Delta AUC and Delta Saliency\nDataset: {os.path.basename(os.path.dirname(parent_dir))} - Augmentation: {augmented}\nAverage plot",
            fontsize=22)

        print(f"\nAverage plot - Augmentation = {augmented}:")
        print("   Dataset: ", os.path.basename(os.path.dirname(parent_dir)))

        rmses_files = []
        rmses_all_files = []
        auc_files = []
        class_labels = []
        datas_nb = []
        datas_nb_tot = []
        max_x = 0
        max_y = 0
        min_y = 100

        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith("_rmses_all"):
                    rmses_all_files.append(os.path.join(root, file))

        class_nb = len(rmses_all_files)

        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith("_auc"):
                    auc_files.append(os.path.join(root, file))
                    for _ in range(class_nb - 1):  # Duplicate the auc file to match rmses_all files length
                        auc_files.append(os.path.join(root, file))

        for directory in os.listdir(parent_dir):
            class_dir_path = os.path.join(parent_dir, directory)
            if directory.startswith("class") and os.path.isdir(class_dir_path) and os.listdir(class_dir_path):
                class_labels.append(directory.replace("class_", "class: "))

        rmses_all_files.sort()
        class_labels.sort(key=lambda x: int(x.split(': ')[1].split('_')[0]))
        class_labels_2 = [label.split('_', 1)[-1].replace("class: ", "") for label in class_labels]
        print("class_labels_2", class_labels_2)

        # Process each matching file
        for l, (auc_file, rmses_all_file) in enumerate(zip(auc_files, rmses_all_files)):
            print("      Class: ", class_labels_2[l])

            rmses_all_data = pd.read_csv(rmses_all_file, header=None)
            auc_data = pd.read_csv(auc_file, header=None)  # Assuming auc_data needs to be read similarly

            # Extract the number of image by class from rmses_all_file name
            parts = re.split(r'[/_-]', rmses_all_file)

            for part in parts:
                if '|' in part:
                    data_nb_tot = int(part.split('|')[1])
                    break
            datas_nb_tot.append(data_nb_tot)

            data_nb = int((rmses_all_data.shape[0] - 1) / 5)
            datas_nb.append(data_nb)

            max_rmses_all_data = rmses_all_data.iloc[1:].max().max()  # Exclude the first row (class labels)
            max_auc_data = auc_data.iloc[1:].max().max()
            min_auc_data = auc_data.iloc[1:].min().min()

            max_rmses_all_data += 5
            max_auc_data += 0.015
            min_auc_data -= 0.015

            max_rmses_all_data = math.ceil(max_rmses_all_data / 5) * 5
            max_auc_data = math.ceil(max_auc_data / 0.015) * 0.015
            min_auc_data = math.floor(min_auc_data / 0.015) * 0.015

            if max_rmses_all_data > max_x:
                max_x = max_rmses_all_data
            if max_auc_data > max_y:
                max_y = max_auc_data
            if min_auc_data < min_y:
                min_y = min_auc_data

            target_labels = [f'{i}_{j}' for i in range(1, num_columns // 2 + 1) for j in range(1, 3)]

            count = 0
            max_x_data = 0

            for j in range(8):  # layers
                for i in range(5):  # perturbations
                    count += 1

                    if count != 40:
                        print(f"\r         Boxplot number: {count}/40", end='', flush=True)
                    else:
                        print(f"\r         Boxplot number: {count}/40\n", end='', flush=True)

                    for k in range(5):  # severities
                        x_data_serie_all = []
                        y_data = []

                        auc_row = 1 + k + (l * 5)
                        auc_column = i

                        y_data = auc_data.iloc[auc_row, auc_column]

                        for h in range(data_nb):  # Images
                            rmses_all_row = 1 + (5 * h) + k
                            rmses_all_column = 8 * i + j

                            x_data_serie = rmses_all_data.iloc[rmses_all_row, rmses_all_column]
                            x_data_serie_all.append(x_data_serie)

                        x_data = np.mean(x_data_serie_all)
                        if x_data > max_x_data:
                            max_x_data = x_data

                        color_index = (l * 5 + i) % len(color_label)  # Cycle through colors for each (l, i)
                        marker_index = l % len(marker_label[l])  # Cycle through markers for each l

                        axs[k, j].scatter(x_data, y_data, color=color_label[color_index],
                                          marker=marker_label[l][marker_index])

                        axs[k, j].set_xlabel('RMSES')
                        axs[k, j].set_ylabel('Delta AUC')
                        axs[k, 0].text(-0.2, 0.5, f'severity {k + 1}', va='center', ha='right', rotation=90,
                                       transform=axs[k, 0].transAxes, fontsize=22)

                axs[0, j].set_title(f'Layer {target_labels[j]}', fontsize=22)

        max_x_data += 5
        max_x = math.ceil(max_x_data / 5) * 5

        for axe in axs:
            for ax in axe:
                ax.set_xlim(0, max_x)
                ax.set_ylim(min_y, max_y)

        # Legend for class labels
        fig.legend(handles=[
            plt.Line2D([], [], linestyle='None', marker=marker_label[i], color='blue',
                       label=f"{label} - {datas_nb[i]}/{datas_nb_tot[i]} images")
            for i, label in enumerate(class_labels_2)],
            loc='upper left', bbox_to_anchor=(0.01, 0.97), fontsize='xx-large', markerscale=2, title="Classes",
            title_fontsize='xx-large')

        perturbation_legend_handles = [
            Patch(facecolor=color_label[h], edgecolor=color_label[h], label=title_label[h])
            for h in range(len(title_label))]

        plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.98])  # Adjust left and right margins

        # Legend for perturbations
        perturbation_legend_handles = [
            Patch(facecolor=color_label[h], edgecolor=color_label[h], label=title_label[h])
            for h in range(len(title_label))]

        fig.legend(handles=perturbation_legend_handles, loc='upper left',
                   bbox_to_anchor=(0.01, 0.87 - (0.02 * class_nb)), fontsize='xx-large',
                   markerscale=2, title="Perturbations", title_fontsize='xx-large')

        save_path = os.path.join(directory_path_0,
                                 f'AUC_averageplot_{os.path.basename(os.path.dirname(parent_dir))}{augment}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"            Average plot {os.path.basename(os.path.dirname(parent_dir))} generated")
