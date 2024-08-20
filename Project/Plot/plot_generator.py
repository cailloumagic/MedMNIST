from file_handler import set_directory_paths, find_csv_files
from data_processor import get_files_and_labels, get_scatter_plot_data, get_scatter_plot_2_data
from plotting import Plotting
import os
import math
import matplotlib.pyplot as plt
import pandas as pd


class PlotGenerator:
    def __init__(self, directory_name='Plots'):
        self.directory_name = directory_name
        self.directory_path_0, self.directory_path_1 = set_directory_paths(self.directory_name)
        self.title_label = ['gaussian', 'speckle', 'motion_blurred', 'contrast', 'brightened']
        self.marker_label = ['o', 'x', '+', 's', 'D', '^', 'v', '<', '>']
        self.color_label = ['y', 'r', 'g', 'b', 'k']
        self.hatch_label = ['////', '\\\\\\', '|||', '-', '+', 'x', 'o', 'O', '.', '*']
        self.prediction = [True, False]
        self.augmentation = [False, True]
        self.plotting = Plotting(self.title_label, self.marker_label, self.color_label, self.hatch_label)

    def generate_plots(self):
        for predicted in self.prediction:
            for augmented in self.augmentation:

                # Initialize a list to collect directories with CSV files across all severities
                accumulated_dirs_with_csv = []

                # Loop through severities to accumulate directories with CSV files
                for sev in range(1, 6):
                    parent_dirs_with_csv, parent_dirs_with_csv_plot = find_csv_files(self.directory_path_0, sev,
                                                                                     augmented, predicted)
                    if parent_dirs_with_csv:
                        # Accumulate directories with CSV files
                        accumulated_dirs_with_csv.extend(parent_dirs_with_csv)

                    if parent_dirs_with_csv_plot:
                        self.generate_rmses_plot(parent_dirs_with_csv_plot, sev, augmented, predicted)

                # After collecting all directories with CSV files, generate other plots
                if accumulated_dirs_with_csv:
                    self.generate_scatter_plot(accumulated_dirs_with_csv, augmented, predicted)
                    self.generate_scatter_plot_2(accumulated_dirs_with_csv, augmented, predicted)
                    self.generate_boxplot(accumulated_dirs_with_csv, augmented, predicted)
                    self.generate_boxplot_2(accumulated_dirs_with_csv, augmented, predicted)

    def generate_rmses_plot(self, parent_dirs_with_csv_plot, sev, augmented, predicted):
        fig, axs = plt.subplots(len(parent_dirs_with_csv_plot), 5, figsize=(30, 5 * len(parent_dirs_with_csv_plot)),
                                squeeze=False)
        plt.suptitle(
            f"Saliency maps RMSEs\nseverity {sev} - Augmentation: {augmented} - Correct prediction: {predicted}",
            fontsize=28)

        # Calculate maximum value for y-axis limit
        # Assuming that rmses_data is available here or calculate for the entire dataset
        rmses_data = pd.concat([pd.read_csv(file, header=None) for file in
                                [get_files_and_labels(parent)[0][0] for parent in parent_dirs_with_csv_plot]])
        max_rmses_data = rmses_data.iloc[1:].max().max()
        max_rmses_data += 5
        max_rmses_data = math.ceil(max_rmses_data / 5) * 5

        for dir_index, parent in enumerate(parent_dirs_with_csv_plot):
            rmses_files, auc_files, class_labels = get_files_and_labels(parent)
            for rmses_file, auc_file in zip(rmses_files, auc_files):
                self.plotting.plot_rmses(axs, rmses_file, auc_file, class_labels, dir_index, sev, max_rmses_data)

            axs[dir_index, 0].text(-0.15, 0.5, f'{os.path.basename(os.path.dirname(parent))}',
                                   va='center', ha='right', rotation=90, transform=axs[dir_index, 0].transAxes,
                                   fontsize=22)

        plt.tight_layout(rect=[0.03, 0, 0.97, 0.95])
        print(f"\nCombined RMSEs plot sev{sev} - Augmentation={augmented} - Correct_prediction={predicted} generated")
        save_path = os.path.join(self.directory_path_1,
                                 f'RMSEs_plot_combined_sev{sev}{self.get_suffix(augmented, predicted)}.png')
        plt.savefig(save_path)
        plt.close(fig)

    def generate_scatter_plot(self, parent_dirs_with_csv, augmented, predicted):
        for dir_index, parent_dir in enumerate(parent_dirs_with_csv):
            fig, axs = plt.subplots(5, 8, figsize=(45, 20), squeeze=False, gridspec_kw={'hspace': 0.2, 'wspace': 0.3})
            plt.suptitle(
                f"Correlation Analysis of Delta AUC and Delta Saliency\nDataset: {os.path.basename(os.path.dirname(parent_dir))} - Augmentation: {augmented} - Correct prediction: {predicted}\nScatter plot",
                fontsize=22)

            print(f"\nScatter plot - Augmentation={augmented} - Correct_prediction={predicted}:")
            print("   Dataset: ", os.path.basename(os.path.dirname(parent_dir)))

            rmses_all_files, auc_files, class_labels, datas_nb, datas_nb_tot, max_x, max_y, min_y = get_scatter_plot_data(
                parent_dir)
            target_labels = [f'{i}_{j}' for i in range(1, 9) for j in range(1, 3)]

            for l, (auc_file, rmses_all_file) in enumerate(zip(auc_files, rmses_all_files)):
                print("      Class: ", class_labels[l])
                auc_data = pd.read_csv(auc_file, header=None)
                rmses_all_data = pd.read_csv(rmses_all_file, header=None)

                count = 0

                for k in range(datas_nb[l]):
                    count += 1
                    print(f"\r{' ' * 80}", end='')  # Clears the line

                    if count != datas_nb[l]:
                        print(f"\r          Image Number: {count}/{datas_nb[l]}", end='', flush=True)
                    else:
                        print(f"\r          Image Number: {count}/{datas_nb[l]}\n", end='', flush=True)

                    min_range = 1 + (5 * k)
                    max_range = 6 + (5 * k)
                    for j in range(8):
                        for i in range(min_range, max_range):
                            x_data_serie = rmses_all_data.iloc[i, j::8].tolist()
                            y_data_serie = auc_data.iloc[(5 * l) + (i - min_range + 1)].tolist()

                            for h in range(5):
                                axs[i - min_range, j].scatter(x_data_serie[h], y_data_serie[h],
                                                              marker=self.marker_label[l],
                                                              color=self.color_label[h])

                            if i - min_range == 0:
                                axs[i - min_range, j].set_title(f'Layer {target_labels[j]}', fontsize=22)

                            axs[i - min_range, j].set_xlabel('RMSEs')
                            axs[i - min_range, j].set_ylabel('Deltas AUC')
                            axs[i - min_range, j].set_ylim(min_y, max_y)
                            axs[i - min_range, j].set_xlim(0, max_x)

            self.plotting.add_legend_scatter_plot(fig, class_labels, datas_nb, datas_nb_tot)
            plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.98])
            save_path = os.path.join(self.directory_path_1,
                                     f'AUC_scatterplot_{os.path.basename(os.path.dirname(parent_dir))}{self.get_suffix(augmented, predicted)}.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Scatter plot {os.path.basename(os.path.dirname(parent_dir))} generated")

    def generate_scatter_plot_2(self, parent_dirs_with_csv, augmented, predicted):
        for dir_index, parent_dir in enumerate(parent_dirs_with_csv):
            fig, axs = plt.subplots(5, 8, figsize=(45, 20), squeeze=False, gridspec_kw={'hspace': 0.2, 'wspace': 0.3})
            plt.suptitle(
                f"Correlation Analysis of Delta AUC and Delta Saliency\nDataset: {os.path.basename(os.path.dirname(parent_dir))} - Augmentation: {augmented} - Correct prediction: {predicted}\nScatter plot 2",
                fontsize=22)

            print(f"\nScatter plot 2 - Augmentation={augmented} - Correct_prediction={predicted}:")
            print("   Dataset: ", os.path.basename(os.path.dirname(parent_dir)))

            # Get data
            try:
                rmses_all_files, auc_files, class_labels, datas_nb, datas_nb_tot, max_x, max_y, min_y = get_scatter_plot_2_data(
                    parent_dir)
            except Exception as e:
                print(f"Error fetching data for Scatter plot 2: {e}")
                continue

            for l, (auc_file, rmses_all_file) in enumerate(zip(auc_files, rmses_all_files)):
                print("      Class: ", class_labels[l])
                try:
                    auc_data = pd.read_csv(auc_file, header=None)
                    rmses_all_data = pd.read_csv(rmses_all_file, header=None)
                except Exception as e:
                    print(f"Error reading files for class {class_labels[l]}: {e}")
                    continue

                if rmses_all_data.empty or auc_data.empty:
                    print(f"Empty data for class {class_labels[l]}. Skipping...")
                    continue

                # Define target_labels after rmses_all_data has been set
                target_labels = [f'{i}_{j}' for i in range(1, rmses_all_data.shape[1] // 2 + 1) for j in range(1, 3)]

                count = 0

                for k in range(datas_nb[l]):
                    count += 1
                    print(f"\r{' ' * 80}", end='')  # Clears the line

                    if count != datas_nb[l]:
                        print(f"\r          Image Number: {count}/{datas_nb[l]}", end='', flush=True)
                    else:
                        print(f"\r          Image Number: {count}/{datas_nb[l]}\n", end='', flush=True)

                    min_range = 1 + (5 * k)
                    max_range = 6 + (5 * k)

                    for j in range(8):  # layers

                        for h in range(5):  # perturbations

                            for i in range(min_range, max_range):  # severities
                                x_data_serie = rmses_all_data.iloc[i, j::8].tolist()
                                y_data_serie = auc_data.iloc[(5 * l) + (i - min_range + 1)].tolist()
                                y_data_serie = auc_data.iloc[((5 * l) + (i - min_range + 1))]
                                y_data_serie = y_data_serie.tolist()

                                x_data = x_data_serie[h]
                                y_data = y_data_serie[h]

                                axs[h, j].scatter(x_data, y_data, marker=self.marker_label[l],
                                                  color=self.color_label[i - min_range])

                            if h == 0:
                                axs[h, j].set_title(f'Layer {target_labels[j]}', fontsize=22)

                            if k == 0:
                                # Add text to the left of the first plot of each row
                                axs[h, 0].text(-0.3, 0.5, self.title_label[h],
                                               va='center', ha='right', rotation=90,
                                               transform=axs[h, 0].transAxes, fontsize=22)

                                axs[h, j].set_xlabel('RMSEs')
                                axs[h, j].set_ylabel('Deltas AUC')
                                axs[h, j].set_ylim(min_y, max_y)
                                axs[h, j].set_xlim(0, max_x)

            self.plotting.add_legend_scatter_plot_2(fig, class_labels, datas_nb, datas_nb_tot)
            plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.98])
            save_path = os.path.join(self.directory_path_1,
                                     f'AUC_scatterplot_2_{os.path.basename(os.path.dirname(parent_dir))}{self.get_suffix(augmented, predicted)}.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Scatter plot 2 {os.path.basename(os.path.dirname(parent_dir))} generated")

    def generate_boxplot(self, parent_dirs_with_csv, augmented, predicted):
        for dir_index, parent_dir in enumerate(parent_dirs_with_csv):
            fig, axs = plt.subplots(5, 8, figsize=(45, 20), squeeze=False, gridspec_kw={'hspace': 0.2, 'wspace': 0.3})
            plt.suptitle(f"Correlation Analysis of Delta AUC and Delta Saliency\nDataset: {os.path.basename(os.path.dirname(parent_dir))} - Augmentation: {augmented} - Correct prediction: {predicted}\nBox plot", fontsize=22)

            print(f"\nBoxplot - Augmentation={augmented} - Correct_prediction={predicted}:")
            print("   Dataset: ", os.path.basename(os.path.dirname(parent_dir)))

            rmses_all_files, auc_files, class_labels, datas_nb, datas_nb_tot, max_x, max_y, min_y = get_scatter_plot_2_data(parent_dir)
            for l, (auc_file, rmses_all_file) in enumerate(zip(auc_files, rmses_all_files)):
                print("      Class: ", class_labels[l])
                auc_data = pd.read_csv(auc_file, header=None)
                rmses_all_data = pd.read_csv(rmses_all_file, header=None)
                num_columns = rmses_all_data.shape[1]
                target_labels = [f'{i}_{j}' for i in range(1, num_columns // 2 + 1) for j in range(1, 3)]
                self.plotting.plot_boxplot(axs, auc_data, rmses_all_data, l, datas_nb, max_x, max_y, min_y, target_labels)

            self.plotting.add_legend_boxplot(fig, class_labels, datas_nb, datas_nb_tot)
            plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.98])
            save_path = os.path.join(self.directory_path_1, f'AUC_boxplot_{os.path.basename(os.path.dirname(parent_dir))}{self.get_suffix(augmented, predicted)}.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Box plot {os.path.basename(os.path.dirname(parent_dir))} generated")

    def generate_boxplot_2(self, parent_dirs_with_csv, augmented, predicted):
        for dir_index, parent_dir in enumerate(parent_dirs_with_csv):
            fig, axs = plt.subplots(5, 8, figsize=(45, 20), squeeze=False, gridspec_kw={'hspace': 0.2, 'wspace': 0.3})
            plt.suptitle(f"Correlation Analysis of Delta AUC and Delta Saliency\nDataset: {os.path.basename(os.path.dirname(parent_dir))} - Augmentation: {augmented} - Correct prediction: {predicted}\nBoxplot 2", fontsize=22)

            print(f"\nBoxplot 2 - Augmentation={augmented} - Correct_prediction={predicted}:")
            print("   Dataset: ", os.path.basename(os.path.dirname(parent_dir)))

            rmses_all_files, auc_files, class_labels, datas_nb, datas_nb_tot, max_x, max_y, min_y = get_scatter_plot_2_data(parent_dir)
            for l, (auc_file, rmses_all_file) in enumerate(zip(auc_files, rmses_all_files)):
                print("      Class: ", class_labels[l])
                auc_data = pd.read_csv(auc_file, header=None)
                rmses_all_data = pd.read_csv(rmses_all_file, header=None)
                num_columns = rmses_all_data.shape[1]
                target_labels = [f'{i}_{j}' for i in range(1, num_columns // 2 + 1) for j in range(1, 3)]
                self.plotting.plot_boxplot_2(axs, auc_data, rmses_all_data, l, datas_nb, max_x, max_y, min_y, target_labels)

            self.plotting.add_legend_boxplot_2(fig, class_labels, datas_nb, datas_nb_tot)
            plt.tight_layout(rect=[0.02, 0.02, 0.99, 0.98])
            save_path = os.path.join(self.directory_path_1, f'AUC_boxplot_2_{os.path.basename(os.path.dirname(parent_dir))}{self.get_suffix(augmented, predicted)}.png')
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Boxplot 2 {os.path.basename(os.path.dirname(parent_dir))} generated")

    @staticmethod
    def get_suffix(augmented, predicted):
        augment = "_augmented" if augmented else ""
        predict = "_correctly-predicted" if predicted else "_incorrectly-predicted"
        return f'{augment}{predict}'

if __name__ == '__main__':
    plot_generator = PlotGenerator()
    plot_generator.generate_plots()
