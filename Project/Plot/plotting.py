import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class Plotting:
    def __init__(self, title_label, marker_label, color_label, hatch_label):
        self.title_label = title_label
        self.marker_label = marker_label
        self.color_label = color_label
        self.hatch_label = hatch_label

    def plot_rmses(self, axs, rmses_file, auc_file, class_labels, dir_index, sev, max_rmses_data):
        rmses_data = pd.read_csv(rmses_file, header=None)
        auc_data = pd.read_csv(auc_file, header=None)
        auc_data_sev = auc_data.iloc[sev]
        class_nb = int((rmses_data.shape[0] - 1) / 5)
        num_columns = rmses_data.shape[1]
        target_labels = [f'{i}_{j}' for i in range(1, num_columns // 2 + 1) for j in range(1, 3)]

        for k in range(class_nb):
            min_range = 1 + (5 * k)
            max_range = 6 + (5 * k)
            for i in range(min_range, max_range):
                axs[dir_index, i - min_range].plot(range(num_columns), rmses_data.iloc[i], label=f'{class_labels[k]}')
                if dir_index == 0:
                    axs[dir_index, i - min_range].set_title(
                        f"{self.title_label[i - min_range]}\nDeltas AUC: {auc_data_sev[i - min_range]}", fontsize=22)
                else:
                    axs[dir_index, i - min_range].set_title(f"Deltas AUC: {auc_data_sev[i - min_range]}", fontsize=16)

                axs[dir_index, i - min_range].set_xlabel('Target_layers', fontsize=18)
                axs[dir_index, i - min_range].set_ylabel('RMSEs', fontsize=18)
                if i - min_range == 0:
                    axs[dir_index, i - min_range].legend()
                axs[dir_index, i - min_range].set_xticks(range(num_columns))
                axs[dir_index, i - min_range].set_xticklabels(target_labels)
                axs[dir_index, i - min_range].set_ylim(0, max_rmses_data)


    def plot_boxplot(self, axs, auc_data, rmses_all_data, l, datas_nb, max_x, max_y, min_y, target_labels):
        data_nb = datas_nb[l]
        count = 0
        for j in range(8):  # layers
            for i in range(5):  # perturbations
                count += 1

                if count != 40:
                    print(f"\r         Boxplot 2 number: {count}/40", end='', flush=True)
                else:
                    print(f"\r         Boxplot 2 number: {count}/40\n")

                for k in range(5):  # severities
                    x_data = []

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

                    box_color = self.color_label[i % len(self.color_label)]
                    hatch_pattern = self.hatch_label[l % len(self.hatch_label)]

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

                    axs[k, j].set_xlabel('RMSEs')
                    axs[k, j].set_ylabel('Deltas AUC')
                    axs[k, 0].text(-0.25, 0.5, f'severity {k + 1}', va='center', ha='right', rotation=90,
                                   transform=axs[k, 0].transAxes, fontsize=22)

            axs[0, j].set_title(f'Layer {target_labels[j]}', fontsize=22)

        for axe in axs:
            for ax in axe:
                ax.set_xlim(0, max_x)
                ax.set_ylim(min_y, max_y)

    def plot_boxplot_2(self, axs, auc_data, rmses_all_data, l, datas_nb, max_x, max_y, min_y, target_labels):
        data_nb = datas_nb[l]
        count = 0

        for j in range(8):  # layers
            for i in range(5):  # perturbations
                count += 1

                if count != 40:
                    print(f"\r         Boxplot 2 number: {count}/40", end='', flush=True)
                else:
                    print(f"\r         Boxplot 2 number: {count}/40\n")

                for k in range(5):  # severities
                    x_data = []

                    auc_row = 1 + k + (l * 5)
                    auc_column = i

                    y_data = auc_data.iloc[auc_row, auc_column]

                    for h in range(data_nb):  # Images
                        rmses_all_row = 1 + (5 * h) + k
                        rmses_all_column = 8 * i + j

                        x_data_serie = rmses_all_data.iloc[rmses_all_row, rmses_all_column]
                        x_data.append(x_data_serie)

                    # Calculate the bottom position for the boxplot 2
                    bottom_position = y_data  # Use y_data directly for bottom position

                    # Customizing boxplot 2 colors based on i (perturbations) and hatches based on l (classes)
                    box_color = self.color_label[k % len(self.color_label)]  # Using modulo to cycle through colors
                    hatch_pattern = self.hatch_label[
                        l % len(self.hatch_label)]  # Using modulo to cycle through hatch patterns

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

                    axs[i, j].boxplot(x_data, positions=[bottom_position], vert=False, widths=0.02,
                                      patch_artist=True, boxprops=boxprops, whiskerprops=whiskerprops,
                                      capprops=capprops, medianprops=medianprops, flierprops=flierprops)

                    axs[i, j].set_xlabel('RMSEs')
                    axs[i, j].set_ylabel('Deltas AUC')
                    axs[i, 0].text(-0.3, 0.5, self.title_label[i],
                                   va='center', ha='right', rotation=90,
                                   transform=axs[i, 0].transAxes, fontsize=22)

            axs[0, j].set_title(f'Layer {target_labels[j]}', fontsize=22)

        for axe in axs:
            for ax in axe:
                ax.set_xlim(0, max_x)
                ax.set_ylim(min_y, max_y)

    def add_legend_scatter_plot(self, fig, class_labels, datas_nb, datas_nb_tot):
        # Create handles for class labels
        class_legend_handles = [
            plt.Line2D([], [], linestyle='None', marker=self.marker_label[i % len(self.marker_label)],
                       color=self.color_label[i % len(self.color_label)], markersize=10,
                       label=f"{label} - {datas_nb[i]}/{datas_nb_tot[i]} images")
            for i, label in enumerate(class_labels)
        ]

        # Add legend for class labels
        fig.legend(handles=class_legend_handles, loc='upper left', bbox_to_anchor=(0.01, 0.97),
                   fontsize='xx-large', markerscale=2, title="Classes", title_fontsize='xx-large')

        # Create handles for perturbations
        perturbation_legend_handles = [
            Patch(facecolor=self.color_label[h % len(self.color_label)],
                  edgecolor=self.color_label[h % len(self.color_label)],
                  label=self.title_label[h % len(self.title_label)])
            for h in range(len(self.title_label))
        ]

        # Add legend for perturbations
        fig.legend(handles=perturbation_legend_handles, loc='upper left',
                   bbox_to_anchor=(0.01, 0.87 - (0.02 * len(class_labels))), fontsize='xx-large',
                   markerscale=2, title="Perturbations", title_fontsize='xx-large')

    def add_legend_scatter_plot_2(self, fig, class_labels, datas_nb, datas_nb_tot):
        fig.legend(handles=[plt.Line2D([], [], linestyle='None', marker=self.marker_label[i], color='blue',
                                       label=f"{label} - {datas_nb[i]}/{datas_nb_tot[i]} images")
                            for i, label in enumerate(class_labels)],
                   loc='upper left', bbox_to_anchor=(0.01, 0.97), fontsize='xx-large', markerscale=2,
                   title="Classes", title_fontsize='xx-large')

        severities_legend_handles = [
            Patch(facecolor=self.color_label[h], edgecolor=self.color_label[h], label=f"Severity {h + 1}")
            for h in range(5)
        ]

        fig.legend(handles=severities_legend_handles, loc='upper left',
                   bbox_to_anchor=(0.01, 0.87 - (0.02 * len(class_labels))), fontsize='xx-large',
                   markerscale=2, title="Severities", title_fontsize='xx-large')

    def add_legend_boxplot(self, fig, class_labels, datas_nb, datas_nb_tot):

        # Legend for classes
        handles = [
            Patch(edgecolor='b', facecolor='none', hatch=self.hatch_label[i % len(self.hatch_label)])
            for i in range(len(class_labels))
        ]
        labels = [f"{label} - {datas_nb[i]}/{datas_nb_tot[i]} images" for i, label in enumerate(class_labels)]

        fig.legend(handles=handles, labels=labels,
                   loc='upper left', bbox_to_anchor=(0.01, 0.97), fontsize='xx-large', markerscale=2,
                   title="Classes", title_fontsize='xx-large')

        # Legend for perturbations
        perturbation_legend_handles = [
            Patch(facecolor=self.color_label[h], edgecolor=self.color_label[h], label=self.title_label[h])
            for h in range(len(self.title_label))
        ]

        fig.legend(handles=perturbation_legend_handles, loc='upper left',
                   bbox_to_anchor=(0.01, 0.87 - (0.02 * len(class_labels))), fontsize='xx-large',
                   markerscale=2, title="Perturbations", title_fontsize='xx-large')

    def add_legend_boxplot_2(self, fig, class_labels, datas_nb, datas_nb_tot):
        handles = [
            Patch(edgecolor='b', facecolor='none', hatch=self.hatch_label[i % len(self.hatch_label)])
            for i in range(len(class_labels))
        ]
        labels = [f"{label} - {datas_nb[i]}/{datas_nb_tot[i]} images" for i, label in enumerate(class_labels)]

        fig.legend(handles=handles, labels=labels,
                   loc='upper left', bbox_to_anchor=(0.01, 0.97), fontsize='xx-large', markerscale=2,
                   title="Classes", title_fontsize='xx-large')

        severities_legend_handles = [
            Patch(facecolor=self.color_label[h], edgecolor=self.color_label[h], label=f"Severity {h + 1}")
            for h in range(5)
        ]

        fig.legend(handles=severities_legend_handles, loc='upper left',
                   bbox_to_anchor=(0.01, 0.87 - (0.02 * len(class_labels))), fontsize='xx-large',
                   markerscale=2, title="Severities", title_fontsize='xx-large')
