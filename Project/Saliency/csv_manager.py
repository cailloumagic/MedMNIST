import os
import numpy as np
import pandas as pd


class CSVManager:
    def __init__(self, directory_path_1, directory_name_3, csv_rmses_all_path, csv_rmses_path, csv_auc_path):
        self.directory_path_1 = directory_path_1
        self.directory_name_3 = directory_name_3
        self.csv_rmses_all_path = csv_rmses_all_path
        self.csv_rmses_path = csv_rmses_path
        self.csv_auc_path = csv_auc_path
        self.create_directory()

    def create_directory(self):
        path = os.path.join(self.directory_path_1, self.directory_name_3)
        if not os.path.isdir(path):
            os.mkdir(path)

    def save_rmse_all_to_csv(self, slices_rmses_all_tot):
        df_saliency_all = pd.DataFrame(slices_rmses_all_tot)  # severities x (perturbations x layers)
        df_saliency_all.columns = list(range(8)) * 5

        if not os.path.isfile(self.csv_rmses_all_path):
            df_saliency_all.to_csv(self.csv_rmses_all_path, index=False)
        else:
            df_saliency_all.to_csv(self.csv_rmses_all_path, mode='a', index=False, header=False)

    def save_rmse_to_csv(self, rmses_saliency_tot, data_nb):
        rmses_saliency_tot /= data_nb  # makes the average over data_nb points (max=100)
        rmses_saliency_tot = np.round(rmses_saliency_tot, 3)
        slices_rmses_tot = [rmses_saliency_tot[i:i + 8] for i in range(0, len(rmses_saliency_tot), 8)]  # Slice RMSE values by layers

        df_saliency = pd.DataFrame(slices_rmses_tot)  # perturbations x layers
        df_saliency.columns = list(range(8))    # Set column names for the 8 layers

        if not os.path.isfile(self.csv_rmses_path):
            df_saliency.to_csv(self.csv_rmses_path, index=False)
            print("New CSV file created successfully.\n")

        else:
            df_saliency.to_csv(self.csv_rmses_path, mode='a', index=False, header=False)
            print("Data appended to CSV file successfully.\n")

    def save_auc_to_csv(self, deltas_auc):
        slices_auc = [deltas_auc[i:i + 5] for i in range(0, len(deltas_auc), 5)]  # len=30 --> 5x5
        df_auc = pd.DataFrame(slices_auc)  # severities x perturbations
        df_auc.columns = list(range(5))

        if not os.path.isfile(self.csv_auc_path):
            df_auc.to_csv(self.csv_auc_path, index=False)
            print("New CSV file created successfully.\n")
        else:
            df_auc.to_csv(self.csv_auc_path, mode='a', index=False, header=False)
            print("Data appended to CSV file successfully.\n")