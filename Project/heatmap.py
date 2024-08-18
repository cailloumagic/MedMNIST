import os
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse

class HeatmapGenerator:
    def __init__(self, data_flag, data_size, sev, desired_image_index, number_epochs, original_class_name,
                 predicted_class_name, method, augmentation, perturbations_names_2, target_layer_list, model,
                 original_tensors_grads, original_image, original_image_uint8, images_sq, results_sev, rmses_image,
                 rmses_saliency_all, rmses_saliency, save, current_time, strides, augmented, bool, output_path, device):
        self.data_flag = data_flag
        self.data_size = data_size
        self.sev = sev
        self.desired_image_index = desired_image_index
        self.number_epochs = number_epochs
        self.original_class_name = original_class_name
        self.predicted_class_name = predicted_class_name
        self.method = method
        self.augmentation = augmentation
        self.perturbations_names_2 = perturbations_names_2
        self.target_layer_list = target_layer_list
        self.model = model
        self.original_tensors_grads = original_tensors_grads
        self.original_image = original_image
        self.original_image_uint8 = original_image_uint8
        self.images_sq = images_sq
        self.results_sev = results_sev
        self.rmses_image = rmses_image
        self.rmses_saliency_all = rmses_saliency_all
        self.rmses_saliency = rmses_saliency
        self.save = save
        self.current_time = current_time
        self.strides = strides
        self.augmented = augmented
        self.bool = bool
        self.output_path = output_path
        self.device = device
        self.heatmaps_original = []
        self.count = -1

    def generate_and_save_heatmaps(self, k):
        if k < 10:      # Display and save heatmaps for the first 10 images only
            self._create_figure()

        for severities in range(1, 6):
            for i, names in zip(range(6), self.perturbations_names_2):
                heatmaps_normalized, heatmaps = self._process_layers(severities, i, names)  # Generate heatmaps

                if k < 10 and severities == self.sev:
                    self._plot_heatmaps(i, names, heatmaps_normalized, k)   # Plot the heatmaps

        if k < 10:
            self._finalize_figure()     # Finalize and save the figure

    def _create_figure(self):
        self.fig, self.axs = plt.subplots(6, 8, figsize=(15, 10))
        plt.suptitle(
            f'2D Dataset: {self.data_flag} - Size: {self.data_size}x{self.data_size} - Severity: {self.sev} - Idx: {self.desired_image_index} - Epochs: {self.number_epochs}\nTrue label: {self.original_class_name} - Predicted label: {self.predicted_class_name}\nMethod: {self.method.__name__} - Augmentation: {self.augmentation}')
        self.target_labels = ['Layer 1_1', 'Layer 1_2', 'Layer 2_1', 'Layer 2_2', 'Layer 3_1', 'Layer 3_2', 'Layer 4_1',
                              'Layer 4_2']

    def _process_layers(self, severities, i, names):
        heatmaps_normalized = []
        heatmaps = []
        self.count += 1

        for j, target_layer in enumerate(self.target_layer_list):
            heatmap, heatmap_normalized = self._generate_heatmap(j, target_layer)   # Generate heatmap
            heatmaps.append(heatmap)    # Store heatmap
            heatmaps_normalized.append(heatmap_normalized)   # Store normalized heatmap

            if i == 0:      # Store the heatmap of the original test dataset
                self.heatmaps_original.append(heatmap)

            if i != 0:      # calculate RMSE between original and current heatmap
                self._calculate_rmse(heatmap, j, severities)

        return heatmaps_normalized, heatmaps

    def _generate_heatmap(self, j, target_layer):
        cam_extractor = self.method(self.model, target_layer)   # Initialize CAM extractor
        out = self.model(self.original_tensors_grads[self.count])   # Model Output
        heatmap = cam_extractor(out.squeeze(0).argmax().item(), out)    # Generate heatmap
        heatmap = heatmap[0].squeeze(0)
        length = len(heatmap)
        heatmap = torch.reshape(heatmap, (length, length))  # Reshape the heatmap to square format
        heatmap_uint8 = cv2.normalize(heatmap.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Resize and normalize heatmap
        heatmap_resized = self._resize_heatmap(heatmap)
        heatmap_normalized = self._normalize_heatmap(heatmap_resized)

        return heatmap_uint8, heatmap_normalized

    def _resize_heatmap(self, heatmap):
        # Resize the heatmap to match the original image dimensions
        if self.device == torch.device("cuda"):
            heatmap_resized = np.array(Image.fromarray(heatmap.cpu().detach().numpy()).resize(
                (self.original_image.shape[2], self.original_image.shape[1])))
        else:
            heatmap_resized = np.array(
                Image.fromarray(heatmap.detach().numpy()).resize(
                    (self.original_image.shape[2], self.original_image.shape[1])))
        return heatmap_resized

    def _normalize_heatmap(self, heatmap_resized):
        # Normalize the resized heatmap to the range [0, 1]
        min_val = np.min(heatmap_resized)
        max_val = np.max(heatmap_resized)
        if min_val == max_val:
            max_val += 1e-5
        heatmap_normalized = (heatmap_resized - min_val) / (max_val - min_val)
        return heatmap_normalized

    def _calculate_rmse(self, heatmap, j, severities):
        # Calculate RMSE between the original heatmap and the current heatmap
        mean_squared_error = mse(self.heatmaps_original[j], heatmap)
        root_mean_squared_error = round(np.sqrt(mean_squared_error), 3)
        self.rmses_saliency_all.append(root_mean_squared_error)  # len=200

        if severities == self.sev:
            self.rmses_saliency.append(root_mean_squared_error) # Store RMSE for the specific severity level

    def _plot_heatmaps(self, i, names, heatmaps_normalized, k):

        for j, heatmap_normalized in enumerate(heatmaps_normalized):
            if j == 0:     # First column (layer), add text with AUC and ACC
                result_text = f"{names}\nAUC: {self.results_sev[i][0]}\nAcc: {self.results_sev[i][1]}\nRMSE: {self.rmses_image[i]}"
                self.axs[i, 0].text(-0.2, 0.5, result_text, va='center', ha='right',
                                    transform=self.axs[i, 0].transAxes)

            # Display heatmap overlay
            if self.original_image.shape[0] == 1:   # Grey scale
                self.axs[i, j].imshow(self.images_sq[i], cmap='gray', alpha=0.3)    # Display the image
            else:   # RGB
                self.axs[i, j].imshow(np.transpose(self.original_image_uint8, (1, 2, 0)), alpha=0.5)    # Display the image

            self.axs[i, j].imshow(heatmap_normalized, cmap='jet', alpha=0.5)    # Display the heatmap
            self.axs[i, j].axis('off')
            if i == 0:      # First row, set title
                self.axs[i, j].set_title(self.target_labels[j])

            # Adjust the aspect ratio and remove ticks for clean visualization
            self.axs[i, j].set_aspect('equal', 'box')
            self.axs[i, j].set_xticks([])
            self.axs[i, j].set_yticks([])

    def _finalize_figure(self):
        plt.tight_layout()

        if self.save:
            filename = f'{self.current_time}_final_4_{self.data_flag}_{self.data_size}x{self.data_size}_stri{self.strides}_sev{self.sev}{self.augmented}{self.bool}_{self.method.__name__}_saliency.png'
            self.fig.savefig(os.path.join(self.output_path, filename))

        plt.close(self.fig)
        del self.fig, self.axs

    def update_rmse(self, rmses_saliency_tot, rmses_saliency_all_tot, data_nb):
        rmses_saliency_tot = np.add(rmses_saliency_tot, self.rmses_saliency)  # len = 40
        rmses_saliency_all_tot = np.append(rmses_saliency_all_tot, self.rmses_saliency_all)  # len = 200*data_nb

        # Slice the RMSE array into segments of 40 (for each image)
        slices_rmses_all_tot = [rmses_saliency_all_tot[i:i + 40] for i in range(0, len(rmses_saliency_all_tot), 40)]
        return rmses_saliency_tot, rmses_saliency_all_tot, slices_rmses_all_tot