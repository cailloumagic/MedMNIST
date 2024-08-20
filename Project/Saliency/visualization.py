import os
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from config import randomized

def plot_loss(train_losses, valid_losses, number_epochs, save, directory_path_1, directory_path_2, current_time_0,
              data_flag, data_size, label_class_name):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss\n{number_epochs} epochs')
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    tick_interval = max(1, len(train_losses) // 10)
    x_ticks = list(range(1, len(train_losses) + 1, tick_interval))
    if x_ticks[-1] != len(train_losses):  # Ensure the last epoch is included in the ticks
        x_ticks.append(len(train_losses))
    plt.xticks(x_ticks)

    if save and randomized == True:
        filename = f'{current_time_0}_{data_flag}_{data_size}x{data_size}_class_{label_class_name}_validation_loss.png'
        plt.savefig(os.path.join(directory_path_2, filename))
    elif save and randomized == False:
        filename = f'{current_time_0}_{data_flag}_{data_size}x{data_size}_validation_loss.png'
        plt.savefig(os.path.join(directory_path_1, filename))

    plt.close()


def plot_montage(k, data_flag, data_size, images_sq, images_uint8, original_image, original_class_name, montage_20_rgb,
                 montage_20, save, directory_path_1, directory_name_2, directory_path_3, current_time, output_path, strides, sev,
                 augmented, bool_flag, randomized, method):

    if k < 10:  # Display and save montages for the first 10 images only

        plt.figure(figsize=(10, 5))
        plt.suptitle(f'2D Dataset: {data_flag}, Size: {data_size}x{data_size}')

        # Display the original image
        plt.subplot(1, 2, 1)
        if original_image.shape[0] == 1:    # Grey scale
            plt.imshow(images_sq[0], cmap='gray')
        else:    # RGB
            plt.imshow(np.transpose(images_uint8[0], (1, 2, 0)))
        plt.axis('off')
        plt.title(f"Label: {original_class_name}")

        # Display the second montage image
        plt.subplot(1, 2, 2)
        if original_image.shape[0] == 1:    # Grey scale
            plt.imshow(montage_20_rgb)
        else:   # RGB
            plt.imshow(montage_20)
        plt.axis('off')
        plt.title('20x20 Images')

        # Save the figure with date and time in the filename
        if save and randomized == True:
            path = os.path.join(directory_path_3, directory_name_2)
            os.makedirs(path, exist_ok=True)
            filename = f'{current_time}_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}{bool_flag}_{method.__name__}_a_montage.png'
            plt.savefig(os.path.join(output_path, filename))

        elif save and randomized == False:
            filename = f'{current_time}_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}{bool_flag}_{method.__name__}_a_montage.png'
            plt.savefig(os.path.join(directory_path_1, filename))

        plt.close()



def plot_perturbations(images_ssim, images_uint8, images_sq, original_image,
                                 k, sev, save, directory_path_1, current_time, data_flag, data_size, strides,
                                 augmented, bool_flag, method, output_path, randomized):

    if k < 10:  # Display and save perturbations for the first 10 images only

        fig, axs = plt.subplots(5, 6, figsize=(11, 8))

        severity_labels = ['Severity 5', 'Severity 4', 'Severity 3', 'Severity 2', 'Severity 1']
        rmses_image = []

        # Loop through different severities
        for i in range(5):
            rmses_tot = []


            fig.text(0.02, (i + 0.5) / 5, severity_labels[i], ha='right', va='center', rotation='vertical', fontsize=12)

            # Define augmentation sequences with varying severity levels
            gaussian_noise = iaa.imgcorruptlike.GaussianNoise(severity=i + 1)
            speckle_noise = iaa.imgcorruptlike.SpeckleNoise(severity=i + 1)
            motion_blur = iaa.imgcorruptlike.MotionBlur(severity=i + 1)
            contrast = iaa.imgcorruptlike.Contrast(severity=i + 1)
            brightness = iaa.imgcorruptlike.Brightness(severity=i + 1)

            # Apply augmentation to the original image
            if original_image.shape[0] == 1:    # Grey scale
                augmented_gaussian_image = gaussian_noise.augment_image(images_ssim[0])  # 64x64x1
                augmented_speckle_image = speckle_noise.augment_image(images_ssim[0])  # 64x64x1
                augmented_motion_blurred_image = motion_blur.augment_image(images_ssim[0])  # 64x64x1
                augmented_contrast_image = contrast.augment_image(images_ssim[0])  # 64x64x1
                augmented_brightened_image = brightness.augment_image(images_ssim[0])  # 64x64x1
            else:   # RGB
                augmented_gaussian_image = gaussian_noise.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))
                augmented_speckle_image = speckle_noise.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))
                augmented_motion_blurred_image = motion_blur.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))
                augmented_contrast_image = contrast.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))
                augmented_brightened_image = brightness.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))

            augmented_images = [images_ssim[0], augmented_gaussian_image, augmented_speckle_image,
                                augmented_motion_blurred_image, augmented_contrast_image, augmented_brightened_image]

            augmented_labels = ["Original", "Gaussian Noise", "Speckle Noise", "Motion Blur", "Contrast", "Brightness"]

            # Loop through each perturbation to calculate RMSE and plot the images
            for j in range(6):
                mean_squared_error = mse(images_ssim[0], augmented_images[j])
                root_mean_squared_error = round(np.sqrt(mean_squared_error), 3)
                rmses_tot.append(root_mean_squared_error)   # Store RMSE

                if i + 1 == sev:
                    rmses_image.append(root_mean_squared_error) # Store RMSE for the specific severity level

                if j == 0:
                    # Plot the original image
                    if original_image.shape[0] == 1:
                        axs[i, 0].imshow(images_sq[0], cmap='gray')
                    else:
                        axs[i, 0].imshow(np.transpose(images_uint8[0], (1, 2, 0)))
                    if i == 0:
                        axs[i, 0].set_title("Original")
                    axs[i, 0].text(0.5, -0.15, f"RMSE: {rmses_tot[0]}",
                                   horizontalalignment='center', verticalalignment='center', fontsize=8,
                                   transform=axs[i, j].transAxes)
                    axs[i, 0].axis('off')
                else:
                    # Plot the augmented images
                    if original_image.shape[0] == 1:    # Grey scale
                        axs[i, j].imshow(np.squeeze(augmented_images[j]), cmap='gray')
                    else:   # RGB
                        axs[i, j].imshow(augmented_images[j])
                    if i == 0:
                        axs[i, j].set_title(augmented_labels[j])
                    axs[i, j].text(0.5, -0.15, f"RMSE: {rmses_tot[j]}",
                                   horizontalalignment='center', verticalalignment='center', fontsize=8,
                                   transform=axs[i, j].transAxes)
                    axs[i, j].axis('off')

        plt.tight_layout()      # Adjust layout to prevent overlap

        if save and randomized == True:
            filename = f'{current_time}_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}{bool_flag}_{method.__name__}_perturbations.png'
            fig.savefig(os.path.join(output_path, filename))

        elif save and randomized == False:
            filename = f'{current_time}_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}{bool_flag}_{method.__name__}_perturbations.png'
            plt.savefig(os.path.join(directory_path_1, filename))
        plt.close()

        plt.close(fig)
        del fig, axs