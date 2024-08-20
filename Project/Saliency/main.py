import os
import re
from datetime import datetime, timedelta
import time
import cv2
import sys
import psutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import medmnist
from medmnist import Evaluator, INFO

# Importing custom modules
from data_processing import preprocessing, load_datasets_and_dataloaders
from data_augmentation import TestDatasetAugmentor
from model_architecture import ResNet18
from model_training import ModelTrainer
from evaluation import ModelTester
from visualization import plot_loss, plot_montage, plot_perturbations
from csv_manager import CSVManager
from heatmap import HeatmapGenerator
from image_processing import ImageProcessor
from config import *

start_script = time.time()
print("Device: ", device)

info = INFO[data_flag]
task = info['task']     # Type of task (e.g., binary, multi-label)
n_channels = info['n_channels']
n_classes = len(info['label'])
label_classes = [int(key) for key in info['label'].keys() if key.isdigit()]
DataClass = getattr(medmnist, info['python_class'])
current = datetime.now()
adjusted_time = current + timedelta(hours=2)

perturbations_names_2 = ['original', 'gaussian', 'speckle', 'motion_blurred', 'contrast', 'brightened']
perturbations_names_3 = 6 * perturbations_names_2
repeated_severities_list = [num for num in range(1, 6) for _ in range(6)]


# Check if the user parameters are valid
if correct_prediction == True and false_prediction == True:
    print("Correct and false predictions cannot be set to True at the same time.")
    sys.exit(1)

if (correct_prediction == True and randomized == False) or (false_prediction == True and randomized == False):
    print("Warning: The correct_prediction and false_prediction flags are set to True, but the randomized flag is set to False. Therefore the index parameter defined will not correspond to the index of the image.")


# Define the files name according to user parameters
if augmentation == True:
    augmented = '_augmented'
else:
    augmented = ''

if false_prediction == True:
    bool_flag = '_false'
else:
    bool_flag = ''


# Set paths for saving outputs based on the operating system
current_time_0 = current.strftime('%Y-%m-%d-%H-%M-%S') if platform.system() == 'Windows' else adjusted_time.strftime('%Y-%m-%d-%H-%M-%S')
directory_path_0 = rf'{base_output_dir}\{data_flag}' if platform.system() == 'Windows' else rf'{base_output_dir}/{data_flag}'
directory_name_0 = f'{current_time_0}_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}{bool_flag}'


# Create directory for the dataset
if save:
    path = os.path.join(directory_path_0)
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(directory_path_0, directory_name_0)
    os.mkdir(path)


# Preprocess and load the datasets and dataloaders
data_transform, augmented_transform = preprocessing()
train_dataset, train_loader, train_loader_at_eval, test_dataset, test_loader, train_dataset_original = load_datasets_and_dataloaders(
    augmentation, data_transform, augmented_transform, DataClass, download, data_size, BATCH_SIZE)


# Create augmented versions of the test dataset
test_augmentor = TestDatasetAugmentor(test_loader, test_dataset, BATCH_SIZE, augmented_transform,
                                      psutil.Process().memory_info)
augmented_datasets_all, augmented_datasets_sev, augmented_loaders_all, augmented_loaders_sev = test_augmentor.augment_test_datasets(
    perturbations_names_2, sev)

'''
print(train_dataset)
print("===================")
print(test_dataset)
'''

for label_class in label_classes:   # Loop through each label class
    number_epochs = NUM_EPOCHS

    label_class_name = info['label'][str(label_class)]
    label_class_name = re.sub(r'[ ,]+', '_', label_class_name)

    if platform.system() == 'Windows':
        directory_path_1 = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}'
        directory_name_1 = f"class_{label_class}_{label_class_name}"
    else:
        directory_path_1 = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}'
        directory_name_1 = f"class_{label_class}_{label_class_name}"

    if save and randomized == True:  # create a subdirectory to save the figures
        path = os.path.join(directory_path_1, directory_name_1)
        os.mkdir(path)

    rmses_saliency_tot = np.zeros(40)
    rmses_saliency_all_tot = []

    model = ResNet18(in_channels=n_channels, num_classes=n_classes)     # Instantiate the ResNet18 model
    if device == torch.device("cuda"):
        model = model.to(device)

    # define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    # Train the model
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        train_loader_at_eval=train_loader_at_eval,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        task=task,
        num_epochs=NUM_EPOCHS,
        patience=patience
    )
    train_losses, valid_losses, number_epochs = trainer.train()

    if platform.system() == 'Windows':
        directory_path_2 = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_1}'
    else:
        directory_path_2 = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_1}'


    if number_epochs != 1:  # Plot the training and validation loss
        plot_loss(train_losses, valid_losses, number_epochs, save, directory_path_1, directory_path_2, current_time_0,
                  data_flag, data_size, label_class_name)


    # Evaluate the model on original and perturbed datasets
    tester = ModelTester(
        model,
        device,
        train_loader,
        train_dataset,
        augmented_loaders_all,
        Evaluator,
        data_flag,
        data_size,
        task,
        augmentation,
        label_class,
        correct_prediction,
        false_prediction,
        randomized,
        perturbations_names_3,
        repeated_severities_list,
        sev
    )
    tester.evaluate('train')
    tester.evaluate('test')
    data_nb_tot, correct_false_predictions, predictions, results, results_sev, deltas_auc = tester.get_variable()

    if not correct_false_predictions:       # If no predictions made, skip to the next label class
        break

    # Determine the number of images to process based on the user input
    if randomized:
        if len(correct_false_predictions) > nb_images:
            data_nb = nb_images
        else:
            data_nb = len(correct_false_predictions)
    else:
        data_nb = 1


    for k in range(data_nb):  # data_nb represents the number of images to process

        # To manage the memory usage, paths, time, index and image selection
        processor = ImageProcessor(data_flag, directory_name_0, directory_name_1, data_nb, data_nb_tot,
                                   augmented_datasets_sev, augmented_datasets_all, test_dataset, info, device,
                                   data_transform, model, correct_false_predictions)
        processor.print_memory_usage(k)
        current_time, directory_name_2, directory_name_3, directory_path_3, output_path, csv_rmses_path, csv_rmses_all_path, csv_auc_path = processor.setup_paths(
            k, directory_name_0, directory_name_1, label_class, label_class_name, sev, augmented, bool_flag)
        desired_image_index = processor.select_image(correct_prediction, false_prediction, correct_false_predictions, randomized, index)
        original_class_name, original_image, original_image_uint8, original_tensors, original_tensors_uint8, images_sq, images_uint8, images_ssim = processor.prepare_original_tensors(
            desired_image_index)
        original_tensors_grads = processor.prepare_tensors_with_grads(desired_image_index)
        target_layer_list = processor.identify_target_layers()

        if randomized == True:
            correct_false_predictions.remove(desired_image_index)   # To not process the same image twice


        # Get the predicted class label and name for the selected image
        predicted_class_label = predictions[desired_image_index].item()
        predicted_class_name = info['label'][str(predicted_class_label)]

        # Create and plot a montage of images
        montage_20 = train_dataset_original.montage(length=20)
        montage_20_rgb = cv2.cvtColor(np.array(montage_20), cv2.COLOR_RGB2BGR)

        plot_montage(k, data_flag, data_size, images_sq, images_uint8, original_image, original_class_name,
                     montage_20_rgb, montage_20, save, directory_path_1, directory_name_2, directory_path_3, current_time, output_path,
                     strides, sev, augmented, bool_flag, randomized, method)


        # Initialize variables for RMSE calculation
        rmses_image = []
        rmses_saliency = []
        rmses_saliency_all = []
        rmses_saliency_all_tot = []
        slices_rmses_all_tot = []

        plot_perturbations(images_ssim, images_uint8, images_sq, original_image,
                           k, sev, save, directory_path_1, current_time, data_flag, data_size, strides,
                           augmented, bool_flag, method, output_path, randomized)


        # Instantiate the HeatmapGenerator to create and save heatmaps for the selected image
        heatmap_generator = HeatmapGenerator(data_flag, data_size, sev, desired_image_index, number_epochs, original_class_name,
                                              predicted_class_name, method, augmentation, perturbations_names_2, target_layer_list,
                                              model, original_tensors_grads, original_image, original_image_uint8, images_sq,
                                              results_sev, rmses_image, rmses_saliency_all, rmses_saliency, save, current_time,
                                              strides, augmented, bool_flag, output_path, directory_path_1, device)
        heatmap_generator.generate_and_save_heatmaps(k)

        # Calculate RMSE for the saliency maps
        rmses_saliency_tot, rmses_saliency_all_tot, slices_rmses_all_tot = heatmap_generator.update_rmse(
            rmses_saliency_tot, rmses_saliency_all_tot, data_nb)
        csv_manager = CSVManager(directory_path_1, directory_name_3, csv_rmses_all_path, csv_rmses_path, csv_auc_path)
        if csv == True:
            csv_manager.save_rmse_all_to_csv(slices_rmses_all_tot)

    if csv == True: # Save the Metrics to a CSV file
        csv_manager.save_auc_to_csv(deltas_auc)
        csv_manager.save_rmse_to_csv(rmses_saliency_tot, data_nb)

    if randomized == False:
        break

# Calculate and print the total duration of the script execution
end_script = time.time()
training_duration = end_script - start_script
hours, remainder = divmod(training_duration, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Duration of the script: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


