import time
import cv2
import numpy as np
import torch
import torch.utils.data as data
from tqdm import tqdm
import imgaug.augmenters as iaa
from utils import get_memory_and_duration



class TrainDatasetAugmentor:
    def __init__(self, train_dataset, total_iterations, augmented_transform):
        self.train_dataset = train_dataset
        self.total_iterations = total_iterations
        self.augmented_transform = augmented_transform
        self.perturbations = [
            iaa.imgcorruptlike.GaussianNoise,
            iaa.imgcorruptlike.SpeckleNoise,
            iaa.imgcorruptlike.MotionBlur,
            iaa.imgcorruptlike.Contrast,
            iaa.imgcorruptlike.Brightness
        ]
        self.augmented_dataset = []

    def augment_image(self, image, perturbation):
        image_sq = np.squeeze(image)  # 64x64
        image_np = np.array(image_sq)  # 64x64
        image_uint8 = cv2.normalize(image_np, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)  # 64x64

        if image.shape[0] == 1:  # grey scale
            augmented_image = perturbation.augment_image(image_uint8)  # 64x64 Apply the perturbation to the grayscale image
            augmented_image_tensor = torch.from_numpy(augmented_image).unsqueeze(0)  # 1x64x64
        else:  # RGB
            augmented_image = perturbation.augment_image(np.transpose(image_uint8, (1, 2, 0)))  # 64x64 Apply the perturbation to the RGB image
            augmented_image_transpose = np.transpose(augmented_image, (2, 0, 1))
            augmented_image_tensor = torch.from_numpy(augmented_image_transpose)

        augmented_image_float = augmented_image_tensor.to(dtype=torch.float32)  # 1x64x64
        augmented_image_transform = self.augmented_transform(augmented_image_float)  # 64x64x1

        return augmented_image_transform    # Return the fully augmented and transformed image

    def augment_dataset(self):
        start_time = time.time()

        print("\nAugmentation of the training dataset...")
        with tqdm(total=self.total_iterations) as pbar:     # Progress bar to track augmentation progress
            for severities in range(5):
                perturbations_list = [p(severity=severities + 1) for p in self.perturbations]

                for image, label in self.train_dataset:
                    self.augmented_dataset.append((image, label))

                    for perturbation in perturbations_list:  # len=5  Apply each perturbation to the image
                        augmented_image = self.augment_image(image, perturbation)
                        self.augmented_dataset.append((augmented_image, label))
                        pbar.update(1)  # Update the progress bar after each augmentation

        memory_info, training_duration = get_memory_and_duration(start_time)
        print(
            f"Augmented training dataset generated, duration: {training_duration:.2f} seconds, memory used: {int(memory_info.rss / (1024 ** 2))} MB\n")

        return self.augmented_dataset


class TestDatasetAugmentor:
    def __init__(self, test_loader, test_dataset, BATCH_SIZE, augmented_transform, get_memory_and_duration):
        self.test_loader = test_loader
        self.test_dataset = test_dataset
        self.BATCH_SIZE = BATCH_SIZE
        self.augmented_transform = augmented_transform
        self.get_memory_and_duration = get_memory_and_duration
        self.perturbations = [
            iaa.imgcorruptlike.GaussianNoise,
            iaa.imgcorruptlike.SpeckleNoise,
            iaa.imgcorruptlike.MotionBlur,
            iaa.imgcorruptlike.Contrast,
            iaa.imgcorruptlike.Brightness
        ]

    def augment_image(self, image, perturbation):
        image_sq = np.squeeze(image)
        image_np = np.array(image_sq)
        image_uint8 = cv2.normalize(image_np, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)

        if image.shape[0] == 1:  # grayscale
            augmented_image = perturbation.augment_image(image_uint8)
            augmented_image_tensor = torch.from_numpy(augmented_image).unsqueeze(0)
        else:  # RGB
            augmented_image = perturbation.augment_image(np.transpose(image_uint8, (1, 2, 0)))
            augmented_image_transpose = np.transpose(augmented_image, (2, 0, 1))
            augmented_image_tensor = torch.from_numpy(augmented_image_transpose)

        augmented_image_float = augmented_image_tensor.to(dtype=torch.float32)
        augmented_image_transform = self.augmented_transform(augmented_image_float)

        return augmented_image_transform

    def augment_test_datasets(self, perturbations_names_2, sev):
        start_time = time.time()

        augmented_loaders_all = [self.test_loader]
        augmented_loaders_sev = [self.test_loader]
        augmented_datasets_all = [self.test_dataset]
        augmented_datasets_sev = [self.test_dataset]

        total_iterations = 5 * 5 * len(self.test_dataset)

        print("\nAugmentation of the testing dataset...")
        with tqdm(total=total_iterations) as pbar:
            for severities in range(1, 6):
                perturbations_list = [p(severity=severities) for p in self.perturbations]

                for i, perturbation, name in zip(range(5), perturbations_list, perturbations_names_2):
                    augmented_dataset = []

                    for image, label in self.test_dataset:
                        augmented_image = self.augment_image(image, perturbation)
                        augmented_dataset.append((augmented_image, label))
                        pbar.update(1)

                    augmented_loader = torch.utils.data.DataLoader(dataset=augmented_dataset, batch_size=self.BATCH_SIZE * 2,
                                                                   shuffle=False)

                    if severities == sev:
                        augmented_datasets_sev.append(augmented_dataset)

                    augmented_datasets_all.append(augmented_dataset)
                    augmented_loaders_all.append(augmented_loader)

                    if (len(augmented_datasets_all)) % 6 == 0 and len(augmented_datasets_all) < 30:
                        augmented_datasets_all.append(self.test_dataset)
                        augmented_loaders_all.append(self.test_loader)

        memory_info, training_duration = get_memory_and_duration(start_time)
        print(
            f"Augmented testing datasets generated, duration: {training_duration:.2f} seconds, memory used: {int(memory_info.rss / (1024 ** 2))} MB\n")

        return augmented_datasets_all, augmented_datasets_sev, augmented_loaders_all, augmented_loaders_sev