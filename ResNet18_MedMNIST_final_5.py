import os
import re
from datetime import datetime, timedelta
import platform
from tqdm import tqdm
import time
import random
import cv2
import psutil
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO, Evaluator
from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
from skimage.metrics import mean_squared_error as mse

start_script = time.time()

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

download = True
save = True
csv = True
randomized = True
correct_prediction = True
false_prediction = False
augmentation = False

data_flag = 'breastmnist'
method = GradCAMpp
data_size = 64  # Size available 64, 128 and 224
NUM_EPOCHS = 20
strides = 2
BATCH_SIZE = 64
lr = 0.001
patience = 10
sev = 1
nb_images = 11  # Number of images to calculate heatmap
index = 1521  # Work only if randomized, correct_prediction and false_prediction are False


# define a ResNET18 CNN model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


class ModelTrainer:
    def __init__(self, model, train_loader, train_loader_at_eval, optimizer, criterion, device, task, num_epochs,
                 patience):
        self.model = model
        self.train_loader = train_loader
        self.train_loader_at_eval = train_loader_at_eval
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.task = task
        self.num_epochs = num_epochs
        self.patience = patience
        self.train_losses = []
        self.valid_losses = []
        self.best_loss = float('inf')
        self.counter = 0

    def train(self):
        print("Training data loading...")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_one_epoch(epoch)
            valid_loss, valid_accuracy = self.validate_one_epoch()

            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(
                        f'Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.')
                    number_epochs = epoch + 1
                    break

                if valid_loss < 0.001:
                    print(f'Early stopping at epoch {epoch + 1} as validation loss is less than 0.001.')
                    number_epochs = epoch + 1
                    break

            number_epochs = epoch + 1

        memory_info, training_duration = get_memory_and_duration(start_time)
        print(
            f"Training done, duration: {training_duration:.2f} seconds, memory used: {int(memory_info.rss / (1024 ** 2))} MB\n")

        return self.train_losses, self.valid_losses, number_epochs

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        tqdm_iterator = tqdm(self.train_loader,
                             desc=f"Epoch {epoch + 1}/{self.num_epochs}, Train loss: {0:.3f}, Train Accuracy: {0:.3f}")

        for inputs, targets in tqdm_iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += self.compute_correct_prediction(outputs, targets)  # Accumulate correct predictions
            total += targets.size(0)  # Accumulate total samples

            # Update tqdm description with current loss and accuracy
            accuracy = correct / total
            tqdm_iterator.set_description(
                f"Epoch {epoch + 1}/{self.num_epochs}, Train loss: {total_loss / total:.3f}, Train Accuracy: {accuracy:.3f}")

        return total_loss / len(self.train_loader.dataset), accuracy

    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.train_loader_at_eval:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                correct += self.compute_correct_prediction(outputs, targets)
                total += targets.size(0)

        accuracy = correct / total
        return total_loss / len(self.train_loader_at_eval.dataset), accuracy

    def compute_loss(self, outputs, targets):
        if self.task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()
        return self.criterion(outputs, targets)

    def compute_correct_prediction(self, outputs, targets):
        if self.task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            predicted = torch.round(torch.sigmoid(outputs))
        else:
            targets = targets.squeeze().long()
            _, predicted = torch.max(outputs, 1)
        return (predicted == targets).sum().item()


class ModelTester:
    def __init__(self, model, device, train_loader, train_dataset, augmented_loaders_all, evaluator_class, data_flag, data_size, task, augmentation, label_class, correct_prediction, perturbations_names_3, repeated_severities_list, sev):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.augmented_loaders_all = augmented_loaders_all
        self.evaluator_class = evaluator_class
        self.data_flag = data_flag
        self.data_size = data_size
        self.task = task
        self.augmentation = augmentation

        self.label_class = label_class
        self.correct_prediction = correct_prediction
        self.perturbations_names_3 = perturbations_names_3
        self.repeated_severities_list = repeated_severities_list
        self.sev = sev

        self.data_nb_tot = 0
        self.predictions = []
        self.correct_predictions = []
        self.false_predictions = []
        self.results = []
        self.results_sev = []
        self.deltas_auc = []

    def evaluate(self, split):
        if split == 'train':
            self._evaluate_train()
        else:
            self._evaluate_augmented()

    def _evaluate_train(self):
        self.model.eval()
        y_true = torch.tensor([]).to(self.device)
        y_score = torch.tensor([]).to(self.device)
        data_loader = self.train_loader

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                if self.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = torch.sigmoid(outputs)
                else:
                    targets = targets.squeeze().long()
                    outputs = F.softmax(outputs, dim=-1)

                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)

            y_true = y_true.cpu().numpy()
            y_score = y_score.cpu().detach().numpy()

            if self.augmentation:
                original_labels = np.array([label for _, label in self.train_dataset])
                num_repeats = len(y_score) // len(original_labels) + 1
                duplicated_labels = np.tile(original_labels, num_repeats)[:len(y_score)].reshape(-1, 1)
                evaluator = self.evaluator_class(self.data_flag, 'train', size=self.data_size)
                evaluator.labels = duplicated_labels[:len(y_score)]  # Ensure labels match y_score size
            else:
                evaluator = self.evaluator_class(self.data_flag, 'train', size=self.data_size)
                evaluator.labels = y_true

            print("y_score:", y_score.shape[0])
            print("labels:", evaluator.labels.shape[0])

            metrics = evaluator.evaluate(y_score)

            print('train auc: %.3f  acc:%.3f' % metrics)

    def _evaluate_augmented(self):
        loop = True

        for i, loader in enumerate(self.augmented_loaders_all):
            self.model.eval()
            y_true = torch.tensor([]).to(self.device)
            y_score = torch.tensor([]).to(self.device)

            start_index = 0

            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    targets = targets.squeeze().long()
                    _, predicted = torch.max(outputs, 1)
                    outputs_1 = F.softmax(outputs, dim=-1)

                    if loop:
                        self.predictions.extend(predicted.cpu().numpy())

                        label_len = torch.sum(targets == self.label_class).item()
                        self.data_nb_tot += label_len

                        correct_indices = ((targets == self.label_class) & (predicted == targets)).nonzero(as_tuple=True)[0]
                        correct_indices = (correct_indices + start_index).cpu().numpy()

                        self.correct_predictions.extend(correct_indices)
                        incorrect_indices = [index for index in range(start_index, start_index + len(inputs)) if
                                             index not in correct_indices]
                        self.false_predictions.extend(incorrect_indices)

                        start_index += len(inputs)

                    y_true = torch.cat((y_true, targets), 0)
                    y_score = torch.cat((y_score, outputs_1), 0)

            loop = False

            if not self.correct_predictions and self.correct_prediction:
                print("No correct predictions made")
                break

            y_true_np = y_true.cpu().numpy()
            y_score_np = y_score.cpu().detach().numpy()

            evaluator = self.evaluator_class(self.data_flag, 'test', size=self.data_size)
            metrics = evaluator.evaluate(y_score_np)
            self.results.append((round(metrics[0], 3), round(metrics[1], 3)))

            if (i >= ((self.sev - 1) * 6) and i < (self.sev * 6)):
                self.results_sev.append((round(metrics[0], 3), round(metrics[1], 3)))

            if i % 6 != 0:
                delta_auc = round(self.results[0][0] - self.results[i][0], 3)
                self.deltas_auc.append(delta_auc)

            if i == 0:
                print(f'test original  auc:%.3f  acc:%.3f\n' % metrics)
            elif i % 6 != 0 and (i - 5) % 6 != 0 and i != 29:
                print(f'test {self.perturbations_names_3[i]} sev:{self.repeated_severities_list[i]}  auc:%.3f  acc:%.3f' % metrics)
            elif (i - 5) % 6 == 0:
                print(f'test {self.perturbations_names_3[i]} sev:{self.repeated_severities_list[i]}  auc:%.3f  acc:%.3f\n' % metrics)
            elif i == 29:
                print(f'test {self.perturbations_names_3[i]} sev:{self.repeated_severities_list[i]}  auc:%.3f  acc:%.3f\n' % metrics)

    def get_variable(self):
        return self.data_nb_tot, self.correct_predictions, self.predictions, self.results, self.results_sev, self.deltas_auc


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
            augmented_image = perturbation.augment_image(image_uint8)  # 64x64
            augmented_image_tensor = torch.from_numpy(augmented_image).unsqueeze(0)  # 1x64x64
        else:  # RGB
            augmented_image = perturbation.augment_image(np.transpose(image_uint8, (1, 2, 0)))  # 64x64
            augmented_image_transpose = np.transpose(augmented_image, (2, 0, 1))
            augmented_image_tensor = torch.from_numpy(augmented_image_transpose)

        augmented_image_float = augmented_image_tensor.to(dtype=torch.float32)  # 1x64x64
        augmented_image_transform = self.augmented_transform(augmented_image_float)  # 64x64x1

        return augmented_image_transform

    def augment_dataset(self):
        start_time = time.time()

        print("\nAugmentation of the training dataset...")
        with tqdm(total=self.total_iterations) as pbar:
            for severities in range(5):
                perturbations_list = [p(severity=severities + 1) for p in self.perturbations]

                for image, label in self.train_dataset:
                    self.augmented_dataset.append((image, label))

                    for perturbation in perturbations_list:  # len=5
                        augmented_image = self.augment_image(image, perturbation)
                        self.augmented_dataset.append((augmented_image, label))
                        pbar.update(1)

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

        if image.shape[0] == 1:  # grey scale
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

                    augmented_loader = torch.utils.data.DataLoader(dataset=augmented_dataset, batch_size=BATCH_SIZE * 2,
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


class ImageProcessor:
    def __init__(self, data_flag, directory_name_0, directory_name_1, data_nb, data_nb_tot, augmented_datasets_sev,
                 augmented_datasets_all, test_dataset, info, device, data_transform, model, correct_predictions,
                 false_predictions):
        self.data_flag = data_flag
        self.directory_name_0 = directory_name_0
        self.directory_name_1 = directory_name_1
        self.data_nb = data_nb
        self.data_nb_tot = data_nb_tot
        self.augmented_datasets_sev = augmented_datasets_sev
        self.augmented_datasets_all = augmented_datasets_all
        self.test_dataset = test_dataset
        self.info = info
        self.device = device
        self.data_transform = data_transform
        self.model = model
        self.correct_predictions = correct_predictions

        self.rmses_saliency_all_tot = []
        self.false_predictions = false_predictions

    def print_memory_usage(self, k):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"\rImage number: {k + 1} / {self.data_nb}, memory used: {int(memory_info.rss / (1024 ** 2))} MB", end='',
              flush=True)

    def adjust_time(self):
        current = datetime.now()
        adjusted_time = current + timedelta(hours=2)
        return current, adjusted_time

    def setup_paths(self, k, data_flag, directory_name_0, directory_name_1, label_class, label_class_name, sev,
                    augmented):

        if platform.system() == 'Windows':
            current_time = current.strftime('%Y-%m-%d-%H-%M-%S')
            directory_path_3 = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_1}'
            directory_name_2 = f'{k}'
            directory_name_3 = "metrics"
            output_path = fr'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_1}\{directory_name_2}'
            csv_rmses_path = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_3}\{data_flag}_sev{sev}{augmented}_rmses'
            csv_rmses_all_path = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_3}\{data_flag}_{label_class}_{label_class_name}_{data_nb}|{data_nb_tot}_images{augmented}_rmses_all'
            csv_auc_path = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_3}\{data_flag}{augmented}_auc'

        else:
            current_time = adjusted_time.strftime('%Y-%m-%d-%H-%M-%S')
            directory_path_3 = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_1}'
            directory_name_2 = f'{k}'
            directory_name_3 = "metrics"
            output_path = fr'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_1}/{directory_name_2}'
            csv_rmses_path = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_3}/{data_flag}_sev{sev}{augmented}_rmses'
            csv_rmses_all_path = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_3}/{data_flag}_{label_class}_{label_class_name}_{data_nb}|{data_nb_tot}_images{augmented}_rmses_all'
            csv_auc_path = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_3}/{data_flag}{augmented}_auc'

        return current_time, directory_name_2, directory_name_3, directory_path_3, output_path, csv_rmses_path, csv_rmses_all_path, csv_auc_path

    def select_image(self, correct_prediction, randomized, index):
        if correct_prediction:
            if randomized:
                desired_image_index = random.choice(self.correct_predictions)
            else:
                desired_image_index = self.correct_predictions[index]
        elif false_prediction:
            if randomized:
                desired_image_index = random.choice(self.false_predictions)
            else:
                desired_image_index = self.false_predictions[index]
        else:
            if randomized:
                desired_image_index = random.randint(0, len(self.test_dataset))
            else:
                desired_image_index = index

        return desired_image_index

    def prepare_original_tensors(self, desired_image_index):
        original_tensors = []
        original_tensors_uint8 = []
        images_sq = []
        images_uint8 = []
        images_ssim = []

        for datasets in self.augmented_datasets_sev:
            original_image, original_label = datasets[desired_image_index]  # 1x64x64
            original_image = original_image.cpu()  # 1x64x64
            original_image_sq = np.squeeze(original_image)  # 64x64
            original_image_np = np.array(original_image_sq)  # 64x64
            original_image_uint8 = cv2.normalize(original_image_np, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)  # 64x64
            original_image_ssim = cv2.normalize(np.array(original_image), None, 255, 0, cv2.NORM_MINMAX,
                                                cv2.CV_8U)  # 1x64x64
            original_image_pil = transforms.ToPILImage()(original_image)  # pil: 64x64

            images_sq.append(original_image_sq)
            images_uint8.append(original_image_uint8)
            images_ssim.append(np.transpose(original_image_ssim, (1, 2, 0)))
            original_label_scalar = original_label[0]
            original_class_name = self.info['label'][str(original_label_scalar.item())]

            original_tensor = self.data_transform(original_image_pil).unsqueeze(0).to(self.device)  # 1x1x64x64
            original_tensors.append(original_tensor)
            original_tensor_uint8 = original_tensor.squeeze(0).cpu().numpy().astype(np.uint8)  # 1x64x64
            original_tensors_uint8.append(np.transpose(original_tensor_uint8, (1, 2, 0)))

        return original_class_name, original_image, original_image_uint8, original_tensors, original_tensors_uint8, images_sq, images_uint8, images_ssim

    def prepare_tensors_with_grads(self, desired_image_index):
        original_tensors_grads = []

        for datasets in self.augmented_datasets_all:
            original_image, original_label = datasets[desired_image_index]
            original_image = original_image.cpu()
            original_image_pil = transforms.ToPILImage()(original_image)
            original_tensor = self.data_transform(original_image_pil).unsqueeze(0).to(self.device)
            original_tensor_grad = original_tensor.requires_grad_()  # 1x1x64x64
            original_tensors_grads.append(original_tensor_grad)  # len = 30

        return original_tensors_grads

    def identify_target_layers(self):
        target_layer_1_1 = self.model.layer1[-1].conv2
        target_layer_1_2 = self.model.layer1[-2].conv2
        target_layer_2_1 = self.model.layer2[-1].conv2
        target_layer_2_2 = self.model.layer2[-2].conv2
        target_layer_3_1 = self.model.layer3[-1].conv2
        target_layer_3_2 = self.model.layer3[-2].conv2
        target_layer_4_1 = self.model.layer4[-1].conv2
        target_layer_4_2 = self.model.layer4[-2].conv2

        return [target_layer_1_1, target_layer_1_2, target_layer_2_1, target_layer_2_2, target_layer_3_1,
                target_layer_3_2, target_layer_4_1, target_layer_4_2]


class HeatmapGenerator:
    def __init__(self, data_flag, data_size, sev, desired_image_index, number_epochs, original_class_name,
                 predicted_class_name, method, augmentation, perturbations_names_2, target_layer_list, model,
                 original_tensors_grads, original_image, original_image_uint8, images_sq, results_sev, rmses_image,
                 rmses_saliency_all, rmses_saliency, save, current_time, strides, augmented, output_path, device):
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
        self.output_path = output_path
        self.device = device
        self.heatmaps_original = []
        self.count = -1

    def generate_and_save_heatmaps(self, k):
        if k < 10:
            self._create_figure()

        for severities in range(1, 6):
            for i, names in zip(range(6), self.perturbations_names_2):
                heatmaps_normalized, heatmaps = self._process_layers(severities, i, names)

                if k < 10 and severities == self.sev:
                    self._plot_heatmaps(i, names, heatmaps_normalized, k)

        if k < 10:
            self._finalize_figure()

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
            heatmap, heatmap_normalized = self._generate_heatmap(j, target_layer)
            heatmaps.append(heatmap)
            heatmaps_normalized.append(heatmap_normalized)

            if i == 0:
                self.heatmaps_original.append(heatmap)

            if i != 0:
                self._calculate_rmse(heatmap, j, severities)

        return heatmaps_normalized, heatmaps

    def _generate_heatmap(self, j, target_layer):
        cam_extractor = self.method(self.model, target_layer)
        out = self.model(self.original_tensors_grads[self.count])
        heatmap = cam_extractor(out.squeeze(0).argmax().item(), out)
        heatmap = heatmap[0].squeeze(0)
        length = len(heatmap)
        heatmap = torch.reshape(heatmap, (length, length))
        heatmap_uint8 = cv2.normalize(heatmap.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Resize and normalize heatmap
        heatmap_resized = self._resize_heatmap(heatmap)
        heatmap_normalized = self._normalize_heatmap(heatmap_resized)

        return heatmap_uint8, heatmap_normalized

    def _resize_heatmap(self, heatmap):
        if self.device == torch.device("cuda"):
            heatmap_resized = np.array(Image.fromarray(heatmap.cpu().detach().numpy()).resize(
                (self.original_image.shape[2], self.original_image.shape[1])))
        else:
            heatmap_resized = np.array(
                Image.fromarray(heatmap.detach().numpy()).resize(
                    (self.original_image.shape[2], self.original_image.shape[1])))
        return heatmap_resized

    def _normalize_heatmap(self, heatmap_resized):
        min_val = np.min(heatmap_resized)
        max_val = np.max(heatmap_resized)
        if min_val == max_val:
            max_val += 1e-5
        heatmap_normalized = (heatmap_resized - min_val) / (max_val - min_val)
        return heatmap_normalized

    def _calculate_rmse(self, heatmap, j, severities):
        mean_squared_error = mse(self.heatmaps_original[j], heatmap)
        root_mean_squared_error = round(np.sqrt(mean_squared_error), 3)
        self.rmses_saliency_all.append(root_mean_squared_error)  # len=200

        if severities == self.sev:
            self.rmses_saliency.append(root_mean_squared_error)

    def _plot_heatmaps(self, i, names, heatmaps_normalized, k):
        for j, heatmap_normalized in enumerate(heatmaps_normalized):
            if j == 0:
                result_text = f"{names}\nAUC: {self.results_sev[i][0]}\nAcc: {self.results_sev[i][1]}\nRMSE: {self.rmses_image[i]}"
                self.axs[i, 0].text(-0.2, 0.5, result_text, va='center', ha='right',
                                    transform=self.axs[i, 0].transAxes)

            # Display heatmap overlay
            if self.original_image.shape[0] == 1:
                self.axs[i, j].imshow(self.images_sq[i], cmap='gray', alpha=0.3)
            else:
                self.axs[i, j].imshow(np.transpose(self.original_image_uint8, (1, 2, 0)), alpha=0.5)

            heat_map_cax = self.axs[i, j].imshow(heatmap_normalized, cmap='jet', alpha=0.5)
            self.axs[i, j].axis('off')
            if i == 0:
                self.axs[i, j].set_title(self.target_labels[j])

            # Increase size of individual subplots
            self.axs[i, j].set_aspect('equal', 'box')
            self.axs[i, j].set_xticks([])
            self.axs[i, j].set_yticks([])

    def _finalize_figure(self):
        plt.tight_layout()

        if self.save:
            filename = f'{self.current_time}_final_4_{self.data_flag}_{self.data_size}x{self.data_size}_stri{self.strides}_sev{self.sev}{self.augmented}_{self.method.__name__}_saliency.png'
            self.fig.savefig(os.path.join(self.output_path, filename))

        plt.close(self.fig)
        del self.fig, self.axs

    def update_rmse(self, rmses_saliency_tot, rmses_saliency_all_tot, data_nb):
        rmses_saliency_tot = np.add(rmses_saliency_tot, rmses_saliency)  # len = 40
        rmses_saliency_all_tot = np.append(rmses_saliency_all_tot, rmses_saliency_all)  # len = 200*data_nb
        slices_rmses_all_tot = [rmses_saliency_all_tot[i:i + 40] for i in range(0, len(rmses_saliency_all_tot), 40)]
        return rmses_saliency_tot, rmses_saliency_all_tot, slices_rmses_all_tot


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
        slices_rmses_tot = [rmses_saliency_tot[i:i + 8] for i in range(0, len(rmses_saliency_tot), 8)]

        df_saliency = pd.DataFrame(slices_rmses_tot)  # perturbations x layers
        df_saliency.columns = list(range(8))

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


def preprocessing():
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    augmented_transform = transforms.Normalize(mean=[0], std=[255])

    return data_transform, augmented_transform


def load_datasets_and_dataloaders(augmentation, data_transform, DataClass, download, data_size, BATCH_SIZE):
    print("\nDataset loading...")
    start_time = time.time()

    # Load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download, size=data_size, mmap_mode='r')
    test_dataset = DataClass(split='test', transform=data_transform, download=download, size=data_size, mmap_mode='r')

    train_dataset_original = train_dataset

    memory_info, training_duration = get_memory_and_duration(start_time)
    print(
        f"Dataset loaded, duration: {training_duration:.2f} seconds, memory used: {int(memory_info.rss / (1024 ** 2))} MB")

    if augmentation:
        total_iterations = len(
            train_dataset) * 5 * 5  # 5 perturbations * 5 severities * number of images in train_dataset
        train_augmentor = TrainDatasetAugmentor(train_dataset, total_iterations, augmented_transform)
        train_dataset = train_augmentor.augment_dataset()

    # Encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)

    return train_dataset, train_loader, train_loader_at_eval, test_dataset, test_loader, train_dataset_original


def plot_loss(train_losses, valid_losses, number_epochs, save, directory_path, current_time,
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

    if save and directory_path:
        filename = f'{current_time}_{data_flag}_{data_size}x{data_size}_class_{label_class_name}_validation_loss.png'
        plt.savefig(os.path.join(directory_path, filename))

    plt.close()


def plot_montage(k, data_flag, data_size, images_sq, images_uint8, original_image, original_class_name, montage_20_rgb,
                 montage_20, save, directory_path_3, directory_name_2, current_time, output_path, strides, sev,
                 augmented, method):
    if k < 10:
        # Create a figure
        plt.figure(figsize=(10, 5))
        plt.suptitle(f'2D Dataset: {data_flag}, Size: {data_size}x{data_size}')

        # Display the original image
        plt.subplot(1, 2, 1)
        if original_image.shape[0] == 1:
            plt.imshow(images_sq[0], cmap='gray')
        else:
            plt.imshow(np.transpose(images_uint8[0], (1, 2, 0)))
        plt.axis('off')
        plt.title(f"Label: {original_class_name}")

        # Display the second montage image
        plt.subplot(1, 2, 2)
        if original_image.shape[0] == 1:
            plt.imshow(montage_20_rgb)
        else:
            plt.imshow(montage_20)
        plt.axis('off')
        plt.title('20x20 Images')

        # Save the figure with date and time in the filename
        if save:
            path = os.path.join(directory_path_3, directory_name_2)
            os.makedirs(path, exist_ok=True)
            filename = f'{current_time}_final_4_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}_{method.__name__}_a_montage.png'
            plt.savefig(os.path.join(output_path, filename))

        plt.close()



def plot_perturbations(images_ssim, images_uint8, images_sq, original_image,
                                 k, sev, save, current_time, data_flag, data_size, strides,
                                 augmented, method, output_path):

    if k < 10:
        # Create a new figure to display the images
        fig, axs = plt.subplots(5, 6, figsize=(11, 8))

        severity_labels = ['Severity 5', 'Severity 4', 'Severity 3', 'Severity 2', 'Severity 1']

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
            if original_image.shape[0] == 1:
                augmented_gaussian_image = gaussian_noise.augment_image(images_ssim[0])  # 64x64x1
                augmented_speckle_image = speckle_noise.augment_image(images_ssim[0])  # 64x64x1
                augmented_motion_blurred_image = motion_blur.augment_image(images_ssim[0])  # 64x64x1
                augmented_contrast_image = contrast.augment_image(images_ssim[0])  # 64x64x1
                augmented_brightened_image = brightness.augment_image(images_ssim[0])  # 64x64x1
            else:
                augmented_gaussian_image = gaussian_noise.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))
                augmented_speckle_image = speckle_noise.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))
                augmented_motion_blurred_image = motion_blur.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))
                augmented_contrast_image = contrast.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))
                augmented_brightened_image = brightness.augment_image(np.transpose(images_uint8[0], (1, 2, 0)))

            augmented_images = [images_ssim[0], augmented_gaussian_image, augmented_speckle_image,
                                augmented_motion_blurred_image, augmented_contrast_image, augmented_brightened_image]

            augmented_labels = ["Original", "Gaussian Noise", "Speckle Noise", "Motion Blur", "Contrast", "Brightness"]

            for j in range(6):
                mean_squared_error = mse(images_ssim[0], augmented_images[j])
                root_mean_squared_error = round(np.sqrt(mean_squared_error), 3)
                rmses_tot.append(root_mean_squared_error)

                if i + 1 == sev:
                    rmses_image.append(root_mean_squared_error)

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
                    if original_image.shape[0] == 1:
                        axs[i, j].imshow(np.squeeze(augmented_images[j]), cmap='gray')
                    else:
                        axs[i, j].imshow(augmented_images[j])
                    if i == 0:
                        axs[i, j].set_title(augmented_labels[j])
                    axs[i, j].text(0.5, -0.15, f"RMSE: {rmses_tot[j]}",
                                   horizontalalignment='center', verticalalignment='center', fontsize=8,
                                   transform=axs[i, j].transAxes)
                    axs[i, j].axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        if save:
            filename = f'{current_time}_final_4_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}_{method.__name__}_perturbations.png'
            fig.savefig(os.path.join(output_path, filename))

        plt.close(fig)
        del fig, axs


def get_memory_and_duration(start_time):
    end_time = time.time()
    training_duration = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info, training_duration



info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
label_classes = [int(key) for key in info['label'].keys() if key.isdigit()]
DataClass = getattr(medmnist, info['python_class'])
current = datetime.now()
adjusted_time = current + timedelta(hours=2)

perturbations_names_2 = ['original', 'gaussian', 'speckle', 'motion_blurred', 'contrast', 'brightened']
perturbations_names_3 = 6 * perturbations_names_2
repeated_severities_list = [num for num in range(1, 6) for _ in range(6)]

if augmentation == True:
    augmented = '_augmented'
else:
    augmented = ''

if platform.system() == 'Windows':
    current_time_0 = current.strftime('%Y-%m-%d-%H-%M-%S')
    directory_path_0 = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}'
    directory_name_0 = f'{current_time_0}_final_4_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}'
else:
    current_time_0 = adjusted_time.strftime('%Y-%m-%d-%H-%M-%S')
    directory_path_0 = rf'/home/ptreyer/Outputs/{data_flag}'
    directory_name_0 = f'{current_time_0}_final_4_{data_flag}_{data_size}x{data_size}_stri{strides}_sev{sev}{augmented}'

# Create directory for the dataset
if save:
    path = os.path.join(directory_path_0)
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(directory_path_0, directory_name_0)
    os.mkdir(path)

data_transform, augmented_transform = preprocessing()
train_dataset, train_loader, train_loader_at_eval, test_dataset, test_loader, train_dataset_original = load_datasets_and_dataloaders(
    augmentation, data_transform, DataClass, download, data_size, BATCH_SIZE)

test_augmentor = TestDatasetAugmentor(test_loader, test_dataset, BATCH_SIZE, augmented_transform,
                                      psutil.Process().memory_info)
augmented_datasets_all, augmented_datasets_sev, augmented_loaders_all, augmented_loaders_sev = test_augmentor.augment_test_datasets(
    perturbations_names_2, sev)

'''
print(train_dataset)
print("===================")
print(test_dataset)
'''

for label_class in label_classes:
    number_epochs = NUM_EPOCHS

    label_class_name = info['label'][str(label_class)]
    label_class_name = re.sub(r'[ ,]+', '_', label_class_name)

    if platform.system() == 'Windows':
        directory_path_1 = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}'
        directory_name_1 = f"class_{label_class}_{label_class_name}"
    else:
        directory_path_1 = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}'
        directory_name_1 = f"class_{label_class}_{label_class_name}"

    if save:  # create a subdirectory to save the figures
        path = os.path.join(directory_path_1, directory_name_1)
        os.mkdir(path)

    rmses_saliency_tot = np.zeros(40)
    rmses_saliency_all_tot = []

    model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    if device == torch.device("cuda"):
        model = model.to(device)

    # define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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

    if number_epochs != 1:
        plot_loss(train_losses, valid_losses, number_epochs, save, directory_path_2,
                  current_time_0, data_flag, data_size, label_class_name)

    # Test the model on different datasets
    tester = ModelTester(model, device, train_loader, train_dataset, augmented_loaders_all, Evaluator, data_flag,
                         data_size, task, augmentation, label_class, correct_prediction, perturbations_names_3,
                         repeated_severities_list, sev)
    tester.evaluate('train')
    tester.evaluate('test')
    data_nb_tot, correct_predictions, false_predictions, predictions, results, results_sev, deltas_auc = tester.get_variable()

    if len(correct_predictions) > nb_images:
        data_nb = nb_images
    else:
        data_nb = len(correct_predictions)

    for k in range(data_nb):  # data_nb = nb_images, data points

        processor = ImageProcessor(data_flag, directory_name_0, directory_name_1, data_nb, data_nb_tot,
                                   augmented_datasets_sev, augmented_datasets_all, test_dataset, info, device,
                                   data_transform, model, correct_predictions, false_predictions)
        processor.print_memory_usage(k)
        current_time, directory_name_2, directory_name_3, directory_path_3, output_path, csv_rmses_path, csv_rmses_all_path, csv_auc_path = processor.setup_paths(
            k, data_flag, directory_name_0, directory_name_1, label_class, label_class_name, sev, augmented)
        desired_image_index = processor.select_image(correct_prediction, randomized, index)
        original_class_name, original_image, original_image_uint8, original_tensors, original_tensors_uint8, images_sq, images_uint8, images_ssim = processor.prepare_original_tensors(
            desired_image_index)
        original_tensors_grads = processor.prepare_tensors_with_grads(desired_image_index)
        target_layer_list = processor.identify_target_layers()

        correct_predictions.remove(desired_image_index)

        predicted_class_label = predictions[desired_image_index].item()
        predicted_class_name = info['label'][str(predicted_class_label)]

        # Create montage of images
        montage_20 = train_dataset_original.montage(length=20)
        montage_20_rgb = cv2.cvtColor(np.array(montage_20), cv2.COLOR_RGB2BGR)

        plot_montage(k, data_flag, data_size, images_sq, images_uint8, original_image, original_class_name,
                     montage_20_rgb, montage_20, save, directory_path_3, directory_name_2, current_time,
                     output_path, strides, sev, augmented, method)


        rmses_image = []
        rmses_saliency = []
        rmses_saliency_all = []
        rmses_saliency_all_tot = []
        slices_rmses_all_tot = []

        plot_perturbations(images_ssim, images_uint8, images_sq, original_image,
                                     k, sev, save, current_time, data_flag, data_size, strides,
                                     augmented, method, output_path)


        heatmap_generator = HeatmapGenerator(data_flag, data_size, sev, desired_image_index, number_epochs, original_class_name,
                                              predicted_class_name, method, augmentation, perturbations_names_2, target_layer_list,
                                              model, original_tensors_grads, original_image, original_image_uint8, images_sq,
                                              results_sev, rmses_image, rmses_saliency_all, rmses_saliency, save, current_time,
                                              strides, augmented, output_path, device)
        heatmap_generator.generate_and_save_heatmaps(k)
        rmses_saliency_tot, rmses_saliency_all_tot, slices_rmses_all_tot = heatmap_generator.update_rmse(
            rmses_saliency_tot, rmses_saliency_all_tot, data_nb)

        csv_manager = CSVManager(directory_path_1, directory_name_3, csv_rmses_all_path, csv_rmses_path, csv_auc_path)

        if csv == True:
            csv_manager.save_rmse_all_to_csv(slices_rmses_all_tot)

    if csv == True:
        csv_manager.save_auc_to_csv(deltas_auc)
        csv_manager.save_rmse_to_csv(rmses_saliency_tot, data_nb)

end_script = time.time()
training_duration = end_script - start_script
hours, remainder = divmod(training_duration, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Duration of the script: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


