import os
from datetime import datetime, timedelta
import platform
import cv2
import psutil
import numpy as np
import torchvision.transforms as transforms
import random


class ImageProcessor:
    def __init__(self, data_flag, directory_name_0, directory_name_1, data_nb, data_nb_tot, augmented_datasets_sev,
                 augmented_datasets_all, test_dataset, info, device, data_transform, model, correct_false_predictions):
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
        self.correct_false_predictions = correct_false_predictions
        self.rmses_saliency_all_tot = []

        # Initialize current and adjusted_time variables
        self.current, self.adjusted_time = self.adjust_time()

    def print_memory_usage(self, k):  # Print current memory usage along image index
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"\rImage number: {k + 1} / {self.data_nb}, memory used: {int(memory_info.rss / (1024 ** 2))} MB", end='',
              flush=True)

    def adjust_time(self):
        # Return both current and adjusted time
        current = datetime.now()
        adjusted_time = current + timedelta(hours=2)
        return current, adjusted_time

    def setup_paths(self, k, data_flag, directory_name_0, directory_name_1, label_class, label_class_name, sev,
                    augmented, bool):
        if platform.system() == 'Windows':  # Paths for Windows
            current_time = self.current.strftime('%Y-%m-%d-%H-%M-%S')
            directory_path_3 = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_1}'
            directory_name_2 = f'{k}'
            directory_name_3 = "metrics"
            output_path = fr'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_1}\{directory_name_2}'
            csv_rmses_path = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_3}\{data_flag}_sev{sev}{augmented}{bool}_rmses'
            csv_rmses_all_path = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_3}\{data_flag}_{label_class}_{label_class_name}_{self.data_nb}^{self.data_nb_tot}_images{augmented}{bool}_rmses_all'
            csv_auc_path = rf'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs\{data_flag}\{directory_name_0}\{directory_name_3}\{data_flag}{augmented}{bool}_auc'

        else:  # Paths for Linux
            current_time = self.adjusted_time.strftime('%Y-%m-%d-%H-%M-%S')
            directory_path_3 = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_1}'
            directory_name_2 = f'{k}'
            directory_name_3 = "metrics"
            output_path = fr'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_1}/{directory_name_2}'
            csv_rmses_path = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_3}/{data_flag}_sev{sev}{augmented}{bool}_rmses'
            csv_rmses_all_path = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_3}/{data_flag}_{label_class}_{label_class_name}_{self.data_nb}^{self.data_nb_tot}_images{augmented}{bool}_rmses_all'
            csv_auc_path = rf'/home/ptreyer/Outputs/{data_flag}/{directory_name_0}/{directory_name_3}/{data_flag}{augmented}{bool}_auc'

        return current_time, directory_name_2, directory_name_3, directory_path_3, output_path, csv_rmses_path, csv_rmses_all_path, csv_auc_path

    def select_image(self, correct_prediction, false_prediction, correct_false_predictions, randomized, index):
        # Select the desired image index based on user input
        if correct_prediction == True or false_prediction == True:
            if randomized:
                desired_image_index = random.choice(correct_false_predictions)
            else:
                desired_image_index = correct_false_predictions[index]
        else:
            if randomized:
                desired_image_index = random.randint(0, len(self.test_dataset))
            else:
                desired_image_index = index

        return desired_image_index

    def prepare_original_tensors(self, desired_image_index):
        # Format the image for different usage
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
        # Format the image with gradients for different usage
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
        # Identify and return target layers of the model
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
