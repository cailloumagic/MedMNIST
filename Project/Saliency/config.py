import torch
import platform
from torchcam.methods import GradCAMpp


# Define base output directory paths based on the operating system
if platform.system() == 'Windows':
    base_output_dir = r'C:\Users\pierr\Desktop\Cours\BME4\Coding\Outputs'
else:  # Path for Linux
    base_output_dir = r'/home/ptreyer/Outputs'


# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
download = True
save = True
csv = True
randomized = True
correct_prediction = True
false_prediction = False
augmentation = False

data_flag = 'breastmnist'
method = GradCAMpp
data_size = 64  # Size available 64, 128, and 224
NUM_EPOCHS = 10
strides = 2
BATCH_SIZE = 64
lr = 0.001
patience = 7
sev = 1
nb_images = 6  # Number of images max to calculate heatmap
index = 100  # Work only if randomized, correct_prediction, and false_prediction are False


