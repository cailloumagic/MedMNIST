import torch
from torchcam.methods import GradCAMpp

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
NUM_EPOCHS = 15
strides = 2
BATCH_SIZE = 64
lr = 0.001
patience = 10
sev = 1
nb_images = 6  # Number of images max to calculate heatmap
index = 1521  # Work only if randomized, correct_prediction, and false_prediction are False
