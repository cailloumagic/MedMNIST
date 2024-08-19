import time
import torch.utils.data as data
import torchvision.transforms as transforms

from data_augmentation import TrainDatasetAugmentor
from utils import get_memory_and_duration


def preprocessing():
    # Data transformations to be applied to the datasets
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    augmented_transform = transforms.Normalize(mean=[0], std=[255])

    return data_transform, augmented_transform


def load_datasets_and_dataloaders(augmentation, data_transform, augmented_transform, DataClass, download, data_size, BATCH_SIZE):
    print("\nDataset loading...")
    start_time = time.time()

    # Load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download, size=data_size, mmap_mode='r')
    test_dataset = DataClass(split='test', transform=data_transform, download=download, size=data_size, mmap_mode='r')

    train_dataset_original = train_dataset

    memory_info, training_duration = get_memory_and_duration(start_time)
    print(f"Dataset loaded, duration: {training_duration:.2f} seconds, memory used: {int(memory_info.rss / (1024 ** 2))} MB")

    if augmentation:
        total_iterations = len(train_dataset) * 5 * 5  # 5 perturbations * 5 severities * number of images in train_dataset
        train_augmentor = TrainDatasetAugmentor(train_dataset, total_iterations, augmented_transform)
        train_dataset = train_augmentor.augment_dataset()   # Perform the augmentation

    # Encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # Training
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False) # Validation
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)   # Test

    return train_dataset, train_loader, train_loader_at_eval, test_dataset, test_loader, train_dataset_original
