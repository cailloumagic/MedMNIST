# Master's Thesis: Interpretability-based robustness analysis of medical image classification models.

This repository contains the implementation and analysis code for the thesis project titled "Interpretability-based Robustness Analysis of Medical Image Classification Models." The objective of this thesis is to explore and evaluate the robustness of medical image classification models with a focus on interpretability techniques.

# Description
Medical image classification is a critical task with significant implications in healthcare. Ensuring that these models are not only accurate but also robust and interpretable is essential. This thesis project delves into the robustness of these models by applying various perturbations to the input images and analyzing the model's responses using interpretability techniques such as saliency maps. The findings aim to contribute to the development of more reliable and trustworthy AI models in the medical domain.

# Key Features
- Data Augmentation: Implementation of extensive data augmentation techniques, including Gaussian noise, motion blur, and contrast adjustments, to simulate different levels of perturbations on medical images.
- Model Interpretability: Use of saliency maps to analyze the model's decision-making process and evaluate how perturbations impact the interpretability of the model.
- Robustness Analysis: Comprehensive analysis of model robustness through metrics such as Root Mean Squared Error (RMSE) and Area Under the Curve (AUC) across different perturbation severities.
- Automated Evaluation: Automated scripts for training, testing, and evaluating models on augmented datasets with detailed performance tracking.

# Code Structure
.
├── data_augmentation/
│   ├── __init__.py                   # Initialize data augmentation module
│   ├── TrainDatasetAugmentor.py      # Data augmentation for training dataset
│   └── TestDatasetAugmentor.py       # Data augmentation for testing dataset
├── models/
│   ├── __init__.py                   # Initialize models module
│   └── resnet.py                     # ResNet model definition
├── utils/
│   ├── __init__.py                   # Initialize utils module
│   └── helper_functions.py           # Utility functions (e.g., memory usage, time tracking)
├── core/
│   ├── ImageProcessor.py             # Image processing and tensor preparation
│   ├── ModelTester.py                # Model testing and evaluation
│   ├── HeatmapGenerator.py           # Heatmap generation and plotting
│   └── CSVManager.py                 # CSV management for saving results
├── main.py                           # Main script for executing the project
└── README.md                         # Project documentation


# Installation and Requirements
## Prerequisites
Before you can run the project, ensure you have the following installed:

Python 3.8+
PyTorch
torchvision
OpenCV
NumPy
tqdm
imgaug
psutil

# Installation

Installation

# Contribution
This project is part of a Master's thesis, but contributions and discussions are welcome. Feel free to open issues for any bugs or feature requests.

