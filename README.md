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
* [`Project/`](Project/):
   * [`Plot/`](Project/Plot/):
       * [`data_processor.py`](Project/Plot/dataset.py): Prepares and processes data used by the plotting functions.
       * [`file_handler.py`](Project/Plot/file_handler.py): Handles file input/output operations, including reading and writing plot data.
       * [`plot_generator.py`](Project/Plot/plot_generator.py): Main script for creating and displaying plots based on processed data.
       * [`plotting.py`](Project/Plot/plotting.py): Supports plot generation by organizing and structuring plot configurations.
   * [`Saliency/`](Project/Saliency/):
       * [`__init__.py`](Project/Saliency/__init__.py): Initializes the Saliency module.
       * [`config.py`](Project/Saliency/config.py): Contains configuration settings and constants used throughout the project.
       * [`csv_manager.py`](Project/Saliency/csv_manager.py): Manages CSV operations, including saving and loading results.
       * [`data_augmentation.py`](Project/Saliency/data_augmentation.py): Implements data augmentation techniques for testing model robustness.
       * [`data_processing.py`](Project/Saliency/data_processing.py): Handles data loading, preprocessing, and transformations.
       * [`evaluation.py`](Project/Saliency/evaluation.py): Scripts for evaluating model performance on various datasets.
       * [`heatmap.py`](Project/Saliency/heatmap.py): Generates and processes heatmaps for interpretability and saliency analysis.
       * [`image_processing.py`](Project/Saliency/image_processing.py): Handles image transformations, processing, and preparation for analysis.
       * [`main.py`](Project/Saliency/main.py): Main execution script that orchestrates data loading, training, testing, and evaluation.
       * [`model_architecture.py`](Project/Saliency/model_architecture.py): Defines model architectures, including ResNet and other CNNs.
       * [`model_training.py`](Project/Saliency/model_training.py): Implements model training, validation, and early stopping logic.
       * [`utils.py`](Project/Saliency/utils.py): Provides utility functions for tasks such as memory tracking and time measurement.
       * [`visualization.py`](Project/Saliency/visualization.py): Tools for visualizing model interpretability results, such as saliency maps and heatmaps.


# Installation and Requirements
## Prerequisites
Before you can run the project, ensure you have the following installed:

- Python 3.8+
- tqdm
- opencv-python
- psutil
- numpy
- Pillow
- imgaug
- pandas
- torch
- torchvision
- matplotlib
- medmnist
- torchcam
- scikit-image



# Contribution
This project is part of a Master's thesis, but contributions and discussions are welcome. Feel free to open issues for any bugs or feature requests.

