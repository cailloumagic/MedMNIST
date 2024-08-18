import numpy as np
import torch
import torch.nn.functional as F


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
        self.correct_false_predictions = []
        self.results = []
        self.results_sev = []
        self.deltas_auc = []


    def evaluate(self, split):
        if split == 'train':
            self._evaluate_train()  # Evaluate on the training set
        else:
            self._evaluate_augmented()  # Evaluate on the augmented datasets

    def _evaluate_train(self):  # Method to evaluate the model on the training dataset
        self.model.eval()
        y_true = torch.tensor([]).to(self.device)
        y_score = torch.tensor([]).to(self.device)
        data_loader = self.train_loader

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)    # Get model predictions

                if self.task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = torch.sigmoid(outputs)
                else:
                    targets = targets.squeeze().long()
                    outputs = F.softmax(outputs, dim=-1)

                y_true = torch.cat((y_true, targets), 0)    # Accumulate true labels
                y_score = torch.cat((y_score, outputs), 0)  # Accumulate predicted scores

            y_true = y_true.cpu().numpy()
            y_score = y_score.cpu().detach().numpy()

            if self.augmentation:
                original_labels = np.array([label for _, label in self.train_dataset])
                num_repeats = len(y_score) // len(original_labels) + 1
                duplicated_labels = np.tile(original_labels, num_repeats)[:len(y_score)].reshape(-1, 1) # Duplicate labels to match y_score size
                evaluator = self.evaluator_class(self.data_flag, 'train', size=self.data_size)
                evaluator.labels = duplicated_labels[:len(y_score)]  # Ensure labels match y_score size
            else:
                evaluator = self.evaluator_class(self.data_flag, 'train', size=self.data_size)
                evaluator.labels = y_true   # Use original labels if no augmentation

            metrics = evaluator.evaluate(y_score)   # Evaluate the model's predictions

            print('train auc: %.3f  acc:%.3f' % metrics)

    def _evaluate_augmented(self):
        loop = True     # Control variable to evaluate only on the original test dataset

        for i, loader in enumerate(self.augmented_loaders_all):
            self.model.eval()
            y_true = torch.tensor([]).to(self.device)
            y_score = torch.tensor([]).to(self.device)

            start_index = 0     # Index to keep track of data positions

            with torch.no_grad():
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)    # Get model predictions
                    targets = targets.squeeze().long()
                    _, predicted = torch.max(outputs, 1)    # Get the predicted classes
                    outputs_1 = F.softmax(outputs, dim=-1)  # Apply softmax to model outputs

                    if loop:    # During the first loop, record predictions and statistics
                        self.predictions.extend(predicted.cpu().numpy())

                        label_len = torch.sum(targets == self.label_class).item()   # Count the number of image in a specific label
                        self.data_nb_tot += label_len   # Update total count

                        correct_indices = ((targets == self.label_class) & (predicted == targets)).nonzero(as_tuple=True)[0]
                        correct_indices = (correct_indices + start_index).cpu().numpy()
                        incorrect_indices = ((targets == self.label_class) & (predicted != targets)).nonzero(as_tuple=True)[0]
                        incorrect_indices = (incorrect_indices + start_index).cpu().numpy()

                        self.correct_predictions.extend(correct_indices)
                        self.false_predictions.extend(incorrect_indices)

                        start_index += len(inputs)      # Update start index for the next batch to maintain correct position tracking

                    y_true = torch.cat((y_true, targets), 0)
                    y_score = torch.cat((y_score, outputs_1), 0)

            loop = False

            if self.correct_prediction:      # Determine if correct or false predictions should be evaluated
                self.correct_false_predictions = self.correct_predictions
            elif self.false_prediction:
                self.correct_false_predictions = self.false_predictions

            if not self.correct_false_predictions:
                print("No correct predictions made")
                break

            y_true_np = y_true.cpu().numpy()
            y_score_np = y_score.cpu().detach().numpy()

            evaluator = self.evaluator_class(self.data_flag, 'test', size=self.data_size)
            metrics = evaluator.evaluate(y_score_np)    # Calculate AUC and Accuracy metrics
            self.results.append((round(metrics[0], 3), round(metrics[1], 3)))

            # Store only the specified severity level in the parameters
            if (i >= ((self.sev - 1) * 6) and i < (self.sev * 6)):
                self.results_sev.append((round(metrics[0], 3), round(metrics[1], 3)))

            # Store the delta AUC values between original and pertubed datasets
            if i % 6 != 0:
                delta_auc = round(self.results[0][0] - self.results[i][0], 3)
                self.deltas_auc.append(delta_auc)

            # Print the results based on the current iteration and conditions
            if i == 0:
                print(f'test original  auc:%.3f  acc:%.3f\n' % metrics)
            elif i % 6 != 0 and (i - 5) % 6 != 0 and i != 29:
                print(f'test {self.perturbations_names_3[i]} sev:{self.repeated_severities_list[i]}  auc:%.3f  acc:%.3f' % metrics)
            elif (i - 5) % 6 == 0:
                print(f'test {self.perturbations_names_3[i]} sev:{self.repeated_severities_list[i]}  auc:%.3f  acc:%.3f\n' % metrics)
            elif i == 29:
                print(f'test {self.perturbations_names_3[i]} sev:{self.repeated_severities_list[i]}  auc:%.3f  acc:%.3f\n' % metrics)

    def get_variable(self):
        return self.data_nb_tot, self.correct_false_predictions, self.predictions, self.results, self.results_sev, self.deltas_auc

