from tqdm import tqdm
import time
import torch

from utils import get_memory_and_duration



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
                self.counter = 0    # Reset early stopping counter
            else:
                self.counter += 1   # Increment early stopping counter
                if self.counter >= self.patience:
                    print(
                        f'Early stopping at epoch {epoch + 1} as validation loss did not improve for {self.patience} epochs.')
                    number_epochs = epoch + 1   # Capture the stopping epoch
                    break

                if valid_loss < 0.001:
                    print(f'Early stopping at epoch {epoch + 1} as validation loss is less than 0.001.')
                    number_epochs = epoch + 1   # Capture the stopping epoch
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

            total_loss += loss.item() * inputs.size(0)  # Accumulate total loss (batch loss * batch size)
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

        with torch.no_grad():   # Disable gradient computation for validation
            for inputs, targets in self.train_loader_at_eval:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                correct += self.compute_correct_prediction(outputs, targets)
                total += targets.size(0)

        accuracy = correct / total
        return total_loss / len(self.train_loader_at_eval.dataset), accuracy    # Return average loss and accuracy

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
            _, predicted = torch.max(outputs, 1)    # Get the class with the highest score
        return (predicted == targets).sum().item()  # Return the number of correct predictions