"""Model Handler Module.

This module provides the `ModelHandler` class for managing the training, evaluation, 
and inference processes of a machine learning model. It also includes functionality 
for saving and loading models.

Dependencies:
    - PyTorch (torch)
    - CNNModel: A custom-defined neural network architecture.
"""

import numpy as np
import torch

import torch.nn as nn
from cnn_model import CNNModel

import time

class ModelHandler:
    """Handles the model training, evaluation, and inference pipeline.

    Attributes:
        model (CNNModel): The machine learning model.
        device (torch.device): The device on which the model is executed (e.g., 'cpu' or 'cuda').
        model_path: Path to where .plt models should be saved.
    """
    
    def __init__(self, model_path: str | None):
        """Initializes the ModelHandler.

        Args:
            model_path (Optional[str]): Path to the pre-trained model file (if available).
        """
        self.model = CNNModel()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path

    def train(self, train_loader, val_loader, epochs: int, learning_rate: float,  ) -> None:
        """Trains the model.

        Args:
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            epochs (int): Number of training epochs.
        """

        self.model.to(self.device)

        # Initalize Optimizer
        loss_function = nn.BCEWithLogitsLoss() # Binary Cross Entropy loss function with sigmoid layer applied
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate, ) # Adam optimizer (Want to try SGD later)


        best_acc = -1
        for epoch in range(epochs):
            train_losses_epoch, val_losses_epoch = [], []

            self.model.train()
            for X_train, y_train, in train_loader:
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                y_train = y_train.float().unsqueeze(1) # Make sure we have correct shape for BCE loss
                y_prediction_train = self.model(X_train)
                
                loss = (y_prediction_train, y_train)
                train_losses_epoch.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_acc = self.evaluate(self.model, train_loader)
            train_loss = np.mean(train_losses_epoch)
            print(f'Train accuracy: {train_acc*100:.2f}% | Training loss: {train_loss:.4f}')

            self.model.eval()
            with torch.no_grad():
                for X_val, y_val, in val_loader:
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)
                    y_prediction_val = self.model(X_val)
                    loss = loss_function(y_prediction_val, y_val)
                    val_losses_epoch.append(loss.item())

            val_acc = self.evaluate(self.model, train_loader)        
            val_loss = np.mean(val_losses_epoch)
            print(f'Validation accuracy: {val_acc*100:.2f}% | Validation loss: {val_loss:.4f}')

            # Here we check weather we have the best model, and then save it if so
            if val_acc > best_acc:
                best_model_state = self.model.state_dict()
                best_acc = val_acc
        
        self.model.load_state_dict(best_model_state)
        self.save_model(best_model_state, path=f"{self.model_path}/model_LR:{learning_rate}_EPOCHS:{epochs}_{time.time()}.plt")
        

    def evaluate(self, test_loader) -> None:
        """Evaluates the model on the test dataset.

        Args:
            test_loader: DataLoader for the test dataset.
        """
        self.model.to(self.device)
        self.model.eval()
        batch_sizes, accs = [], []
        with torch.no_grad():
            for X_test, y_test, in test_loader:
                X_test.to(self.device)
                y_test.to(self.device)

                prediction = self.model(X_test)
                batch_sizes.append(X_test.shape[0])

                prediction = torch.sigmoid(prediction)
                prediction_classes = (prediction > 0.5).float() # This converts to binary classes 0 and 1

                acc = torch.mean((prediction_classes == y_test).float()).item()
                accuracy.append(acc)

        # Find average accuracy
        accuracy = np.average(accs, weights=batch_sizes)


    def predict(self, spectrogram: torch.Tensor, model_name: str) -> int:
        """Performs inference on a single spectrogram.

        Args:
            spectrogram (torch.Tensor): Input spectrogram for inference.

        Returns:
            torch.Tensor: The predicted output from the model.
        """
        self.load_model(self.model_path +f"/{model_name}")
        spectrogram = spectrogram.unsqueeze(0).to(self.device)

        with torch.no_grad:
            logits = self.model(spectrogram)

            probability = torch.sigmoid(logits)

            prediciton = (probability > 0.5).float() # Turn probability into binary classificaiton

        return prediciton.item()
        

    def save_model(self, model_name: str | None) -> None:
        """Saves the model to the specified file path.

        Args:
            path (str): Path to save the model file.
        """
        path = self.model_path + "/" + model_name
        torch.save(self.state_dict(), path)


    def load_model(self, path: str) -> None:
        """Loads a model from the specified file path.

        Args:
            path (str): Path to the model file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()
