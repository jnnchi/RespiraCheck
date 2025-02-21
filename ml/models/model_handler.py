"""Model Handler Module.

This module provides the `ModelHandler` class for managing the training, evaluation, 
and inference processes of a machine learning model. It also includes functionality 
for saving and loading models.

Dependencies:
    - PyTorch (torch)
    - CNNModel: A custom-defined neural network architecture.
"""

import numpy as np
import collections
import time
import sys
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from cnn_model import CNNModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.audio_processor import AudioProcessor 
from data_processing.spectrogram_processor import SpectrogramProcessor
from data_processing.data_pipeline import DataPipeline


class ModelHandler:
    """Handles the model training, evaluation, and inference pipeline.

    Attributes:
        model (CNNModel): The machine learning model.
        device (torch.device): The device on which the model is executed (e.g., 'cpu' or 'cuda').
        model_path: Path to where .plt models should be saved.
    """
    
    def __init__(self, model, model_path: str, optimizer: torch.optim.Optimizer, loss_function: nn.Module):
        """Initializes the ModelHandler.

        Args:
            model_path (str | None): Path to the pre-trained model file (if available).
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.optimizer = optimizer
        self.loss_function = loss_function

 
    import numpy as np

    def train(self, train_loader, epochs: int, model_name: str) -> None:
        """Trains the model

        Args:
            train_loader: DataLoader for the training dataset.
            epochs (int): Number of training epochs.
            model_name (str): Name to save the trained model.
        """

        self.model.to(self.device)

        for epoch in range(epochs):
            train_losses_epoch = []

            self.model.train()
            for X_train, y_train in train_loader:
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                y_train = y_train.float().unsqueeze(1)  # Ensure correct shape for BCE loss
                y_prediction_train = self.model(X_train)
                
                # Compute loss
                loss = self.loss_function(y_prediction_train, y_train) 
                train_losses_epoch.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Evaluate model after each epoch
            train_acc = self.evaluate(train_loader)
            train_loss = np.mean(train_losses_epoch)
            print(f'Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc*100:.2f}% | Training Loss: {train_loss:.4f} | LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        self.save_model(model_state_dict=self.model.state_dict(), model_name=model_name)



    def validate(self, val_loader, hyperparams: dict, save_best: bool = True) -> tuple[float, float]:
        """Validates the model on the validation dataset.

        Args:
            val_loader: DataLoader for the validation dataset.

        Returns:
            tuple: (validation accuracy, validation loss)
        """
        
        self.model.to(self.device)
        self.model.eval() 

        val_losses_epoch, batch_sizes, accs = [], [], []
        best_acc = -1
        best_model_state = None  # Track the best model weights

        with torch.no_grad(): 
            for X_val, y_val in val_loader:
                X_val = X_val.to(self.device)
                y_val = y_val.to(self.device).float().unsqueeze(1)  

                y_prediction_val = self.model(X_val)  # forward pass
                loss = self.loss_function(y_prediction_val, y_val) 
                val_losses_epoch.append(loss.item())

                # Compute accuracy
                y_prediction_val = torch.sigmoid(y_prediction_val)  # Convert logits to probabilities
                prediction_classes = (y_prediction_val > 0.5).float()  # Convert to binary 0/1

                acc = torch.mean((prediction_classes == y_val).float()).item()
                accs.append(acc)
                batch_sizes.append(X_val.shape[0])

        # Compute final validation loss and accuracy
        val_loss = np.mean(val_losses_epoch)
        val_acc = np.average(accs, weights=batch_sizes)  # Weighted average accuracy

        print(f'Validation accuracy: {val_acc*100:.2f}% | Validation loss: {val_loss:.4f}')

        if save_best and val_acc > best_acc:
            best_acc = val_acc
            best_model_state = self.model.state_dict()

            # Create model filename using hyperparameters
            hyperparam_str = "_".join(f"{key}:{value}" for key, value in hyperparams.items())
            model_filename = f"model_{hyperparam_str}_{time.time()}.pth"

            # Save the best model
            save_path = os.path.join(self.model_path, model_filename)
            torch.save(best_model_state, save_path)
            print(f"Best model saved at: {save_path}")
        return val_acc, val_loss
    

    def evaluate(self, test_loader) -> float:
        """Evaluates the model on the test dataset.

        Args:
            test_loader: DataLoader for the test dataset.
        """
        self.model.to(self.device)
        self.model.eval()
        batch_sizes, accs = [], []
        with torch.no_grad():
            for X_test, y_test, in test_loader:
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)

                prediction = self.model(X_test)
                batch_sizes.append(X_test.shape[0])

                prediction = torch.sigmoid(prediction)
                prediction_classes = (prediction > 0.5).float() # This converts to binary classes 0 and 1

                acc = torch.mean((prediction_classes == y_test).float()).item()
                accs.append(acc)

        # Return average accuracy
        return 0.0 if not accs else np.average(accs, weights=batch_sizes)


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

            prediction = (probability > 0.5).float() # Turn probability into binary classificaiton

        return prediction.item()
        

    def save_model(self, model_state_dict: collections.OrderedDict, model_name: str | None) -> None:
        """Saves the model to the specified file path.

        Args:
            path (str): Path to save the model file.
        """
        path = self.model_path + "/" + model_name
        torch.save(model_state_dict, path)


    def load_model(self, path: str) -> None:
        """Loads a model from the specified file path.

        Args:
            path (str): Path to the model file.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        self.model.eval()

if __name__ == "__main__":


    audioproccessor = AudioProcessor()
    spectroproccessor = SpectrogramProcessor()
    datapipline = DataPipeline(test_size=0.15, val_size=0.15, audio_processor=audioproccessor, spectrogram_processor=spectroproccessor, metadata_df=None, metadata_path="data/cough_data/metadata.csv")
    
    
    cnn_model = CNNModel()
    loss_function = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.SGD(params=cnn_model.parameters(), lr=0.01, momentum=0.9) ###SDG
    optimizer = torch.optim.Adam(params=cnn_model.parameters(), lr=0.01) ### ADAM

    model_handler = ModelHandler(model=cnn_model, model_path="ml/models", optimizer=optimizer, loss_function=loss_function)

    train_loader, val_loader, test_loader = datapipline.create_dataloaders(batch_size=32)

    # Train the model
    epochs = 1

    model_handler.train(train_loader=train_loader, epochs=epochs, model_name="g1_model")

    best_model = None
    best_acc = 0.0

    # Hyperparameters for validation
    hyperparameter_options = [
        {"learning_rate": 0.01},
        {"learning_rate": 0.001},
        {"learning_rate": 0.0001}
    ]

    for hyperparams in hyperparameter_options:
        print(f"Validating model with hyperparameters: {hyperparams}")

        cnn_model = CNNModel()
        # optimizer = torch.optim.SGD(params=cnn_model.parameters(), lr=hyperparams["learning_rate"], momentum=0.9) ###SDG
        optimizer = torch.optim.Adam(params=cnn_model.parameters(), lr=0.01) ### ADAM

        # Create new ModelHandler for each hyperparameter set
        model_handler = ModelHandler(model=cnn_model, model_path="ml/models", optimizer=optimizer, loss_function=loss_function)
        
        # Perform validation
        val_acc, val_loss = model_handler.validate(val_loader, hyperparams)

        # Save the best model based on accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model_handler

        print(f"Validation accuracy: {val_acc*100:.2f}% | Validation loss: {val_loss:.4f}")

    # Final testing with the best model
    if best_model:
        test_acc = best_model.evaluate(test_loader)
        print(f"Test accuracy: {test_acc*100:.2f}%. Awesome!")
