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
import torch.optim as opt

from .cnn_model import CNNModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.audio_processor import AudioProcessor
from data_processing.spectrogram_processor import SpectrogramProcessor
from data_processing.data_pipeline import DataPipeline


class ModelHandler:
    """Handles the model training, evaluation, and inference pipeline.

    Attributes:
        device (torch.device): The device on which the model is executed (e.g., 'cpu' or 'cuda').
        model_path: Path to where .pth models should be saved.
    """

    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        optimizer: opt.Optimizer,
        loss_function: nn.Module,
        lr_scheduler: opt.lr_scheduler.LRScheduler,
    ):
        """Initializes the ModelHandler.

        Args:
            model (nn.Module): The machine learning model to be trained/evaluated.
            model_path (str | None): Path to the pre-trained model file (if available).
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            loss_function (nn.Module): The loss function used for training the model.
            lr_scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.

        Example Usage:
            model = CNNModel()
            optimizer = opt.Adam(model.parameters(), lr=0.001)
            loss_function = nn.BCEWithLogitsLoss()
            lr_scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            model_handler = ModelHandler(model, model_path, optimizer, loss_function, lr_scheduler)
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_function = loss_function

    def train_step(self, dataloader) -> dict:
        """Used by self.train(). Trains the model for a single epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.

        Returns:
            Dictionary of training information.
                "avg_loss_per_batch": Average loss per batch.
                "avg_acc_per_batch": Average accuracy per batch.
        """
        self.model.train()  # Set model to training mode
        avg_loss, acc = (
            0,
            0,
        )  # We will calculate the average loss and accuracy per batch
        for in_tensor, labels in dataloader:
            in_tensor, labels = in_tensor.to(self.device), labels.to(self.device)
            labels = labels.float().unsqueeze(1)  # Ensure correct shape for BCE loss

            logits = self.model(in_tensor)  # Feed input into model

            loss = self.loss_function(logits, labels)  # Calculate batch loss
            avg_loss += loss.item()  # Add to cumulative loss

            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Calculate batch accuracy and add it to cumulative accuracy
            prediction_classes = torch.round(torch.sigmoid(logits))
            batch_acc = torch.mean((prediction_classes == labels).float()).item()
            acc += batch_acc

        avg_loss /= len(dataloader)  # Calculate avg loss for epoch from cumulative loss
        acc /= len(
            dataloader
        )  # Calculate avg accuracy for epoch from cumulative accuracy
        train_results = {"avg_loss_per_batch": avg_loss, "avg_acc_per_batch": acc * 100}
        return train_results

    def val_step(self, dataloader) -> dict:
        """Used by self.train(). Evaluates the model on the validation dataset.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

        Returns:
            Dictionary of validation information.
                "avg_loss_per_batch": Average loss per batch.
                "avg_acc_per_batch": Average accuracy per batch.
        """

        self.model.eval()
        with torch.inference_mode():
            avg_loss, acc = 0, 0
            for in_tensor, labels in dataloader:
                in_tensor, labels = in_tensor.to(self.device), labels.to(self.device)
                labels = labels.float().unsqueeze(
                    1
                )  # Ensure correct shape for BCE loss

                logits = self.model(in_tensor)  # Feed input into model

                loss = self.loss_function(logits, labels)  # Calculate batch loss
                avg_loss += loss.item()  # Add to cumulative loss

                # Calculate batch accuracy and add it to cumulative accuracy
                prediction_classes = torch.round(torch.sigmoid(logits))
                batch_acc = torch.mean((prediction_classes == labels).float()).item()
                acc += batch_acc

            avg_loss /= len(
                dataloader
            )  # Calculate avg loss for each epoch from cumulative loss
            acc /= len(
                dataloader
            )  # Calculate avg accuracy for each epoch from cumulative accuracy
            valid_results = {
                "avg_loss_per_batch": avg_loss,
                "avg_acc_per_batch": acc * 100,
            }
            return valid_results

    def train(
        self, train_loader, val_loader, epochs: int, model_name: str
    ) -> tuple[dict, dict]:
        """Trains the model.

        Args:
            train_loader: DataLoader for the training datasets
            epochs (int): Number of training epochs.
            model_name (str): Name to save the trained model.

        Returns:
            Two dictionaries containing the following training and validation information:
                "epoch": List of epoch numbers.
                "loss": List of average loss per batch.
                "accuracy": List of average accuracy per
        """
        self.model.to(self.device)
        training_results = {"epoch": [], "loss": [], "accuracy": []}
        validation_results = {"epoch": [], "loss": [], "accuracy": []}

        for epoch in range(epochs):

            # Train the model
            training_data = self.train_step(train_loader)
            training_results["epoch"].append(epoch)
            training_results["loss"].append(training_data["avg_loss_per_batch"])
            training_results["accuracy"].append(training_data["avg_acc_per_batch"])

            # Check the validation loss after training
            validation_data = self.val_step(val_loader)
            validation_results["epoch"].append(epoch)
            validation_results["loss"].append(validation_data["avg_loss_per_batch"])
            validation_results["accuracy"].append(validation_data["avg_acc_per_batch"])

            # Adjust learning rate if necessary
            if self.lr_scheduler:
                # Some LR schedulers take validation loss as input, others will ignore it (I think)
                self.lr_scheduler.step(validation_data["avg_loss_per_batch"])

            print(f"{epoch}:")
            print(f"LR: {self.optimizer.param_groups[0]['lr']}")
            print(
                f"Loss - {training_data['avg_loss_per_batch']:.5f} | Accuracy - {training_data['avg_acc_per_batch']:.2f}%"
            )
            print(
                f"VLoss - {validation_data['avg_loss_per_batch']:.5f} | VAccuracy - {validation_data['avg_acc_per_batch']:.2f}%\n"
            )

        self.save_model(model_state_dict=self.model.state_dict(), model_name=model_name)
        return training_results, validation_results

    def validate(
        self, val_loader, hyperparams: dict, save_best: bool = True
    ) -> tuple[float, float]:
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
                y_prediction_val = torch.sigmoid(
                    y_prediction_val
                )  # Convert logits to probabilities
                prediction_classes = (
                    y_prediction_val > 0.5
                ).float()  # Convert to binary 0/1

                acc = torch.mean((prediction_classes == y_val).float()).item()
                accs.append(acc)
                batch_sizes.append(X_val.shape[0])

        # Compute final validation loss and accuracy
        val_loss = np.mean(val_losses_epoch)
        val_acc = np.average(accs, weights=batch_sizes)  # Weighted average accuracy

        print(
            f"Validation accuracy: {val_acc*100:.2f}% | Validation loss: {val_loss:.4f}"
        )

        if save_best and val_acc > best_acc:
            best_acc = val_acc
            best_model_state = self.model.state_dict()

            # Create model filename using hyperparameters
            hyperparam_str = "_".join(
                f"{key}:{value}" for key, value in hyperparams.items()
            )
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
            for (
                X_test,
                y_test,
            ) in test_loader:
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)

                prediction = self.model(X_test)
                batch_sizes.append(X_test.shape[0])

                prediction = torch.sigmoid(prediction)
                prediction_classes = (
                    prediction > 0.5
                ).float()  # This converts to binary classes 0 and 1

                acc = torch.mean((prediction_classes == y_test).float()).item()
                accs.append(acc)

        # Return average accuracy
        return 0.0 if not accs else np.average(accs, weights=batch_sizes)

    def predict(self, spectrogram: torch.Tensor) -> int:
        """Performs inference on a single spectrogram.

        Args:
            spectrogram (torch.Tensor): Input spectrogram for inference.

        Returns:
            torch.Tensor: The predicted output from the model.
        """

        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device("cpu"))
        )
        self.model.to(self.device)
        self.model.eval()

        spectrogram = spectrogram.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(spectrogram)

            probability = torch.sigmoid(logits)
            print(probability)

            prediction = (
                probability > 0.5
            ).float()  # Turn probability into binary classificaiton
        print("Performed prediction on image.")
        return prediction.item()

    def save_model(
        self, model_state_dict: collections.OrderedDict, model_name: str | None
    ) -> None:
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

    data_pipeline = DataPipeline(
        test_size=0.15,
        val_size=0.15,
        audio_processor=audioproccessor,
        image_processor=spectroproccessor,
    )

    cnn_model = CNNModel()
    loss_function = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.SGD(params=cnn_model.parameters(), lr=0.01, momentum=0.9) ###SDG
    optimizer = torch.optim.Adam(params=cnn_model.parameters(), lr=0.01)  ### ADAM
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model_handler = ModelHandler(
        model=cnn_model,
        model_path="ml/models",
        optimizer=optimizer,
        loss_function=loss_function,
        lr_scheduler=lr_scheduler,
    )

    train_loader, val_loader, test_loader = data_pipeline.create_dataloaders(
        batch_size=32,
        dataset_path="ml/data/cough_data/tensor_dataset",
        spectro_dir_path=None,
        upsample=True,
        aug_spectro_dir_path=None,
        aug_dataset_path=None,
    )

    # Train the model
    epochs = 1

    model_handler.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        model_name="g1_model",
    )

    best_model = None
    best_acc = 0.0

    # Hyperparameters for validation
    hyperparameter_options = [
        {"learning_rate": 0.01},
        {"learning_rate": 0.001},
        {"learning_rate": 0.0001},
    ]

    for hyperparams in hyperparameter_options:
        print(f"Validating model with hyperparameters: {hyperparams}")

        cnn_model = CNNModel()
        # optimizer = torch.optim.SGD(params=cnn_model.parameters(), lr=hyperparams["learning_rate"], momentum=0.9) ###SDG
        optimizer = torch.optim.Adam(params=cnn_model.parameters(), lr=0.01)  ### ADAM

        # Create new ModelHandler for each hyperparameter set
        model_handler = ModelHandler(
            model=cnn_model,
            model_path="ml/models",
            optimizer=optimizer,
            loss_function=loss_function,
            lr_scheduler=lr_scheduler,
        )

        # Perform validation
        val_acc, val_loss = model_handler.validate(val_loader, hyperparams)

        # Save the best model based on accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model_handler

        print(
            f"Validation accuracy: {val_acc*100:.2f}% | Validation loss: {val_loss:.4f}"
        )

    # Final testing with the best model
    if best_model:
        test_acc = best_model.evaluate(test_loader)
        print(f"Test accuracy: {test_acc*100:.2f}%. Awesome!")
