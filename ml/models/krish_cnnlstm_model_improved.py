import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMNoBatchNorm(nn.Module):
    def __init__(self):
        super(CNNLSTMNoBatchNorm, self).__init__()

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Calculate CNN output size (assuming 224x224 input)
        self.cnn_output_size = 256 * 7 * 7  # 12544

        # LSTM parameters
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 2

        # Fully connected layer to transform CNN features to LSTM input
        self.fc_before_lstm = nn.Linear(self.cnn_output_size, self.lstm_hidden_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch_size, 256, 7, 7)
        cnn_features = cnn_features.view(batch_size, -1)  # (batch_size, 12544)

        # Prepare sequence for LSTM
        transformed_features = self.fc_before_lstm(cnn_features)  # (batch_size, lstm_hidden_size)
        lstm_input = transformed_features.unsqueeze(1)  # (batch_size, 1, lstm_hidden_size)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(x.device)

        # LSTM
        lstm_output, _ = self.lstm(lstm_input, (h0, c0))  # lstm_output: (batch_size, 1, lstm_hidden_size)

        # Use the last output for classification
        last_output = lstm_output[:, -1, :]  # Take the last timestep output

        # Classification
        out = self.fc(last_output)  # (batch_size, 1)

        return out
