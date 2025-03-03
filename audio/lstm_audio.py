import torch
import torch.nn as nn

class AudioLSTM(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=2, output_dim=2):
        super(AudioLSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last time step's output
        output = self.fc(last_hidden)
        return self.softmax(output)