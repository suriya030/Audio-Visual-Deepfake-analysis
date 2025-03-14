import torch
import torch.nn as nn
import torch.optim as optim
from lstm_audio import AudioLSTM
from audio_main import train_loader, val_loader, test_loader  # Import data

learning_rate = 0.001
num_epochs = 10
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize LSTM model
#model = AudioLSTM(input_dim=13, hidden_dim=64, num_layers=2, output_dim=2)
model = AudioLSTM(input_dim=13, hidden_dim=128, num_layers=3, output_dim=2, bidirectional=True) #testing bidirectional LSTM

model.to(device)
entr = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0
    correct_train = 0
    total_train = 0

    for mfccs, labels in train_loader:
        mfccs, labels = mfccs.to(device), labels.to(device)
        outputs = model(mfccs)
        loss = entr(outputs, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    # Validation Loop
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for mfccs, labels in val_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            loss = entr(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val

    print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

torch.save(model.state_dict(), "deepfake_lstm.pth")
print("Model saved successfully!")

# Testing
model.eval()
test_loss = 0
correct_test = 0
total_test = 0

with torch.no_grad():
    for mfccs, labels in test_loader:
        mfccs, labels = mfccs.to(device), labels.to(device)
        outputs = model(mfccs)
        loss = entr(outputs, labels)
        test_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct_test += (predictions == labels).sum().item()
        total_test += labels.size(0)

avg_test_loss = test_loss / len(test_loader)
test_accuracy = correct_test / total_test

print(f"Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
