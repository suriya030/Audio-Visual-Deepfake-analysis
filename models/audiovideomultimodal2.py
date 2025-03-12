import os
import torch
import random
import torchaudio
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2Model, ViTImageProcessor, ViTModel
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Select device (MPS for Apple Silicon, CPU otherwise)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load models
wav2vec_model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name).eval().to(device)

vit_model_name = "google/vit-base-patch16-224-in21k"
vit_model = ViTModel.from_pretrained(vit_model_name).eval().to(device)
feature_extractor = ViTImageProcessor.from_pretrained(vit_model_name)

# Function to load and preprocess audio
def load_audio(wav_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.to(device), target_sample_rate

# Function to extract video frames
def extract_video_frames(video_path, frame_rate=1):
    #cap = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, round(fps / frame_rate))
    frames = []
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            print("Entered")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_count += 1
        success, frame = cap.read()
    cap.release()
    return frames

# Define Temporal Transformer
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=768, num_heads=8, num_layers=4):
        super(TemporalTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
    
    def forward(self, frame_features):
#        frame_features = frame_features.permute(1, 0, 2)  # (frames, batch, feature_dim)
        print ("frame_features",frame_features.shape)
        encoded_features = self.transformer_encoder(frame_features)
        print ("encoded_features",encoded_features.shape)
        encoded_features1 = encoded_features.mean(dim=0)  # Aggregate over frames
        print ("encoded_features1",encoded_features1.shape)
        return encoded_features1  # Aggregate over frames

# Initialize Transformer
temporal_transformer = TemporalTransformer().to(device)

# Function to process dataset
def process_files(file_list_path, video_root):
    with open(file_list_path, "r") as f:
        lines = [line.strip().split() for line in f.readlines() if line.strip()]
        file_paths = [line[0] for line in lines]
        y_labels = [float(line[1]) for line in lines]
    
    features, labels = [], []
    for i, video_file in enumerate(file_paths):
        video_file = os.path.join(video_root, video_file)
        if not video_file.endswith(".mp4"):
            continue
        wav_file = video_file.replace(".mp4", ".wav")
        if not os.path.exists(wav_file):
            continue
        waveform, _ = load_audio(wav_file)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values.squeeze(0).to(device)
        with torch.no_grad():
            audio_features = wav2vec_model(inputs).last_hidden_state.mean(dim=1)
        frames = extract_video_frames(video_file)
        video_features = []
        for frame in frames:
            inputs = feature_extractor(images=frame, return_tensors="pt").to(device)
            with torch.no_grad():
                frame_features = vit_model(**inputs).last_hidden_state.mean(dim=1)
                print("frame_features",frame_features.shape)
                video_features.append(frame_features)
        
        if video_features: 
            video_features = torch.stack(video_features, dim=0) 
#            print("video_features after cat",video_features.shape) [4,1,768]
            video_features1 = temporal_transformer(video_features)
            print("video_features1 after mean",video_features1.shape) 
        else:
            torch.zeros(1, 768).to(device)

        # Ensure both feature vectors are 2D (batch, feature_dim)
        print("audio_features shape:", audio_features.shape)  # Expected (B, audio_dim)
        print("video_features1 shape:", video_features1.shape)  # Expected (B, video_dim)

            
        multimodal_features = torch.cat((audio_features,video_features1 ), dim=1)
        print("multimodal_features after cat",multimodal_features.shape)
        features.append(multimodal_features.detach().cpu().numpy())

        labels.append(y_labels[i])
    return np.array(features), np.array(labels)

# Load data
video_root = "/Users/mgupta/Documents/Education/Stanford_DL/Final_project/code/test_fairseq_final_model/data/datasets/FakeAVCeleb2/"
file_list_path = "/Users/mgupta/Documents/Education/Stanford_DL/Final_project/code/test_fairseq_final_model/data/datasets/FakeAVCeleb2/test_files_balanced.txt"
X, y = process_files(file_list_path, video_root)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).to(device).view(-1, 1)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(y_val, dtype=torch.float32).to(device).view(-1, 1)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device).view(-1, 1)

# Define model
"""
class FakeVideoDetector(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=512):
        super(FakeVideoDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x).squeeze(-1)
"""    
# Define model with encoder
class FakeVideoDetector(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=512, num_layers=2, nhead=8):
        super(FakeVideoDetector, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        #Expected input shape: (batch_size, feature_dim)
        #Transformer requires: (seq_len, batch, feature_dim) -> Adjust accordingly.


        # Reshape to match Transformer input format (seq_len=1, batch_size, feature_dim)
        
        x = x.squeeze(1)  # Remove extra singleton dimension -> (batch_size, feature_dim)

        # Reshape for Transformer Encoder (seq_len=1, batch_size, feature_dim)
        x = x.unsqueeze(0)  # Shape: (1, batch_size, feature_dim)


        # Pass through Transformer Encoder
        x = self.encoder(x)

        # Remove sequence dimension (back to batch_size, feature_dim)
        x = x.squeeze(0)

        # Pass through classifier
        return self.fc(x).squeeze(-1)
    
classifier = FakeVideoDetector().to(device)

# Training function
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, lr=1e-4):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.view(-1,1), y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.view(-1, 1), y_val)
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Train model
train_model(classifier, X_train, y_train, X_val, y_val)

# Prediction function
def predict(model, X):
    model.eval()
    with torch.no_grad():
        predictions = model(X).cpu().numpy()
    return ["Fake" if pred > 0.5 else "Real" for pred in predictions]

# Test predictions
y_pred = predict(classifier, X_test)
print("Predictions:", y_pred)
print("Actual label:", y_test)

# Convert Fake/Real predictions into binary format for comparison
y_test_label = np.array(['Fake' if y == 0 else 'Real' for y in y_test])
print("Actual label Fake/Real:", y_test_label)

# Convert Fake/Real predictions into binary format for comparison
y_pred_binary = np.array([0 if pred == 'Fake' else 1 for pred in y_pred])
print("y_pred_binary", y_pred_binary)

y_test_numpy = y_test.cpu().numpy().astype(int).reshape(-1)
print("y_test_numpy", y_test_numpy)

# Compute accuracy
accuracy = np.mean(y_pred_binary == y_test_numpy )
print("Prediction Accuracy:", accuracy)