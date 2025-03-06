# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#from fairseq import checkpoint_utils
import cv2
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch.nn as nn

# Load Wav2Vec 2.0 model and processor from Hugging Face
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).eval()

# Ensure execution on CPU (Mac does not support CUDA natively)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Load and preprocess audio
def load_audio(wav_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(wav_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform.to(device), target_sample_rate

# Extract audio features
wav_file = "/Users/mgupta/Documents/Education/Stanford_DL/Final_project/code/test_fairseq/RealVideo-RealAudio2/African/men/id00076/00109.wav"
waveform, sample_rate = load_audio(wav_file)
print("waveform shape",waveform.shape)

# Process input for Wav2Vec 2.0
#inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(device)
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True).input_values.squeeze(0).to(device)

print("inputs shape",inputs.shape)
with torch.no_grad():
    audio_features = model(inputs).last_hidden_state  # Shape: [batch, time_steps, 1024]

# Pooling to get a single vector representation
audio_features = audio_features.mean(dim=1)  # Shape: [1, 1024]
print("Extracted Audio Features Shape:", audio_features.shape)


# Check if MPS is available for Mac; otherwise, use CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load a pretrained Vision Transformer model
model_name = "google/vit-base-patch16-224-in21k"
vit_model = ViTModel.from_pretrained(model_name).eval().to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)


def extract_video_frames(video_path, frame_rate=1):
    """Extract 1 frame per second from a video."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps / frame_rate)  # Extract 1 frame per second

    frames = []
    frame_count = 0
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_count += 1
        success, frame = cap.read()

    cap.release()
    return frames

# Extract frames from the video
video_file = "/Users/mgupta/Documents/Education/Stanford_DL/Final_project/code/test_fairseq/RealVideo-RealAudio2/African/men/id00076/00109.mp4"
frames = extract_video_frames(video_file)

# Extract features for each frame
video_features = []
for frame in frames:
    inputs = feature_extractor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        frame_features = vit_model(**inputs).last_hidden_state.mean(dim=1)  # Shape: [1, 768]
        video_features.append(frame_features)

# Stack into a single tensor
video_features = torch.cat(video_features, dim=0)  # Shape: [11, 768]
print("Extracted Video Features Shape:", video_features.shape)

#Concantenation option 1
# Expand audio features to match video frames
#audio_features_expanded = audio_features.repeat(video_features.shape[0], 1)  # Shape: [11, 1024]

# Concatenate along feature dimension
#multimodal_features = torch.cat((audio_features_expanded, video_features), dim=1)  # Shape: [11, 1024+768]
#print("Final Multimodal Feature Shape:", multimodal_features.shape)

#Concatenation option 2
# Pool video features
video_features_pooled = video_features.mean(dim=0, keepdim=True)  # Shape: [1, 768]
# Concatenate
multimodal_features = torch.cat((audio_features, video_features_pooled), dim=1)  # Shape: [1, 1792]
print("Final Multimodal Feature Shape1 (Pooled Video):", multimodal_features.shape)

class FakeVideoDetector(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=512):
        super(FakeVideoDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability of being "fake"
        )

    def forward(self, x):
        return self.fc(x)

# Initialize classifier
classifier = FakeVideoDetector().to(device)

# Example input tensor (batch_size=1, feature_dim=1792)
# example_input = multimodal_features.to(device)

"""
# Get prediction
with torch.no_grad():
    output = classifier(example_input)

# Interpret prediction
fake_probability = output.item()
threshold = 0.5
prediction = "Fake Video" if fake_probability > threshold else "Real Video"

print(f"Fake Probability: {fake_probability:.4f} → Prediction: {prediction}")
"""
epochs = 1

#Collect labeled data: X = multimodal_features, y = 0 (real), 1 (fake)
#Train the classifier using PyTorch’s nn.BCELoss() (binary cross-entropy loss).
#Fine-tune on a dataset like FaceForensics++.

X_train = multimodal_features
y_train = torch.tensor([[0.0]], dtype=torch.float32).to(device)  # Shape: [1, 1] (Binary label)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)


# Training loop
for epoch in range(epochs):
    classifier.train()
    optimizer.zero_grad()
    outputs = classifier(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

