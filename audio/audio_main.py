import torch
import os
from glob import glob
from mfcc_extraction import load_audio, extract_mfcc
from wavfeature_extraction import load_wav2vec_model, extract_wav2vec
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import warnings
from transformers import logging as transformers_logging
from lstm_audio import AudioLSTM

#Suppress warnings from librosa and transformers
warnings.simplefilter("ignore")
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Path to save the dataset: No need to process audio files repeatedly
dataset_path = "processed_dataset.pt"

# Check if the processed dataset exists
if os.path.exists(dataset_path):
    print("Loading preprocessed dataset...")
    saved_data = torch.load(dataset_path)
    padded_features = saved_data["features"]
    labels_tensor = saved_data["labels"]
    print(f"Loaded dataset with shape: Features {padded_features.shape}, Labels {labels_tensor.shape}")

else:
    print("No preprocessed dataset found. Processing audio files...")

    processor, model = load_wav2vec_model()

    audio_dir = "AVLips/wav/"
    categories = {"0_real": 0, "1_fake": 1}  # Assign labels: 0 for real, 1 for fake

    features = []
    labels = []
    for folder, label in categories.items():
        folder_path = os.path.join(audio_dir, folder)

        audio_files = glob(os.path.join(folder_path, "*.wav"))
        print(f"Processing {len(audio_files)} files from {folder}")

        for audio_file in audio_files:
            print(f"Processing: {audio_file}")

            try:
                waveform = load_audio(audio_file)  #shape (1, num_samples)
                wav2vec_features = extract_wav2vec(waveform, processor, model)
                mfcc_temporal = extract_mfcc(wav2vec_features)
                mfcc_tensor = torch.tensor(mfcc_temporal).unsqueeze(0)  # (1, T, n_mfcc) add batch dimension
                features.append(mfcc_tensor)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

# Pad sequences to the longest sequence
    def pad_sequences(feature_list):
        lengths = torch.tensor([f.shape[1] for f in feature_list])
        max_len = max(lengths)  # Find the longest sequence
        padded_features = torch.zeros(len(feature_list), max_len, feature_list[0].shape[2])  # Initialize padded tensor
        for i, f in enumerate(feature_list):
            seq_len = f.shape[1]  # Current sequence length
            padded_features[i, :seq_len, :] = f.squeeze(0)
        return padded_features, lengths
    padded_features, sequence_lengths = pad_sequences(features)
    labels_tensor = torch.tensor(labels)

    print(f"Final Dataset Shape: Features {padded_features.shape}, Labels {labels_tensor.shape}")
    torch.save({"features": padded_features, "labels": labels_tensor}, dataset_path)
    print(f"Dataset saved to {dataset_path}")

# **SPLIT DATASET: 80% Train, 10% Validation, 10% Test**
train_features, temp_features, train_labels, temp_labels = train_test_split(
    padded_features, labels_tensor, test_size=0.2, random_state=42, stratify=labels_tensor
)
val_features, test_features, val_labels, test_labels = train_test_split(
    temp_features, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train Size: {train_features.shape}, Validation Size: {val_features.shape}, Test Size: {test_features.shape}")

# Create PyTorch Datasets
train_dataset = TensorDataset(train_features, train_labels)
val_dataset = TensorDataset(val_features, val_labels)
test_dataset = TensorDataset(test_features, test_labels)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)