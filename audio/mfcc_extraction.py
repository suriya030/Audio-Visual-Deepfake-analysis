import torch
import librosa
import librosa.display, librosa.feature
import numpy as np

#resamples to 16kHz to match wav2vec
#returns (1, num_samples)
def load_audio(file_path, target_sr=16000):
    waveform, sr = librosa.load(file_path, sr=target_sr)
    waveform = torch.tensor(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, num_samples)
    return waveform

def extract_mfcc(wav2vec_features, sr=16000, n_mfcc=13):
    mfcc_list = []
    for frame in wav2vec_features.numpy():  # Iterate over time frames
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
        mfcc_list.append(mfcc[:, 0]) #temporal alignment
    return np.array(mfcc_list)  # Shape: (T, n_mfcc)
