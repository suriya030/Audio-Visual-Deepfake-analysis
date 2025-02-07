import librosa
import os
import numpy as np
import librosa.display, librosa.feature
from numpy import ndarray
import pandas as pd

def extract_mfcc(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc: ndarray = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract 13 MFCCs
    return np.mean(mfcc, axis=1) # Mean values for each MFCC

#shape, mfcc_features = extract_mfcc("AVLips/wav/1_fake/3109.wav")
#print("MFCC Features:", mfcc_features)
#print("MFCC Shape:", shape)

def extract_zcr(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.mean(zcr)

#zcr_feature = extract_zcr("AVLips/wav/0_real/0.wav")
#print("Zero Crossing Rate:", zcr_feature)

def extract_chroma(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)

#chroma_feature = extract_chroma("AVLips/wav/0_real/0.wav")
#print("Chroma Features:", chroma_feature)

def extract_tempo(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return tempo

#tempo_feature = extract_tempo("AVLips/wav/0_real/0.wav")
#print("Tempo Feature:", tempo_feature)

def process_audio(audio_folder, output_csv):
    data = []
    for folder in os.listdir(audio_folder):  # Loops over 0_real & 1_fake
        folder_path = os.path.join(audio_folder, folder)
        for audio_file in os.listdir(folder_path):
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(folder_path, audio_file)

                # Extract features
                mfcc = extract_mfcc(audio_path)
                zcr = extract_zcr(audio_path)
                chroma = extract_chroma(audio_path)
                tempo = extract_tempo(audio_path)

                data.append([audio_file, folder] + list(mfcc) + [zcr] + list(chroma) + [tempo])
                print(f"Processed {audio_file} in {folder}")

    columns = ["filename", "label"] + [f"mfcc_{i}" for i in range(13)] + ["zcr"] + [f"chroma_{i}" for i in
                                                                             range(12)] + ["tempo"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved audio features to {output_csv}")



