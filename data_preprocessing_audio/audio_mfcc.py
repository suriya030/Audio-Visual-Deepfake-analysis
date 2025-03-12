import librosa
import librosa.feature
import os
import json

#Extracts Audio features (MFCCs) from wav folder
def extract_mfcc_json(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Shape:(13, time_steps)

    return mfcc.T # (time_steps, 13)

def process_audio_json(audio_folder, output_json):
    data = []

    for folder in os.listdir(audio_folder): #loops pver 0_real and 1_fake
        folder_path = os.path.join(audio_folder, folder)
        for audio_file in os.listdir(folder_path):
            if audio_file.endswith(".wav"):
                audio_path = os.path.join(folder_path, audio_file)
                mfcc = extract_mfcc_json(audio_path).tolist()  # Convert to list for JSON serialization

                data.append({"filename": audio_file, "label": folder, "mfcc": mfcc})
                print(f"Processed {audio_file} in {folder}")

    # Save as JSON
    with open(output_json, "w") as f:
        json.dump(data, f)
    print(f"Saved audio features to {output_json}")

process_audio_json("AVLips/wav/", "audio_mfcc.json")