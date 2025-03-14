# CS229 Project : Fake or Real? Audio-Visual Deepfake Detection

Baseline models

- Audio features
  - To run the baseline models for audio features, run CS229_project/data_preprocessing_audio
    /audio_processing.py, followed by CS229_project/models/baseline_audio.py

- Video features
  - To run the baseline models for video features, run CS229_project/data_preprocessing_video
    /extract_swin_features.ipynb

Multi-Modal Fusion models
- Extract Wav2Vec embeddings, MFCC features and split dataset using 80% training, 10% validation and 10% test using CS229_project/audio
/audio_main.py
- Process padded MFCC sequences using LSTM model CS229_project/audio/audio_train.py

To dowload directly extracted Audio features, Video features and its corresponding labels :
https://drive.google.com/drive/folders/1gOPUzGyHsepW1n7Q7mPDdxJ5hqfcBLG1?usp=sharing
