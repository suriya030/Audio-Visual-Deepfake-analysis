# CS229 Project: Fake or Real? Audio-Visual Deepfake Detection

## Feature Extraction

**Audio Features:**  
Run `CS229_project/data_preprocessing_audio/audio_processing.py` followed by `CS229_project/models/baseline_audio.py`.

**SWIN Features:**  
Execute `CS229_project/data_preprocessing_video/extract_swin_features.ipynb`.  
Download the Video-Swin Transformer weights from [Kaggle](https://www.kaggle.com/models/kaggle/video-swin-transformer) or use the extracted features from [Google Drive](https://drive.google.com/drive/folders/1gOPUzGyHsepW1n7Q7mPDdxJ5hqfcBLG1?usp=sharing).

**WAV2VEC2 / MFCC Features:**  
Run `CS229_project/audio/audio_main.py`.

## Training and Inference

**Video Baseline:**  
Run `CS229_project/models/video_baseline.ipynb` to train and run inference (ensure SWIN features are extracted beforehand).

**Audio-Visual Baseline 1:**  
Run `CS229_project/models/AVbaseline.ipynb` to train and run inference (ensure both SWIN features and WAV2VEC2/MFCC features are extracted beforehand).

Additional processing includes extracting Wav2Vec embeddings, MFCC features, and splitting the dataset (80% training, 10% validation, 10% test) using `CS229_project/audio/audio_main.py`, as well as processing padded MFCC sequences with the LSTM model by running `CS229_project/audio/audio_train.py`.
