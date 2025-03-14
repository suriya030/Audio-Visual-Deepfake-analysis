# CS229 Project: Fake or Real? Audio-Visual Deepfake Detection

## Feature Extraction

### SWIN Features
- **Extraction:**
  - Execute the notebook:
    - `CS229_project/data_preprocessing_video/extract_swin_features.ipynb`
- **Weights:**
  - Download Video-Swin Transformer weights from:
    - [Kaggle](https://www.kaggle.com/models/kaggle/video-swin-transformer)
- Alternatively, download pre-extracted features from:
  - [Google Drive](https://drive.google.com/drive/folders/1gOPUzGyHsepW1n7Q7mPDdxJ5hqfcBLG1?usp=sharing)

### WAV2VEC2 / MFCC Features
- Run the feature extraction script:
  - `CS229_project/audio/audio_main.py`

## Training and Inference

### Video Baseline
- **Run:**
  - `CS229_project/models/video_baseline.ipynb`
- **Note:** Ensure that SWIN features have been extracted beforehand.

### Audio-Visual Baseline 1
- **Run:**
  - `CS229_project/models/AVbaseline.ipynb`
- **Note:** Both SWIN features and WAV2VEC2/MFCC features must be extracted prior to running this notebook.




- Extract Wav2Vec embeddings, MFCC features, and split the dataset (80% training, 10% validation, 10% test) by running:
    - `CS229_project/audio/audio_main.py`
  - Process padded MFCC sequences using an LSTM model by running:
    - `CS229_project/audio/audio_train.py`

### Audio Features
  1. Run the audio preprocessing script:
     - `CS229_project/data_preprocessing_audio/audio_processing.py`
  2. Run the audio baseline model:
     - `CS229_project/models/baseline_audio.py`

