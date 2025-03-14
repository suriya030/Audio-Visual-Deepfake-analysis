# CS229 Project : Fake or Real? Audio-Visual Deepfake Detection

Feature Extraction 
- Audio features
  - To run the baseline models for audio features, run CS229_project/data_preprocessing_audio
    /audio_processing.py, followed by CS229_project/models/baseline_audio.py

- SWIN FEATURES
  - run CS229_project/data_preprocessing_video/extract_swin_features.ipynb. Video-Swin Transformer
    weights can be downloaded from [here](https://www.kaggle.com/models/kaggle/video-swin-transformer) or download the extracted features from [here](https://drive.google.com/drive/folders/1gOPUzGyHsepW1n7Q7mPDdxJ5hqfcBLG1?usp=sharing)

- WAV2VEC2 FEATURES/MFCC FEATURES
  -  CS229_project/audio/audio_main.py.
  
For Training and Inference,
- For Video baseline : run CS229_project/models/video_baseline.ipynb to train and run inference.
  SWIN FEATURES need to be extracted before for running this file.
- For Audio-Visual 1 : run CS229_project/models/AVbaseline.ipynb to train and run inference.
  SWIN FEATURES and WAV2VEC2 FEATURES/MFCC FEATURES need to be extracted before for running this file.
- Extract Wav2Vec embeddings, MFCC features and split dataset using 80% training, 10% validation and 10% test using CS229_project/audio
/audio_main.py
- Process padded MFCC sequences using LSTM model CS229_project/audio/audio_train.py


