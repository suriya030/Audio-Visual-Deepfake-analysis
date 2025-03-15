# CS229 Project: Fake or Real? Audio-Visual Deepfake Detection

For details, Here is the [REPORT](https://drive.google.com/file/d/1Xa_185_c-Pz8MhV0K1k-iFc4f6RXQjCz/view?usp=drive_link)

## Feature Extraction

**SWIN Features:**  
Execute `CS229_project/data_preprocessing_video/extract_swin_features.ipynb`.  
Download the Video-Swin Transformer weights from [Kaggle](https://www.kaggle.com/models/kaggle/video-swin-transformer) or use the extracted features from [Google Drive](https://drive.google.com/drive/folders/1gOPUzGyHsepW1n7Q7mPDdxJ5hqfcBLG1?usp=sharing).

**WAV2VEC2 / MFCC Features:**  
Run `CS229_project/audio/audio_main.py` to extract Wav2Vec embeddings and MFCC features or use the extracted features from [Google Drive](https://drive.google.com/drive/folders/1gOPUzGyHsepW1n7Q7mPDdxJ5hqfcBLG1?usp=sharing).


## Training and Inference

**Audio Baseline:**  
Run `CS229_project/data_preprocessing_audio/audio_processing.py` to extract MFCC, Zero Crossing Rate, Chroma features followed by `CS229_project/models/baseline_audio.py` to train and run inference.

**Video Baseline:**  
Run `CS229_project/models/baselinemodel_video.ipynb` to train and run inference (ensure SWIN features are extracted beforehand).

**Multi-modal Fusion Model:**  
Run `CS229_project/models/AVbaseline.ipynb` to train and run inference (ensure both SWIN features and WAV2VEC2/MFCC features are extracted beforehand).

Additional processing includes extracting Wav2Vec embeddings, MFCC features, and splitting the dataset (80% training, 10% validation, 10% test) using `CS229_project/audio/audio_main.py`, as well as processing padded MFCC sequences with the LSTM model to process audio sequence and output prediction by running `CS229_project/audio/audio_train.py`.

**Multi-modal Contrastive Learning Model**  
Root directory for the project is 'CS229/Multimodal_Contrastive_learning'.
To pretrain the model, run pretrain.py.
To evaluate the model, run evaluate.py.

Reference:Modification folder from SpeechForensics project
