import cv2
import numpy as np
import skvideo
skvideo.setFFmpegPath('/opt/miniconda3/envs/avhubert_test/bin')
import skvideo.io
from tqdm import tqdm
import os
import os.path as osp
import sys
from base64 import b64encode
import tempfile
from argparse import Namespace
import utils as avhubert_utils
from av_hubert.fairseq.fairseq import checkpoint_utils, options, tasks
import av_hubert.fairseq.fairseq.utils as fairseq_utils
from av_hubert.fairseq.fairseq.dataclass.configs import GenerationConfig
from glob import glob
from scipy.io import wavfile
import shutil
#from av_hubert.avhubert import utils as avhubert_utils
import soundfile as sf
import json
import torch.nn.functional as F
from sklearn import metrics
import argparse
import torch
#from fairseq.models.avhubert import AVHubertModel
#from av_hubert.fairseq.fairseq.models.hubert import hubert, hubert_pretraining
import sys
sys.path.append('/Users/mgupta/Documents/Education/Stanford_DL/Final_project/code/SpeechForensics_autoencoder/av_hubert')
from av_hubert.fairseq.fairseq.tasks import TASK_REGISTRY, TASK_DATACLASS_REGISTRY
from argparse import Namespace
from av_hubert.fairseq.fairseq import utils
from python_speech_features import logfbank
from torch.utils.data import DataLoader, Dataset
import random
from torch.nn.utils.rnn import pad_sequence


# Specify the path to the custom user directory
utils.import_user_module(Namespace(user_dir="/Users/mgupta/Documents/Education/Stanford_DL/Final_project/code/SpeechForensics_autoencoder/av_hubert/avhubert"))



# export PYTORCH_ENABLE_MPS_FALLBACK=1 Type this to run the script
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim

# Lists to collect all features
audio_features_list = []
video_features_list_real = []
video_features_list_fake = []

def trim_to_min_length(tensor_a, tensor_b):
    min_length = min(tensor_a.shape[0], tensor_b.shape[0])
    return tensor_a[:min_length], tensor_b[:min_length]

def contrastive_loss(anchor, positive, negative, temperature=0.5):
    # Ensure cosine similarity is computed across the last dimension (feature dimension)
    pos_sim = F.cosine_similarity(anchor, positive, dim=-1).mean(dim=1).unsqueeze(1)  # [batch_size, 1]
    neg_sim = F.cosine_similarity(anchor, negative, dim=-1).mean(dim=1).unsqueeze(1)  # [batch_size, 1]

    # Concatenate positive and negative similarities into logits
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # Shape: [batch_size, 2]

    # Apply temperature scaling
    logits = logits / temperature

    # Labels: 0 for positive samples
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)

    # Debugging prints
    print(f"Logits shape: {logits.shape}")  # Should be [16, 2]
    print(f"Labels shape: {labels.shape}")  # Should be [16]

    # Cross-entropy loss expects [batch_size, num_classes]
    return F.cross_entropy(logits, labels)




# Dummy Dataset Class for Audio and Video
class AudioVideoDataset(Dataset):
    def __init__(self, audio_features, video_features):
        self.audio = audio_features
        self.video = video_features

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return self.audio[idx], self.video[idx]

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, projection_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, nhead=4, num_layers=2):
        super(TransformerEncoder, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, 
            nhead=nhead, 
            dim_feedforward=256,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.input_proj(x)  # Shape: (batch_size, seq_length, latent_dim)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, latent_dim)
        output = self.transformer_encoder(x)
        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_length, latent_dim)
        return output    
   
    
# Function to downsample video frames
import torch.nn.functional as F
def preprocess_video(video, size=(32, 32)):
    # video shape: [1, 1, 1, 251, 88, 88]
    batch_size, channels,frames, height, width = video.shape

    # Reshape to process frames individually
    video = video.view(-1, channels, height, width)  # Shape: [251, 1, 88, 88]

    # Downsample frames to 32x32
    video = F.interpolate(video, size=size, mode='bilinear', align_corners=False)  # Shape: [251, 1, 32, 32]

    # Flatten and reshape for encoder
    video = video.view(frames, -1)  # Shape: [251, 1024]

    return video

def preprocess_audio(audio):
    # audio shape: [1, 1, 104, 251]
    return audio.squeeze(0).squeeze(0).transpose(0, 1)  # Shape: [251, 104]



def extract_visual_feature(video_path,max_length):
    
    # Detect the device (MPS for Mac, else CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Manually specify crop size, mean, and std
    image_crop_size = 88  # Example size, change as per your requirements
    image_mean = 0.421 
    image_std = 0.165 
    
    from torchvision import transforms

    transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x),
    transforms.Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
    transforms.CenterCrop((image_crop_size, image_crop_size)),  # Center crop
    transforms.Normalize(mean=image_mean, std=image_std)  # Standard normalization  
    ])

    video = cv2.VideoCapture(video_path)
    fps=video.get(cv2.CAP_PROP_FPS)
    
    #Added by Manisha
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Convert frame to grayscale to ensure shape (h, w)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    #####
    
    video.release()
    if len(frames)>fps*max_length:
        frames=frames[:int(fps*max_length)]
        
    #  Convert frames to a NumPy array Added by Manisha
    frames = np.array(frames, dtype=np.float32)    
    #####

    print("Frames Shape Before Transform:", frames.shape)
    
    frames = transform(frames)
    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
    return frames

def stacker(feats, stack_order):
    """
    Concatenating consecutive audio frames
    Args:
    feats - numpy.ndarray of shape [T, F]
    stack_order - int (number of neighboring frames to concatenate
    Returns:
    feats - numpy.ndarray of shape [T', F']
    """
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis = 0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
    return feats

def extract_audio_feature(audio_path):
    # Detect device (MPS for Mac, otherwise CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
        
    sample_rate, wav_data = wavfile.read(audio_path)
    assert sample_rate == 16_000 and len(wav_data.shape) == 1
    audio_feats = logfbank(wav_data, samplerate = sample_rate).astype(np.float32)  # [T, F]
    audio_feats = stacker(audio_feats, 4)
    with torch.no_grad():
        audio_feats=torch.FloatTensor(audio_feats).to(device)
    
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    audio_feats=audio_feats.transpose(0,1).unsqueeze(dim=0)
    return audio_feats

tmp_dir = tempfile.mkdtemp()
def collect_features(mouth_roi_path,wav_path,max_length=50, realorfake=0):
    #trim audio
    wav,sr=sf.read(wav_path)
    if len(wav)>sr*max_length:
        wav_path=osp.join(tmp_dir,'audio.wav')
        if osp.exists(wav_path):
            os.remove(wav_path)
        sf.write(wav_path,wav[:sr*max_length],sr)

     # Convert similarity scores to binary predictions

    real = 0 if realorfake == 0 else 1 
    video_features=extract_visual_feature(mouth_roi_path,max_length) #(1, 1, 250, 88, 88)
 
    if real == 0:
        video_features_list_real.append(video_features) #real
        audio_features=extract_audio_feature(wav_path) #(1, 104, 256)
        audio_features_list.append(audio_features)
    else:
        video_features_list_fake.append(video_features)

        
def pretrain_audio_visual_feature():
    batch_size = 8
    epochs = 10
    total_samples = len(audio_features_list) 
       
    # Training Loop
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle the data at the beginning of each epoch
        indices = list(range(total_samples))
        random.shuffle(indices) 
        
        for i in range(0, total_samples, batch_size):
            optimizer.zero_grad()
            
            # Select batch indices
            batch_indices = indices[i:i + batch_size]

            # Batch processing
            #anchor_audio = torch.stack([preprocess_audio(audio_features_list[idx]) for idx in batch_indices])
            # Pad audio features in batch
            anchor_audio = pad_sequence(
             [preprocess_audio(audio_features_list[idx]) for idx in batch_indices],
            batch_first=True
            )
            #positive_video = torch.stack([preprocess_video(video_features_list[idx]) for idx in batch_indices])            
            # Pad video features in the batch
            positive_video = pad_sequence(
             [preprocess_video(video_features_list_real[idx]) for idx in batch_indices],
                batch_first=True
            )
            
            # Pad video features in the batch
            negative_video = pad_sequence(
             [preprocess_video(video_features_list_fake[idx]) for idx in batch_indices],
                batch_first=True
            )            

            # Trim to minimum length for consistency
            min_length = min(positive_video.shape[1], negative_video.shape[1])
            positive_video = positive_video[:, :min_length]
            negative_video = negative_video[:, :min_length]
            
            # Compute embeddings
            anchor_emb = audio_encoder(anchor_audio)
            positive_emb = video_encoder(positive_video)
            negative_emb = video_encoder(negative_video)
            

            # Apply projection head
            anchor_emb = common_proj(anchor_emb)
            positive_emb = common_proj(positive_emb)
            negative_emb = common_proj(negative_emb)            

            # Trim embeddings to ensure equal length
            min_emb_length = min(anchor_emb.shape[1], positive_emb.shape[1], negative_emb.shape[1])
            anchor_emb = anchor_emb[:, :min_emb_length]
            positive_emb = positive_emb[:, :min_emb_length]
            negative_emb = negative_emb[:, :min_emb_length]           

            # Calculate loss
            #loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            loss = contrastive_loss(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    torch.save({
         'audio_encoder': audio_encoder.state_dict(),
         'video_encoder': video_encoder.state_dict(),
         'common_proj': common_proj.state_dict(),
        'optimizer': optimizer.state_dict()
        
        }, 'pretrained_contrastive_model.pth')
  

 


def pretrain_args(args):
    video_root=args.video_root
    file_list1=args.file_list1
    file_list2=args.file_list2
    cropped_mouth_dir=args.mouth_dir
    max_length=args.max_length

    with open(file_list1,'r') as f:
        video_list1=f.read().split('\n')

    for video_item in tqdm(video_list1): #real
        print("video_item",video_item)
        video_path=osp.join(video_root,video_item.split(' ')[0])
        video_label=video_item.split(' ')[1]
        mouth_roi_path=video_path.replace(video_root,cropped_mouth_dir)
        wav_path=mouth_roi_path.replace('.mp4','.wav')
        if not  ((osp.exists(mouth_roi_path) and osp.exists(wav_path))):
            continue
        #Extract visual and audio speech representations respectively and compute their cosine similarity
        collect_features(mouth_roi_path, wav_path, args.max_length,0) # 0 for real
        
    with open(file_list2,'r') as f:
        video_list2=f.read().split('\n')

    for video_item in tqdm(video_list2): #fake
        print("video_item",video_item)
        video_path=osp.join(video_root,video_item.split(' ')[0])
        video_label=video_item.split(' ')[1]
        mouth_roi_path=video_path.replace(video_root,cropped_mouth_dir)
        wav_path=mouth_roi_path.replace('.mp4','.wav')
        if not  ((osp.exists(mouth_roi_path) and osp.exists(wav_path))):
            continue
        #Extract visual and audio speech representations respectively and compute their cosine similarity
        collect_features(mouth_roi_path, wav_path, args.max_length,1) #1 for fake
    
        
    pretrain_audio_visual_feature()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extracting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path',type=str,default='checkpoints/large_vox_iter5.pt',help='checkpoint path')
    parser.add_argument('--video_root', type=str,required=True,help='video root dir')
    parser.add_argument('--file_list1',type=str,required=True,help='file list')
    parser.add_argument('--file_list',type=str,required=True,help='file list')
    parser.add_argument('--file_list2',type=str,required=True,help='file list')
    parser.add_argument('--mouth_dir',type=str,required=True,help='cropped mouth dir')
    parser.add_argument('--max_length',type=int, default=50, help='maximum video length consumed by model')
    parser.add_argument('--ffmpeg', type=str, default='/opt/miniconda3/envs/avhubert_test/bin/ffmpeg',
                        help='ffmpeg path')
    args = parser.parse_args()

    ckpt_path = args.checkpoint_path
    user_dir = os.getcwd()
    fairseq_utils.import_user_module(Namespace(user_dir=user_dir))

    # Check for MPS device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
        
     
    # Initialize Encoders
    audio_encoder = TransformerEncoder(input_dim=104, latent_dim=512).to(device)
    video_encoder = TransformerEncoder(input_dim=1024, latent_dim=512).to(device)   

    common_proj = ProjectionHead(512).to(device)
  
    
    # Loss and Optimizer
    #triplet_loss = nn.TripletMarginLoss(margin=2.0)
    optimizer = optim.Adam(list(audio_encoder.parameters()) + list(video_encoder.parameters()), lr=1e-3)
 
    
    pretrain_args(args)
