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
from sklearn.metrics import accuracy_score, roc_auc_score
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
import torch.nn as nn
from torchvision import transforms


# Specify the path to the custom user directory
utils.import_user_module(Namespace(user_dir="/Users/mgupta/Documents/Education/Stanford_DL/Final_project/code/SpeechForensics_autoencoder/av_hubert/avhubert"))

# Set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

def trim_to_min_length(tensor_a, tensor_b):
    min_length = min(tensor_a.shape[0], tensor_b.shape[0])
    return tensor_a[:min_length], tensor_b[:min_length]

def preprocess_audio(audio):
    # audio shape: [1, 1, 104, 251]
    return audio.squeeze(0).squeeze(0).transpose(0, 1)  # Shape: [251, 104]

# Define Encoder Architecture
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)
  
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
#        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, latent_dim)
        output = self.transformer_encoder(x)
#        output = output.permute(1, 0, 2)  # Convert back to (batch_size, seq_length, latent_dim)
        return output   

# export PYTORCH_ENABLE_MPS_FALLBACK=1 Type this to run the script
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def calc_cos_dist(feat1,feat2,vshift=15):
    feat1=torch.nn.functional.normalize(feat1,p=2,dim=1)
    feat2 = torch.nn.functional.normalize(feat2, p = 2, dim = 1)
    if len(feat1)!=len(feat2):
        sample=np.linspace(0,len(feat1)-1,len(feat2),dtype = int)
        feat1=feat1[sample.tolist()]
    win_size = vshift*2+1
    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))
    dists = []
    for i in range(0,len(feat1)):
        dists.append(torch.nn.functional.cosine_similarity(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]).cpu().numpy())
    dists=np.asarray(dists)
    return dists

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
    audio_feats=torch.FloatTensor(audio_feats).to(device)
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    audio_feats=audio_feats.transpose(0,1).unsqueeze(dim=0)
    
    return audio_feats

tmp_dir = tempfile.mkdtemp()
def evaluate_audio_visual_feature(mouth_roi_path,wav_path,max_length=50):
    #trim audio
    wav,sr=sf.read(wav_path)
    if len(wav)>sr*max_length:
        wav_path=osp.join(tmp_dir,'audio.wav')
        if osp.exists(wav_path):
            os.remove(wav_path)
        sf.write(wav_path,wav[:sr*max_length],sr)

    video_features=extract_visual_feature(mouth_roi_path,max_length)
    audio_features=extract_audio_feature(wav_path)
    
     # Preprocess the inputs
    audio_features = preprocess_audio(audio_features)
    video_features = preprocess_video(video_features)

    # Compute embeddings
    with torch.no_grad():
        audio_embedding = audio_encoder(audio_features)
        video_embedding = video_encoder(video_features)
        
        # Apply projection head
        audio_embedding = common_proj(audio_embedding)
        video_embedding = common_proj(video_embedding)


    # Trim embeddings to match lengths
    audio_embedding, video_embedding = trim_to_min_length(audio_embedding, video_embedding)
#    print("audio embedding",audio_embedding)
#    print("video embedding",video_embedding)
#    print("diff",sum(audio_embedding - video_embedding))
    dist=calc_cos_dist(video_embedding.cpu(),audio_embedding.cpu()) #cosine
    dist=dist.mean(axis = 0)
#    print("cosine values frame mean",dist)
    dist=dist.mean(axis = 0)
    print("cosine values max",dist)

    return float(dist)

def evaluate_auc(args):
    video_root=args.video_root
    file_list=args.file_list
    cropped_mouth_dir=args.mouth_dir
    max_length=args.max_length

    with open(file_list,'r') as f:
        video_list=f.read().split('\n')

    outputs=[]
    labels=[]
    for video_item in tqdm(video_list):
        print("video_item",video_item)
        video_path=osp.join(video_root,video_item.split(' ')[0])
        video_label=video_item.split(' ')[1]
        mouth_roi_path=video_path.replace(video_root,cropped_mouth_dir)
        wav_path=mouth_roi_path.replace('.mp4','.wav')
        if not  ((osp.exists(mouth_roi_path) and osp.exists(wav_path))):
            continue
        #Extract visual and audio speech representations respectively and compute their cosine similarity
        sim=evaluate_audio_visual_feature(mouth_roi_path,wav_path,max_length)
        print("sim",sim)
        print("video label",video_label)
        outputs.append(sim)
        labels.append(int(video_label))

    outputs=np.asarray(outputs)
    labels=np.asarray(labels)
    fpr,tpr,_ = metrics.roc_curve(labels,outputs)
    auc=metrics.auc(fpr, tpr)
    print(len(outputs))
    print('AUC:{}'.format(auc))
    
    # Convert similarity scores to binary predictions
    predictions = [0 if output >= 0.5 or output <= -0.5 else 1 for output in outputs]
    print("prediction",predictions)

    # Calculate Accuracy
    accuracy = accuracy_score(labels, predictions)
    
    print('accuracy:{}'.format(accuracy*100))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extracting facial landmarks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_path',type=str,default='checkpoints/large_vox_iter5.pt',help='checkpoint path')
    parser.add_argument('--video_root', type=str,required=True,help='video root dir')
    parser.add_argument('--file_list',type=str,required=True,help='file list')
    parser.add_argument('--file_list1',type=str,required=True,help='file list')
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

    # Path to the checkpoint
    #ckpt_path = "/Users/mgupta/Documents/Education/Stanford_DL/Final_project/code/SpeechForensics/checkpoints/large_vox_iter5.pt"
    
    print("Registered Tasks:", TASK_REGISTRY.keys())
    print("Registered Dataclasses:", TASK_DATACLASS_REGISTRY.keys())
    
    # Load the complete checkpoint
    checkpoint = torch.load('pretrained_contrastive_model.pth', map_location=device)
    
    # Initialize Encoders
    audio_encoder = TransformerEncoder(input_dim=104, latent_dim=512).to(device)
    video_encoder = TransformerEncoder(input_dim=1024, latent_dim=512).to(device)  
    
    common_proj = ProjectionHead(512).to(device)
    
        

    # Load the weights into the models
    audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    video_encoder.load_state_dict(checkpoint['video_encoder'])
    common_proj.load_state_dict(checkpoint['common_proj'])


    # If needed, load optimizer (for fine-tuning or further training)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # Set to evaluation mode
    audio_encoder.eval()
    video_encoder.eval()
    common_proj.eval()
    
    
    evaluate_auc(args)
