import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load the pretrained Wav2Vec 2.0 model
def load_wav2vec_model():
    path = "facebook/wav2vec2-base"
    processor = Wav2Vec2Processor.from_pretrained(path)
    model = Wav2Vec2Model.from_pretrained(path)
    model.eval()
    return processor, model

def extract_wav2vec(waveform, processor, model):
    with torch.no_grad():
        inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.squeeze(0)  # Shape: (T, D)


