# model_initializer.py
from transformers import SeamlessM4Tv2Model, AutoProcessor
import torch
import torchaudio
#print(torch.__version__) version 2.3.0



def initialize_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("running on mps")
    elif torch.backends.cuda.is_available():
        device = torch.device("cuda")
        print("running on cuda")
    else:
        device = torch.device("cpu")
        print("running on cpu")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    model = model.to(device)
    return model, processor, device

model, processor, device = initialize_model()

#from text
text_inputs = processor(text = "Hello, my name is andrew and I am graduating from college", src_lang="eng", return_tensors="pt")
audio_array_from_text = model.generate(**text_inputs, tgt_lang="rus")[0].cpu().numpy().squeeze()