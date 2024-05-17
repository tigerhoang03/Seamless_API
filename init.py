# model_initializer.py
#note install with:
#pip install git+https://github.com/huggingface/transformers.git sentencepiece

from transformers import SeamlessM4Tv2Model, AutoProcessor
import torch
import scipy
import torchaudio 
#import sounddevice as sd
#print(torch.__version__) version 2.3.0

def initialize_model():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("running on mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("running on cuda")
    else:
        device = torch.device("cpu")
        print("running on cpu")
    device = "cpu" #temp fix for cuda issue
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    model = model.to(torch.device(device)) 
    return model, processor, device




def text_to_speech(text: str, src_lang:str, tgt_lang=str):
    #from text
    model, processor, device = initialize_model()
    text_inputs = processor(text = text, src_lang=src_lang, return_tensors="pt").to(device)
    audio_array_from_text = model.generate(**text_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
    scipy.io.wavfile.write("T2S_output.wav", rate=16000, data=audio_array_from_text)
    return audio_array_from_text


def speech_to_speech(file_path: str, tgt_lang:str):
    #from audio
    #note audio is a tensor of the .wav, and also returns the inital sample rate
    model, processor, device = initialize_model()
    audio, orig_freq = torchaudio.load(file_path)
    audio =  torchaudio.functional.resample(audio, orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
    audio_inputs = processor(audios=audio , sampling_rate=16000,return_tensors="pt").to(device)
    audio_array_from_audio = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
    scipy.io.wavfile.write("S2S_output.wav", rate=16000, data=audio_array_from_audio)
    return audio_array_from_audio

#example usage
# text_to_speech("Hello, my name is andrew and I am graduating from college", "eng", "spa")
# speech_to_speech("MTgwMTU1MDUyMTgwMTYx_fLo0NGj3U1o.wav", "eng")

