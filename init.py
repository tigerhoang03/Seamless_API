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
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    #model = model.to(device)
    model = model.to(torch.device("cpu"))
    return model, processor, device

model, processor, device = initialize_model()


def text_to_text(text: str, src_lang:str, tgt_lang=str):
    #from text
    text_inputs = processor(text = "Hello, my name is andrew and I am graduating from college", src_lang="eng", return_tensors="pt")
    audio_array_from_text = model.generate(**text_inputs, tgt_lang="hin")[0].cpu().numpy().squeeze()
    scipy.io.wavfile.write("out_from_text.wav", rate=16000, data=audio_array_from_text)
    return audio_array_from_text

#file_path: str, tgt_lang:str
def speech_to_speech():
    #from audio
    #note audio is a tensor of the .wav, and also returns the inital sample rate
    audio, orig_freq = torchaudio.load("MTgwMTU1MDUyMTgwMTYx_fLo0NGj3U1o.wav")
    audio =  torchaudio.functional.resample(audio, orig_freq, new_freq=16_000) # must be a 16 kHz waveform array
    audio_inputs = processor(audios=audio , sampling_rate=16000,return_tensors="pt")
    audio_array_from_audio = model.generate(**audio_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
    scipy.io.wavfile.write("two_out_from_text.wav", 16000, data=audio_array_from_audio)
    return audio_array_from_audio

speech_to_speech()

# sample_rate = model.config.sampling_rate
# print(sample_rate)
# waveform, init_sample_rate = torchaudio.load("https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav")
# resample_audio = torchaudio.functional.resample(waveform, init_sample_rate,16000)
# audio_input = processor(resample_audio, return_tensors="pt")
# audio_array_from_audio = model.generate(**audio_input, tgt_lang="hin")[0].cpu().numpy().squeeze()