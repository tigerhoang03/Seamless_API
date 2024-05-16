# from flask import Flask, request, jsonify , render_template

# app = Flask(__name__)

# @app.root("/")
# def home():
#     return "Home"

# if __name__ == "__main__":
#     app.run(debug=True)
from init import initialize_model
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
from io import BytesIO
import soundfile as sf
import numpy as np

import torchaudio

model, processor, device = initialize_model()
app = Flask(__name__)       
CORS(app)


@app.route('/s2s', methods=['POST'])
def s2s():
    # Check if the part 'file' is present in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty part without a filename.
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Load and resample audio using torchaudio
    audio, orig_freq = torchaudio.load(BytesIO(file.read()))
    if orig_freq != 16000:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)

    # Process the audio file
    tgt_lang = request.form.get('tgtLang', '')  # Assuming 'tgtLang' is sent as a form field
    audio_inputs = processor(audios=audio, sampling_rate=16000, return_tensors="pt").to(device)
    audio_array_from_audio = model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()

    return jsonify({'audioData': audio_array_from_audio.tolist(), 'sample_rate': 16000})

if __name__ == '__main__':
    app.run(debug=True)
    