# flask_app.py
from flask import Flask, request, jsonify
from init import text_to_speech, speech_to_speech

app = Flask(__name__)

@app.route('/t2t', methods=['POST'])
def t2t():
    data = request.get_json()
    text = data.get('text')
    src_lang = data.get('src_lang')
    tgt_lang = data.get('tgt_lang')
    if not all([text, src_lang, tgt_lang]):
        return jsonify({'error': 'Missing parameters'}), 400
    result = text_to_speech(text, src_lang, tgt_lang)
    return jsonify({'result': result.tolist()})

@app.route('/s2s', methods=['POST'])
def s2s():
    data = request.get_json()
    file_path = data.get('file_path')
    tgt_lang = data.get('tgt_lang')
    if not all([file_path, tgt_lang]):
        return jsonify({'error': 'Missing parameters'}), 400
    result = speech_to_speech(file_path, tgt_lang)
    return jsonify({'result': result.tolist()})

if __name__ == '__main__':
    app.run(debug=True)