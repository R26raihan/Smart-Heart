from flask import Flask, request, jsonify, send_file
import io
from gtts import gTTS
import random
import torch
import json
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)

# Memuat intents dan model
with open('D:\\GEMASTIK 2025 MACHINE LEARNING\\FlaskChatbot\\intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "D:\GEMASTIK 2025 MACHINE LEARNING\FlaskChatbot\data.pth"
data = torch.load(FILE, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
last_response = ""
audio_cache = {}  # Caching audio

@app.route('/')
def index():
    return send_file("D:\GEMASTIK 2025 MACHINE LEARNING\FlaskChatbot\Templates\index.html")

@app.route('/get_response', methods=['POST'])
def get_response():
    global last_response
    data = request.json
    sentence = tokenize(data['message'])
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                last_response = random.choice(intent['responses'])
                return jsonify({'response': last_response})
    else:
        last_response = "Saya tidak mengerti..."
        return jsonify({'response': last_response})

@app.route('/get_audio')
def get_audio():
    global last_response
    if last_response in audio_cache:
        # Mengembalikan audio dari cache
        return send_file(audio_cache[last_response], mimetype='audio/mpeg')

    tts = gTTS(text=last_response, lang='id')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)

    # Simpan ke cache
    audio_cache[last_response] = audio_fp

    return send_file(audio_fp, mimetype='audio/mpeg')

if __name__ == "__main__":
    app.run(debug=True)
