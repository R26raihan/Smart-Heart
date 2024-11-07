from flask import Flask, render_template, request, redirect, url_for, flash, jsonify,session
import math
import random
import torch
import json
import numpy as np
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import firebase_admin
from firebase_admin import credentials, auth
import requests


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Inisialisasi Firebase Admin SDK
cred = credentials.Certificate('D:\TB2IMK_Kelompok11\smart-heart-ec522-firebase-adminsdk-9xckr-b0e0136c62.json')
firebase_admin.initialize_app(cred)

@app.route('/halamanUtama', methods=['GET', 'POST'])
def home():
    # Mengecek apakah user sudah login dengan memeriksa session
    if 'email' in session:
        return render_template('halamanUtama.html', email=session['email'])
    return redirect(url_for('login'))  # Jika belum login, arahkan ke halaman login

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['username']
        password = request.form['password']

        try:
            # URL API Firebase untuk login
            login_url = 'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=AIzaSyCdITy9deqVHZo-X6vqIAJ3gJN_RIqBU8s'
            payload = {
                "email": email,
                "password": password,
                "returnSecureToken": True
            }
            response = requests.post(login_url, json=payload)
            result = response.json()

            # Cek jika login berhasil (status 200)
            if response.status_code == 200:
                if 'email' in result:
                    flash('Login berhasil!', 'success')
                    session['email'] = result['email']  # Menyimpan email di session
                    return redirect(url_for('home'))  # Arahkan ke halaman utama
                else:
                    flash('Login gagal, email atau password salah!', 'error')
            else:
                error_message = result.get("error", {}).get("message", "Username atau password salah!")
                flash(error_message, 'error')

        except Exception as e:
            flash(f'Terjadi kesalahan: {str(e)}', 'error')

    return render_template('login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Password dan konfirmasi password tidak cocok!', 'error')
        else:
            try:
                user = auth.create_user(
                    email=email,
                    password=password,
                    display_name=username
                )
                flash('Pendaftaran berhasil! Silakan login.', 'success')
                return redirect(url_for('login'))
            except firebase_admin.exceptions.FirebaseError as e:
                flash(f'Terjadi kesalahan: {str(e)}', 'error')

    return render_template('register.html')

@app.route('/verify_token', methods=['POST'])
def verify_token():
    token = request.form['idToken']
    try:
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
        return jsonify({'status': 'success', 'user_id': user_id}), 200
    except auth.InvalidIdTokenError:
        return jsonify({'status': 'error', 'message': 'Token tidak valid'}), 401


@app.route('/bmi', methods=['GET', 'POST'])
def bmi():
    if request.method == 'POST':
        berat = float(request.form['berat'])
        tinggi = float(request.form['tinggi']) / 100  # konversi cm ke meter
        bmi_value = berat / (tinggi ** 2)

        # Menentukan kategori BMI
        if bmi_value < 18.5:
            kategori = "BMI Anda: Kurus (Kekurangan Berat Badan)"
        elif 18.5 <= bmi_value < 24.9:
            kategori = "BMI Anda: Normal"
        elif 25 <= bmi_value < 29.9:
            kategori = "BMI Anda: Gemuk (Kelebihan Berat Badan)"
        else:
            kategori = "BMI Anda: Obesitas"

        return render_template('bmi.html', bmi=round(bmi_value, 2), kategori=kategori)
    
    return render_template('bmi.html')

@app.route('/kalori', methods=['GET', 'POST'])
def kalori():
    if request.method == 'POST':
        berat = float(request.form['berat'])
        tinggi = float(request.form['tinggi'])
        usia = int(request.form['usia'])
        gender = request.form['gender']
        aktivitas = request.form['aktivitas']

        # Menghitung BMR menggunakan rumus Harris-Benedict
        if gender == 'pria':
            bmr = 88.362 + (13.397 * berat) + (4.799 * tinggi) - (5.677 * usia)
        else:
            bmr = 447.593 + (9.247 * berat) + (3.098 * tinggi) - (4.330 * usia)

        # Menentukan faktor aktivitas
        if aktivitas == 'sedentary':
            faktor_aktivitas = 1.2
        elif aktivitas == 'light':
            faktor_aktivitas = 1.375
        elif aktivitas == 'moderate':
            faktor_aktivitas = 1.55
        elif aktivitas == 'intense':
            faktor_aktivitas = 1.725

        # Menghitung kebutuhan kalori harian
        kalori_harian = bmr * faktor_aktivitas
        return render_template('kalori.html', kalori=round(kalori_harian, 2))

    return render_template('kalori.html')





@app.route('/kalkulator_Jantung', methods=['GET', 'POST'], endpoint='kalkulator_Jantung')
def risk_calculator():
    risk = None
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            total_cholesterol = int(request.form['total_cholesterol'])
            hdl_cholesterol = int(request.form['hdl_cholesterol'])
            systolic_bp = int(request.form['systolic_bp'])
            treated_hypertension = int(request.form['treated_hypertension'])
            smoker = int(request.form['smoker'])
            sex = request.form['sex'].lower()

            # Validasi input
            if age <= 0 or total_cholesterol <= 0 or hdl_cholesterol <= 0 or systolic_bp <= 0:
                raise ValueError("Semua nilai harus positif.")
            
            risk = calculate_risk(age, total_cholesterol, hdl_cholesterol, systolic_bp,
                                  treated_hypertension, smoker, sex)
        except (ValueError, KeyError) as e:
            risk = "Input tidak valid. Pastikan semua nilai diisi dengan benar."

    return render_template('kalkulator_Jantung.html', risk=risk)

def calculate_risk(age, total_cholesterol, hdl_cholesterol, systolic_bp,
                   treated_hypertension, smoker, sex):
    import math

    coefficients = {
        'men': {
            'ln_age': 52.00961,
            'ln_total_cholesterol': 20.014077,
            'ln_hdl_cholesterol': -0.905964,
            'ln_systolic_bp': 1.305784,
            'treated_blood_pressure': 0.241549,
            'smoker': 12.096316,
            'ln_age_ln_total_cholesterol': -4.605038,
            'ln_age_smoker': -2.84367,
            'ln_age_ln_age': -2.93323,
            'constant': -172.300168
        },
        'women': {
            'ln_age': 31.764001,
            'ln_total_cholesterol': 22.465206,
            'ln_hdl_cholesterol': -1.187731,
            'ln_systolic_bp': 2.552905,
            'treated_blood_pressure': 0.420251,
            'smoker': 13.07543,
            'ln_age_ln_total_cholesterol': -5.060998,
            'ln_age_smoker': -2.996945,
            'ln_age_ln_age': -0.0,  
            'constant': -146.5933061
        }
    }
    
    coef = coefficients[sex.lower()]

    ln_age = math.log(age)
    ln_total_cholesterol = math.log(total_cholesterol)
    ln_hdl_cholesterol = math.log(hdl_cholesterol)
    ln_systolic_bp = math.log(systolic_bp)

    L = (coef['ln_age'] * ln_age + 
         coef['ln_total_cholesterol'] * ln_total_cholesterol +
         coef['ln_hdl_cholesterol'] * ln_hdl_cholesterol + 
         coef['ln_systolic_bp'] * ln_systolic_bp + 
         coef['treated_blood_pressure'] * treated_hypertension + 
         coef['smoker'] * smoker + 
         coef['ln_age_ln_total_cholesterol'] * ln_age * ln_total_cholesterol +
         coef['ln_age_smoker'] * ln_age * smoker +
         (coef['ln_age_ln_age'] * ln_age * ln_age if sex == 'men' else 0) +
         coef['constant'])

    P = 1 - (0.9402 ** math.exp(L)) if sex == 'men' else 1 - (0.98767 ** math.exp(L))

   
    return round(P * 100, 2)


# Memuat intents dan model
with open('D:\\GEMASTIK 2025 MACHINE LEARNING\\FlaskChatbot\\intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "D:\\GEMASTIK 2025 MACHINE LEARNING\\FlaskChatbot\\data.pth"
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



# Memuat intents dan model chatbot
with open('D:\TB2IMK_Kelompok11\intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "D:\TB2IMK_Kelompok11\data.pth"
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


# Route untuk chatbot
@app.route('/ai')
def ai():
    return render_template("AI.html")


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
    

if __name__ == '__main__':
    app.run(debug=True)
