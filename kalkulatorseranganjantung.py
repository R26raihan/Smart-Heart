import math

def calculate_risk(age, total_cholesterol, hdl_cholesterol, systolic_bp,
                   treated_hypertension, smoker, sex):
    # Koefisien untuk pria dan wanita
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
            'constant': -146.5933061
        }
    }
    
    # Memilih koefisien berdasarkan jenis kelamin
    if sex.lower() == 'men':
        coef = coefficients['men']
    elif sex.lower() == 'women':
        coef = coefficients['women']
    else:
        raise ValueError("Jenis kelamin tidak valid. Pilih 'men' atau 'women'.")

    # Menghitung nilai logaritma
    ln_age = math.log(age)
    ln_total_cholesterol = math.log(total_cholesterol)
    ln_hdl_cholesterol = math.log(hdl_cholesterol)
    ln_systolic_bp = math.log(systolic_bp)

    # Menghitung L
    L = (coef['ln_age'] * ln_age + coef['ln_total_cholesterol'] * ln_total_cholesterol +
         coef['ln_hdl_cholesterol'] * ln_hdl_cholesterol + 
         coef['ln_systolic_bp'] * ln_systolic_bp + 
         coef['treated_blood_pressure'] * treated_hypertension + 
         coef['smoker'] * smoker + 
         coef['ln_age_ln_total_cholesterol'] * ln_age * ln_total_cholesterol +
         coef['ln_age_smoker'] * ln_age * smoker +
         (coef['ln_age_ln_age'] * ln_age * ln_age if sex == 'men' else 0) +
         coef['constant'])

    # Menghitung probabilitas
    if sex.lower() == 'men':
        P = 1 - (0.9402 ** math.exp(L))
    else:
        P = 1 - (0.98767 ** math.exp(L))

    return P

# Input dari pengguna
age = int(input("Masukkan usia Anda: "))
total_cholesterol = int(input("Masukkan kadar kolesterol total (mg/dL): "))
hdl_cholesterol = int(input("Masukkan kadar kolesterol HDL (mg/dL): "))
systolic_bp = int(input("Masukkan tekanan darah sistolik (mmHg): "))
treated_hypertension = int(input("Apakah Anda sedang diobati untuk hipertensi? (1 = Ya, 0 = Tidak): "))
smoker = int(input("Apakah Anda merokok? (1 = Ya, 0 = Tidak): "))
sex = input("Masukkan jenis kelamin Anda (men/women): ")

# Menghitung risiko
risk = calculate_risk(age, total_cholesterol, hdl_cholesterol, systolic_bp,
                      treated_hypertension, smoker, sex)

# Menampilkan hasil
print(f"Risiko 10 tahun untuk penyakit jantung koroner (CHD): {risk * 100:.2f}%")
