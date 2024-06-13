import streamlit as st
import pandas as pd
import pickle

# Memuat model, scaler, dan label encoder dari file
def load_model():
    with open('diagnosis_penyakit.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return model, scaler, label_encoder

# Memuat model, scaler, dan label encoder
model, scaler, label_encoder = load_model()

# Fungsi untuk memprediksi penyakit berdasarkan gejala
def predict_disease(symptoms):
    # Membuat dataframe dari gejala
    input_data = pd.DataFrame([symptoms])
    
    # Standarisasi input data
    input_data_scaled = scaler.transform(input_data)
    
    # Memprediksi penyakit
    prediction = model.predict(input_data_scaled)
    return label_encoder.inverse_transform(prediction)[0]

# Judul aplikasi
st.title('Aplikasi Diagnosa Penyakit Berbasis Gejala')

# Input gejala
fever = st.selectbox('Demam (fever):', [0, 1])
cough = st.selectbox('Batuk (cough):', [0, 1])
sore_throat = st.selectbox('Sakit Tenggorokan (sore throat):', [0, 1])
shortness_of_breath = st.selectbox('Sesak Napas (shortness of breath):', [0, 1])
headache = st.selectbox('Sakit Kepala (headache):', [0, 1])

# Tombol untuk prediksi
if st.button('Diagnosa'):
    symptoms = {
        'fever': fever,
        'cough': cough,
        'sore_throat': sore_throat,
        'shortness_of_breath': shortness_of_breath,
        'headache': headache,
    }
    diagnosis = predict_disease(symptoms)
    st.write(f'Hasil Diagnosa: {diagnosis}')

# Menjalankan aplikasi dengan streamlit run app.py
