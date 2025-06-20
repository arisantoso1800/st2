import streamlit as st
import pandas as pd
import joblib
import os
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# === UNZIP MODEL JIKA BELUM ADA ===
import requests, zipfile, io, os

# Hanya jalankan jika belum diekstrak
if not os.path.exists('rf2.pkl'):
    FILE_ID = '1mxm8VF5n6VIPFcug_55lMWyjoWnSUm3t'  # ← ganti dengan ID file zip kamu
    download_url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    
    # Unduh dan ekstrak ZIP langsung dari Google Drive
    response = requests.get(download_url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall()

# === LOAD MODEL DAN ENCODER ===
model = joblib.load('rf2.pkl')
le_diag_proc = joblib.load('le_diag_proc.pkl')  # LabelEncoder fitur
le_inacbg = joblib.load('le_inacbg.pkl')        # LabelEncoder target

# === UI ===
st.title("📋 Prediksi INACBG dari Diagnosa dan Prosedur")

diagnosa = st.text_input("🩺 Diagnosa (contoh: A91;K30)")
prosedur = st.text_input("🔧 Prosedur (contoh: 99.18;99.29)")

if st.button("Prediksi"):
    if diagnosa:
        sample_input = diagnosa.strip() + ',' + prosedur.strip()

        if sample_input not in le_diag_proc.classes_:
            st.warning("🚫 Kombinasi diagnosa dan prosedur belum dikenali oleh model.")
        else:
            X_input = le_diag_proc.transform([sample_input]).reshape(-1, 1)
            y_pred = model.predict(X_input)
            y_decoded = le_inacbg.inverse_transform(y_pred)[0]

            code, tarif_str, deskripsi = y_decoded.split(',')
            tarif_rupiah = f"Rp {int(float(tarif_str)):,}".replace(",", ".")

            st.success(f"✅ Kode INACBG: {code}")
            st.write(f"📄 Deskripsi: {deskripsi}")
            st.write(f"💰 Tarif: {tarif_rupiah}")
    else:
        st.warning("Masukkan diagnosa terlebih dahulu.")
