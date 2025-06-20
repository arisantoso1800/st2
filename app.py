import streamlit as st
import pandas as pd
import joblib
import os
import zipfile
import io
import requests
from sklearn.preprocessing import LabelEncoder

# === KONFIGURASI ===
FILE_ID = '1AbCdEfGhIjKlMnOpQrS'  # Ganti dengan ID file Google Drive rf2.zip kamu
ZIP_FILE = 'rf2.zip'
ZIP_URL = f'https://drive.google.com/uc?export=download&id={FILE_ID}'

# === DOWNLOAD ZIP JIKA BELUM ADA ===
if not os.path.exists('rf2.pkl'):
    st.info("📥 Mengunduh model dari Google Drive...")
    try:
        response = requests.get(ZIP_URL)
        content_type = response.headers.get('Content-Type', '')
        if 'zip' not in content_type and not response.content.startswith(b'PK'):
            st.error("❌ Respons dari Google Drive bukan file ZIP. Periksa apakah file kamu sudah di-share secara publik.")
            st.stop()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall()
        st.success("✅ Model berhasil diunduh dan diekstrak.")
    except Exception as e:
        st.error(f"❌ Gagal mengunduh atau mengekstrak model: {e}")
        st.stop()

# === LOAD MODEL DAN ENCODER ===
try:
    rf_model = joblib.load('rf2.pkl')
    le_diag_proc = joblib.load('le_diag_proc.pkl')
    le_inacbg = joblib.load('le_inacbg.pkl')
except Exception as e:
    st.error(f"Gagal memuat model atau encoder: {e}")
    st.stop()

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
