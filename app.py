import zipfile
import os

# Unzip rf2.zip jika file .pkl belum diekstrak
if not os.path.exists('rf2.pkl'):
    with zipfile.ZipFile('rf2.zip', 'r') as zip_ref:
        zip_ref.extractall()

import joblib
rf_full = joblib.load('rf2.pkl')

st.title("Prediksi Kode INACBG")

# Form input
diagnosa = st.text_input("Diagnosa (contoh: A91;K30)")
procedur = st.text_input("Prosedur (contoh: 99.18;99.29;90.59)")

if st.button("Prediksi"):
    if diagnosa:

        # Gabungkan fitur kelas rawat (numerik) + diagnosis (vektor teks)
        # input_data = pd.DataFrame([[kelas,diagnosa]], columns=['KELAS_RAWAT','DIAGLIST'])
        # sample_input = 'N18.5;J81;J80'
        sample_input = diagnosa + ',' + procedur
        # sample_input_encoded = le_diag.transform([diagnosa]).reshape(-1, 1)

        # predicted_inacbg = le_inacbg.inverse_transform(rf_full.predict(sample_input_encoded))
        sample_input_encoded = le_diag_proc.transform([sample_input]).reshape(-1, 1)
        predicted_inacbg = le_inacbg.inverse_transform(rf_full.predict(sample_input_encoded))

        predicted_inacbg_string = predicted_inacbg[0]
        predicted_inacbg_parts = predicted_inacbg_string.split(',')
        predicted_inacbg_code = predicted_inacbg_parts[0]
        predicted_tarif_string = predicted_inacbg_parts[1]
        predicted_deskripsi = predicted_inacbg_parts[2]

        predicted_tarif_float = float(predicted_tarif_string)

        tarif = f"Rp {int(predicted_tarif_float):,}".replace(",", ".")

        # Prediksi
        # pred = rf_full.predict(input_data)[0]
        st.success(f"💡 Kode INACBG: {predicted_inacbg_code}")
        st.write(f"💡 deskripsi INACBG : {predicted_deskripsi}")
        st.write(f"💡 tarif INACBG : {tarif}")
        # st.success(f"💡 Kode INACBG: {diagnosa}")
    else:
        st.warning("Masukkan teks diagnosis terlebih dahulu.")
