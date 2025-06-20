import pandas as pd
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # untuk load model

df0 = pd.read_pickle('/content/drive/MyDrive/data_eklaim_2022_2024.pkl')
df0 = df0[['DIAGLIST','PROCLIST','INACBG','TARIF_INACBG','DESKRIPSI_INACBG']]

df = df0.sample(n=10000, random_state=42)

# Gabungkan DIAGLIST dan PROCLIST sebagai fitur baru
df['DIAG_PROCLIST'] = df['DIAGLIST'] + ',' + df['PROCLIST']
df['INA_TARIF'] = df['INACBG'] + ',' + df['TARIF_INACBG'].astype(str) + ',' + df['DESKRIPSI_INACBG']

# Encode fitur gabungan dan target
le_diag_proc = LabelEncoder()
le_inacbg = LabelEncoder()  # Sudah ada, tapi kita gunakan ulang

X_full = le_diag_proc.fit_transform(df['DIAG_PROCLIST']).reshape(-1, 1)
y_full = le_inacbg.fit_transform(df['INA_TARIF'])

# Split data
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Train RandomForest
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train_full, y_train_full)

# Save the trained model before loading it
joblib.dump(rf_full, 'rf2.pkl')

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
