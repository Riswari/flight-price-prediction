
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# 1. Setup Pemetaan Durasi untuk Kombinasi Kota Asal, Kota Tujuan, dan Jumlah Transit
duration_map = {
    ('Bangalore', 'Chennai', 'one'): 15.2, 
    ('Bangalore', 'Chennai', 'two or more'): 19.6, 
    ('Bangalore', 'Chennai', 'zero'): 1.0, 
    ('Bangalore', 'Delhi', 'one'): 11.1,
    ('Bangalore', 'Delhi', 'two or more'): 10.0, 
    ('Bangalore', 'Delhi', 'zero'): 2.8, 
    ('Bangalore', 'Hyderabad', 'one'): 15.2, 
    ('Bangalore', 'Hyderabad', 'two or more'): 23.0, 
    ('Bangalore', 'Hyderabad', 'zero'): 1.2, 
    ('Bangalore', 'Kolkata', 'one'): 13.7, 
    ('Bangalore', 'Kolkata', 'two or more'): 15.1, 
    ('Bangalore', 'Kolkata', 'zero'): 2.5, 
    ('Bangalore', 'Mumbai', 'one'): 12.2, 
    ('Bangalore', 'Mumbai', 'two or more'): 15.2, 
    ('Bangalore', 'Mumbai', 'zero'): 1.8, 
    ('Chennai', 'Bangalore', 'one'): 15.0, 
    ('Chennai', 'Bangalore', 'two or more'): 14.1, 
    ('Chennai', 'Bangalore', 'zero'): 1.1, 
    ('Chennai', 'Delhi', 'one'): 12.3, 
    ('Chennai', 'Delhi', 'two or more'): 15.0, 
    ('Chennai', 'Delhi', 'zero'): 2.8, 
    ('Chennai', 'Hyderabad', 'one'): 14.1, 
    ('Chennai', 'Hyderabad', 'two or more'): 16.5, 
    ('Chennai', 'Hyderabad', 'zero'): 1.2, 
    ('Chennai', 'Kolkata', 'one'): 15.3, 
    ('Chennai', 'Kolkata', 'two or more'): 17.6, 
    ('Chennai', 'Kolkata', 'zero'): 2.4, 
    ('Chennai', 'Mumbai', 'one'): 13.7, 
    ('Chennai', 'Mumbai', 'two or more'): 16.2, 
    ('Chennai', 'Mumbai', 'zero'): 2.0, 
    ('Delhi', 'Bangalore', 'one'): 11.8, 
    ('Delhi', 'Bangalore', 'two or more'): 11.9, 
    ('Delhi', 'Bangalore', 'zero'): 2.8, 
    ('Delhi', 'Chennai', 'one'): 13.3, 
    ('Delhi', 'Chennai', 'two or more'): 16.7, 
    ('Delhi', 'Chennai', 'zero'): 2.8, 
    ('Delhi', 'Hyderabad', 'one'): 14.4, 
    ('Delhi', 'Hyderabad', 'two or more'): 13.8, 
    ('Delhi', 'Hyderabad', 'zero'): 2.2, 
    ('Delhi', 'Kolkata', 'one'): 14.3,
    ('Delhi', 'Kolkata', 'two or more'): 14.8, 
    ('Delhi', 'Kolkata', 'zero'): 2.2, 
    ('Delhi', 'Mumbai', 'one'): 12.9, 
    ('Delhi', 'Mumbai', 'two or more'): 10.1,
    ('Delhi', 'Mumbai', 'zero'): 2.2,
    ('Hyderabad', 'Bangalore', 'one'): 13.4,
    ('Hyderabad', 'Bangalore', 'two or more'): 11.5, 
    ('Hyderabad', 'Bangalore', 'zero'): 1.2, 
    ('Hyderabad', 'Chennai', 'one'): 14.4, 
    ('Hyderabad', 'Chennai', 'two or more'): 14.7, 
    ('Hyderabad', 'Chennai', 'zero'): 1.3, 
    ('Hyderabad', 'Delhi', 'one'): 12.5, 
    ('Hyderabad', 'Delhi', 'two or more'): 9.8, 
    ('Hyderabad', 'Delhi', 'zero'): 2.3, 
    ('Hyderabad', 'Kolkata', 'one'): 14.3, 
    ('Hyderabad', 'Kolkata', 'two or more'): 14.6, 
    ('Hyderabad', 'Kolkata', 'zero'): 2.0, 
    ('Hyderabad', 'Mumbai', 'one'): 12.6, 
    ('Hyderabad', 'Mumbai', 'two or more'): 18.6, 
    ('Hyderabad', 'Mumbai', 'zero'): 1.5, 
    ('Kolkata', 'Bangalore', 'one'): 15.0, 
    ('Kolkata', 'Bangalore', 'two or more'): 11.9, 
    ('Kolkata', 'Bangalore', 'zero'): 2.7, 
    ('Kolkata', 'Chennai', 'one'): 15.3, 
    ('Kolkata', 'Chennai', 'two or more'): 19.8, 
    ('Kolkata', 'Chennai', 'zero'): 2.4, 
    ('Kolkata', 'Delhi', 'one'): 13.3, 
    ('Kolkata', 'Delhi', 'two or more'): 12.1, 
    ('Kolkata', 'Delhi', 'zero'): 2.5, 
    ('Kolkata', 'Hyderabad', 'one'): 14.4, 
    ('Kolkata', 'Hyderabad', 'two or more'): 16.6,
    ('Kolkata', 'Hyderabad', 'zero'): 2.2, 
    ('Kolkata', 'Mumbai', 'one'): 13.7, 
    ('Kolkata', 'Mumbai', 'two or more'): 16.6, 
    ('Kolkata', 'Mumbai', 'zero'): 2.8, 
    ('Mumbai', 'Bangalore', 'one'): 12.8, 
    ('Mumbai', 'Bangalore', 'two or more'): 19.7, 
    ('Mumbai', 'Bangalore', 'zero'): 1.7, 
    ('Mumbai', 'Chennai', 'one'): 14.1, 
    ('Mumbai', 'Chennai', 'two or more'): 12.9, 
    ('Mumbai', 'Chennai', 'zero'): 2.0, 
    ('Mumbai', 'Delhi', 'one'): 12.3, 
    ('Mumbai', 'Delhi', 'two or more'): 9.0, 
    ('Mumbai', 'Delhi', 'zero'): 2.2, 
    ('Mumbai', 'Hyderabad', 'one'): 14.1, 
    ('Mumbai', 'Hyderabad', 'two or more'): 16.4, 
    ('Mumbai', 'Hyderabad', 'zero'): 1.4, 
    ('Mumbai', 'Kolkata', 'one'): 13.8, 
    ('Mumbai', 'Kolkata', 'two or more'): 11.6, 
    ('Mumbai', 'Kolkata', 'zero'): 2.6
}

# 2. Memuat Model
@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="Riswari/flight-price-prediction",
        filename="flight_price_prediction.joblib"
    )
    return joblib.load(path)

model = load_model()

# 3. Merancang Antarmuka Pengguna (UI) Reaktif
st.set_page_config(page_title="Prediksi Harga Tiket Pesawat", layout="centered")
st.title("Sistem Prediksi Harga Tiket Pesawat")
st.markdown("Masukkan parameter penerbangan di bawah ini. Mesin akan mengeksekusi arsitektur Random Forest untuk memprediksi harga tiket secara akurat.")

# Menghapus st.form agar UI menjadi dinamis/reaktif
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Maskapai")
        airline = st.selectbox("Airline", ['Vistara', 'Air India', 'Indigo', 'Go First', 'Airasia', 'Spicejet'])
        
        # LOGIKA BISNIS: Filter Kelas Penerbangan berdasarkan Maskapai
        if airline in ['Vistara', 'Air India']:
            class_options = ['Economy', 'Business']
        else:
            class_options = ['Economy']
            
        flight_class = st.selectbox("Class", class_options)
        source_city = st.selectbox("Departure City", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        destination_city = st.selectbox("Destination City", ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        
    with col2:
        st.subheader("Waktu & Transit")
        departure_time = st.selectbox("Departure Time", ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'])
        arrival_time = st.selectbox("Arrival Time", ['Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late Night'])
        stops = st.selectbox("Number of Stops", ['zero', 'one', 'two or more'])
        
    st.subheader("Durasi")
    duration = duration_map.get((source_city, destination_city, stops), 3.0)
    st.info(f"✈️ Estimasi durasi penerbangan: {duration} jam")

    st.subheader("Jangka Waktu Menuju Keberangkatan")
    days_left = st.number_input("Days Left (for Departure)", min_value=1, max_value=50, value=15, step=1)
    
    # Mengganti form_submit_button menjadi button standar dengan gaya utama
    st.markdown("---")
    submit = st.button("Calculate Price Prediction", type="primary", use_container_width=True)

# 4. Logika Pemrosesan saat Tombol Ditekan
if submit:
    if source_city == destination_city:
        st.error("Logika tidak valid: Kota Asal dan Kota Tujuan tidak boleh sama.")
    else:
        # Menyusun data input menjadi format matriks
        input_data = pd.DataFrame({
            'airline': [airline],
            'source_city': [source_city],
            'departure_time': [departure_time],
            'stops': [stops],
            'arrival_time': [arrival_time],
            'destination_city': [destination_city],
            'class': [flight_class],
            'duration': [duration],
            'days_left': [days_left]
        })
        
        try:
            # Prediksi
            predicted_price = model.predict(input_data)
            
            # Menampilkan Hasil
            st.success(f"### Estimasi Harga Tiket: **INR {predicted_price[0]:,.2f}**")
            st.info("Prediksi dihitung menggunakan algoritma Random Forest dengan RMSLE Score 0.14")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan komputasi internal: {e}")

