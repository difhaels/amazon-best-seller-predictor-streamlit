import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load Model 4MB Lu
model = joblib.load('models/amazon_bestseller_predictor.pkl')

st.set_page_config(page_title="The Best Seller Predictor", page_icon="🔮")
st.title("Amazon Best Seller Predictor")
st.write("Prediksi status Best Seller produk Amazon secara instan dengan akurasi model Random Forest 97%")

# 2. Input UI (Samakan dengan input Gradio lu sebelumnya)
col1, col2 = st.columns(2)

with col1:
    rating = st.slider("Rating Produk", 0.0, 5.0, 4.5)
    reviews = st.number_input("Jumlah Review", min_value=0, value=500)
    bought_last_month = st.number_input("Terjual Bulan Lalu", min_value=0, value=1000)
    is_sponsored = st.radio("Apakah Sponsored/Iklan?", ["Ya", "Tidak"])

with col2:
    current_price = st.number_input("Harga Sekarang ($)", min_value=0.0, value=25.0)
    original_price = st.number_input("Harga Asli/Sebelum Diskon ($)", min_value=0.0, value=30.0)
    buy_box = st.radio("Buy Box Availability", ["Tersedia", "Tidak"])

# 3. Logika Pemrosesan Fitur (Harus 8 Fitur!)
if st.button("Ramal Sekarang!"):
    # Hitung fitur tambahan: discount_percentage
    discount_pct = 0
    if original_price > 0:
        discount_pct = ((original_price - current_price) / original_price) * 100

    # Susun DataFrame dengan 8 kolom sesuai urutan training model lu
    # Pastikan nama kolom sama persis dengan X.columns saat training
    feature_names = [
        'rating', 'number_of_reviews', 'bought_in_last_month', 'current_price',
        'original_price', 'discount_percentage', 'is_sponsored', 'buy_box_availability'
    ]
    
    input_df = pd.DataFrame([[
        rating, 
        reviews, 
        bought_last_month, 
        current_price,
        original_price, 
        discount_pct,
        1 if is_sponsored == "Ya" else 0,
        1 if buy_box == "Tersedia" else 0
    ]], columns=feature_names)

    # 4. Prediksi
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"🔥 **HASIL: BEST SELLER!**")
        st.write(f"Tingkat keyakinan model: **{probability:.2%}**")
    else:
        st.error(f"☁️ **HASIL: Bukan Best Seller**")
        st.write(f"Probabilitas menjadi Best Seller cuma: **{probability:.2%}**")