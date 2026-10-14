import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediccion precio gasolina", page_icon=" ")
art = joblib.load("modelo_gasolina.pkl")
modelo, scaler, feats = art["modelo"], art["scaler"], art["features"]

st.title("Predicción del precio de la gasolina")
st.write("Ingresa los valores de las variables y obten la prediccion en USD/galon.")

valores = {}
for f in feats:
    if f == "Eventos_Geopoliticos":
        valores[f] = float(st.selectbox("Eventos geopoliticos (0 = no, 1 = si)", [0, 1]))
    else:
        valores[f] = st.number_input(
            f.replace("_", " "),
            min_value=float(art["minimos"][f]) * 0.5,
            max_value=float(art["maximos"][f]) * 1.5,
            value=float(art["medias"][f]),
        )

if st.button("Predecir"):
    x = scaler.transform(pd.DataFrame([valores])[feats])
    pred = modelo.predict(x)[0]
    lo, hi = pred - art["t"] * art["s"], pred + art["t"] * art["s"]
    st.success(f"Precio estimado: **{pred:.2f} USD/galon**")
    st.info(f"Intervalo de confianza 95%: [{lo:.2f} ; {hi:.2f}] USD/galon")

with st.expander("Metricas del modelo"):
    st.write(art["metricas"])