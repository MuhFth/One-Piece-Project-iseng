# app/main.py
import streamlit as st
import pandas as pd
from app.preprocess import clean_text, stem_text
from app.sentiment import predict_bert  # or predict_lexicon
from visuals.charts import plot_sentiment_distribution, generate_wordcloud

st.title("🎌 Analisis Sentimen: Bendera One Piece")

uploaded = st.file_uploader("Upload file tweet CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview data", df.head())

    st.write("Membersihkan dan menganalisis...")
    df["clean_text"] = df["content"].apply(clean_text).apply(stem_text)
    df["sentiment"] = df["clean_text"].apply(predict_bert)  # or predict_lexicon
    
    

    st.write("Hasil Analisis", df[["content", "sentiment"]])
    df.to_csv("data/hasil_sentimen.csv", index=False)

    st.success("Analisis selesai. Menampilkan visualisasi...")

    plot_sentiment_distribution(df)
    st.image("visuals/sentiment_dist.png")

    for label in ["positif", "negatif", "netral"]:
        generate_wordcloud(df[df["sentiment"] == label]["clean_text"], label)
        st.image(f"visuals/wordcloud_{label}.png", caption=f"WordCloud: {label}")
    

    st.download_button("Download Hasil Analisis", data=df.to_csv(index=False), file_name="hasil_analisis.csv")
