# app/main.py
import sys
import pathlib
import re
from typing import Iterable

# Ensure project root in sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app.preprocess import clean_text, stem_text
from app.sentiment import predict_bert
import visuals.charts as charts  # use generate_wordcloud from here

st.set_page_config(page_title="Analisis Sentimen: Bendera One Piece", layout="centered")
st.title("🎌 Analisis Sentimen: Bendera One Piece")
st.write("Upload CSV berisi kolom `content` (tweet/post).")

uploaded = st.file_uploader("Upload file tweet CSV", type="csv")

def _remove_tokens_from_texts(texts: Iterable[str], tokens_to_remove):
    if not tokens_to_remove:
        return list(texts)
    pattern = r'\b(?:' + '|'.join(re.escape(t.lower()) for t in tokens_to_remove) + r')\b'
    cleaned = []
    for t in texts:
        if not isinstance(t, str):
            cleaned.append("")
            continue
        s = re.sub(pattern, " ", t.lower())
        s = re.sub(r"\s+", " ", s).strip()
        cleaned.append(s)
    return cleaned

# label mapping if model gives English
_en_to_id = {"positive": "positif", "negative": "negatif", "neutral": "netral"}

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error membaca CSV: {e}")
        st.stop()

    if "content" not in df.columns:
        st.error("CSV harus memiliki kolom 'content'.")
        st.stop()

    st.subheader("Preview data")
    st.dataframe(df.head())

    with st.spinner("Membersihkan teks dan menjalankan prediksi..."):
        # preprocessing
        df["clean_text"] = df["content"].apply(clean_text).apply(stem_text)

        # predictions (assumes predict_bert returns a label string)
        df["raw_sentiment"] = df["clean_text"].apply(lambda x: predict_bert(x))

        # normalize labels to Indonesian
        df["sentiment"] = df["raw_sentiment"].astype(str).str.lower().map(lambda x: _en_to_id.get(x, x))

    st.success("Analisis selesai — menyiapkan visualisasi...")

    # ----------------- DISTRIBUTION (inline) -----------------
    try:
        counts = df["sentiment"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax, color=["#00c49a", "#ff6b6b", "#f0ad4e"])
        ax.set_title("Distribusi Sentimen")
        ax.set_xlabel("Sentimen")
        ax.set_ylabel("Jumlah Tweet")
        ax.set_xticklabels([str(x).capitalize() for x in counts.index], rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Gagal membuat plot distribusi: {e}")

    # ----------------- SENTIMENT OVER TIME (if date present) -----------------
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df_time = df.dropna(subset=["date"]).copy()
            if not df_time.empty:
                df_grouped = df_time.groupby([df_time["date"].dt.date, "sentiment"]).size().reset_index(name="count")
                # pivot for plotting
                pivot = df_grouped.pivot(index="date", columns="sentiment", values="count").fillna(0)
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                pivot.plot(ax=ax2, marker="o")
                ax2.set_title("Perubahan Sentimen dari Waktu ke Waktu")
                ax2.set_xlabel("Tanggal")
                ax2.set_ylabel("Jumlah")
                fig2.tight_layout()
                st.pyplot(fig2)
        except Exception as e:
            st.info(f"plot_sentiment_over_time gagal: {e}")

    # ----------------- WORDCLOUDS -----------------
    extra_tokens = ["onepiece", "one", "piece", "bendera", "17agustus", "17", "agustus", "rt", "amp"]
    labels_to_show = ["positif", "negatif", "netral"]

    for label in labels_to_show:
        subset = df[df["sentiment"] == label]
        texts = subset["clean_text"].astype(str).tolist()
        cleaned_texts = _remove_tokens_from_texts(texts, extra_tokens)

        try:
            # charts.generate_wordcloud returns a path string (per your visuals code)
            wc_path = charts.generate_wordcloud(
                cleaned_texts,
                sentiment_label=label,
                mask_path=None,  # optionally set a mask path like "visuals/mask.png"
                max_words=200,
                colormap="tab20",
                background_color="white",
                prefer_horizontal=0.9,
                random_state=42
            )
            if isinstance(wc_path, str) and wc_path:
                st.image(wc_path, caption=f"WordCloud: {label.capitalize()}", use_column_width=True)
            else:
                # fallback - if function returned PIL Image or array
                st.image(wc_path, caption=f"WordCloud: {label.capitalize()}", use_column_width=True)
        except Exception as e:
            st.warning(f"WordCloud {label} gagal dibuat: {e}")

    # ----------------- DOWNLOAD -----------------
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "Download Hasil Analisis",
        data=csv_bytes,
        file_name="hasil_analisis_bendera_onepiece.csv",
        mime="text/csv"
    )

    # show sample
    to_show = df[["content", "clean_text", "sentiment"]].head(20)
    st.subheader("Hasil sample")
    st.dataframe(to_show)
