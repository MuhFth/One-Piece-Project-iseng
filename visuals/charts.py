import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd
def plot_bar_distribution(df):
    fig, ax = plt.subplots()
    sns.countplot(x="sentiment", data=df, palette="Set2", ax=ax)
    ax.set_title("Distribusi Sentimen (Bar Chart)")
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah Tweet")
    return fig

def plot_pie_chart(df):
    data = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90, colors=["#00c49a", "#ff6b6b", "#fdd835"])
    ax.set_title("Proporsi Sentimen (Pie Chart)")
    return fig

def generate_wordcloud(texts, sentiment_label):
    text = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wc.to_image()

def plot_sentiment_over_time(df):
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df_grouped = df.groupby([df["date"].dt.date, "sentiment"]).size().reset_index(name='count')
    fig = px.line(df_grouped, x="date", y="count", color="sentiment", markers=True,
                title="Perubahan Sentimen dari Waktu ke Waktu")
    return fig