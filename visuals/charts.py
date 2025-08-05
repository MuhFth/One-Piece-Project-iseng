# add these imports at top of visuals/charts.py if not present
import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords as nltk_stopwords
import re
from collections import Counter

# ensure Indonesian stopwords loaded once
try:
    IND_STOPWORDS = set(nltk_stopwords.words("indonesian"))
except Exception:
    # fallback: small custom list if NLTK indonesian missing
    IND_STOPWORDS = {"dan","di","ke","yang","ini","itu","dengan","ga","yg","yg","rt","amp"}

# Helper: default circular mask if no custom mask provided
def _make_circle_mask(size=800, blur=10):
    # size: pixel width/height of square image
    x = np.arange(0, size)
    y = np.arange(0, size)[:, None]
    cx, cy = size // 2, size // 2
    r = size // 2 - 10
    circle = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
    mask = 255 * circle.astype(np.uint8)
    # convert to 3-channel (wordcloud expects 2D or 3D)
    return mask

def _tokenize_and_filter(texts, extra_stopwords=None, min_len=3):
    """
    texts: iterable of strings
    returns: Counter of tokens
    """
    # join and basic clean
    s = " ".join([str(t).lower() for t in texts if isinstance(t, str)])
    # remove urls, mentions, hashtags but keep meaningful hashtags by removing # only:
    s = re.sub(r"http\S+|www\S+|@\w+", "", s)
    s = re.sub(r"#", "", s)
    # split on non-word chars
    tokens = re.findall(r"\b[^\d\W]{%d,}\b" % min_len, s, flags=re.UNICODE)
    # filter stopwords and very short tokens
    stopset = set(STOPWORDS) | IND_STOPWORDS
    if extra_stopwords:
        stopset |= set([w.lower() for w in extra_stopwords])
    tokens = [t for t in tokens if t not in stopset]
    # optional: further stemming already done earlier in pipeline; if not, ok
    return Counter(tokens)

def generate_wordcloud(texts, sentiment_label, save_to_file=True,
                    mask_path=None, max_words=200, colormap="tab20",
                    background_color="white", prefer_horizontal=0.9,
                    random_state=42):
    """
    texts: iterable/series of cleaned text strings (preferably after stem)
    sentiment_label: used for filename
    mask_path: optional path to PNG image to use as mask (white background, shape)
    """
    # Build freq dictionary
    freqs = _tokenize_and_filter(texts, min_len=3)
    if not freqs:
        freqs = Counter({"no_data": 1})

    # Prepare mask
    mask = None
    if mask_path and os.path.isfile(mask_path):
        mask_img = Image.open(mask_path).convert("L")
        mask = np.array(mask_img)
    else:
        # use circular mask by default
        mask = _make_circle_mask(size=800)

    wc = WordCloud(
        width=1200,
        height=800,
        background_color=background_color,
        max_words=max_words,
        mask=mask,
        colormap=colormap,
        prefer_horizontal=prefer_horizontal,
        contour_width=1,
        contour_color="#333333",
        random_state=random_state,
        scale=2  # generate at higher resolution then downsample for crisp PNG
    )

    # If you want bigrams preserved, you could set collocations=True
    wc.generate_from_frequencies(freqs)

    # Save file
    out_dir = os.path.dirname(__file__) or "visuals"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wordcloud_{sentiment_label}.png")

    wc.to_file(out_path)

    return out_path
