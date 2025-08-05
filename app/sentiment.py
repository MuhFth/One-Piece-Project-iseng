# Load model only once
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    
    label_map = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    return label_map[predicted_class]