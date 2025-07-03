import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# === Load dataset
df = pd.read_csv("faq_dataset12.csv")
labels = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["label"].map(label2id)

model_path = "./model_chatbot_classify"
model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = TFAutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# === Prediksi
texts = ["kenapa saya ga bisa login?"]
inputs = tokenizer(texts, return_tensors="tf", truncation=True, padding=True)
logits = model(**inputs).logits
pred_ids = tf.argmax(logits, axis=1).numpy()

# === Konversi prediksi id ke nama label
pred_labels = [id2label[i] for i in pred_ids]

print(pred_labels)
print(pred_ids)
print(logits)
print(id2label)