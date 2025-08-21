import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

##load model
model = tf.keras.models.load_model('issue_classifier_model.keras')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI(title="Issue Classifier Model")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_issue(input: TextInput):
    
    ## convert to tf.dataset
    sample_text = tf.data.Dataset.from_tensor_slices([input.text]).batch(1)

    prediction = model.predict(sample_text)

    probs = prediction[0]                      
    percents = (probs * 100).astype(float)

    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    pred_confidence = round(float(percents[pred_idx]), 2)

    return {"label": pred_label, "confidence": pred_confidence}