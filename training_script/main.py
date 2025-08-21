import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

# --------------------------
# 0️⃣ Initialize global lists
epoch_settings = [10.00,12.00]
final_val_accuracies = [.8245,0.82]
final_val_losses = [0.5678,0.7]

# 1️⃣ Load dataset
df = pd.read_csv('/content/sample_data/datasets/issue_classification_5000.csv')

# 2️⃣ Encode sentiment labels
label_encoder = LabelEncoder()
df['issue_type'] = label_encoder.fit_transform(df['issue_type'])
num_labels = df['issue_type'].nunique()

# 3️⃣ Preprocess text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

df['complaint_text'] = df['complaint_text'].apply(clean_text)

# 4️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['complaint_text'], df['issue_type'], test_size=0.2, random_state=42
)

# 5️⃣ Convert to NumPy arrays for tf.data.Dataset
X_train = X_train.astype(str).to_numpy()
X_test = X_test.astype(str).to_numpy()
y_train = y_train.astype(np.float32).to_numpy()
y_test = y_test.astype(np.float32).to_numpy()

# 6️⃣ Text vectorization
vocab_size = 10000
max_len = 100
vectorizer = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=max_len
)
vectorizer.adapt(X_train)

# 7️⃣ Create tf.data.Dataset
batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 8️⃣ Build the model
model = models.Sequential([
    vectorizer,
    layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_labels, activation='softmax')
])

# 9️⃣ Compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🔟 Train
model_epochs_size = 20
early_stops = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

history = model.fit(train_ds, validation_data=val_ds, epochs=model_epochs_size, callbacks=[early_stops])

# 1️⃣1️⃣ Evaluate
loss, accuracy = model.evaluate(val_ds)
print(f"Test Accuracy: {accuracy:.2f}")

# 1️⃣2️⃣ Predict
# sample_text = tf.data.Dataset.from_tensor_slices(["Please clean garbage from Pine Road street."]).batch(1)
# prediction = model.predict(sample_text)

# 1) store the prediction and all percent ration
# probs = prediction[0]                      # shape: (num_labels,)
# percents = (probs * 100).astype(float)

# 2) Get predicted class index and label
# pred_idx = int(np.argmax(probs))
# pred_label = label_encoder.inverse_transform([pred_idx])[0]
# pred_confidence = float(percents[pred_idx])

# print(f"Predicted label: {pred_label} ({pred_confidence:.2f}%)")

# --------------------------
# 1️⃣6️⃣ Save the trained model for deployment
# This saves: model architecture, weights, optimizer state, AND the vectorizer inside
model.save("issue_classifier_model.keras")

# --------------------------
# 1️⃣7️⃣ Save the LabelEncoder (needed to decode labels if multiclass later)
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

# --------------------------
# 1️⃣8️⃣ Save training history (optional but useful for monitoring)
import json
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("✅ Model, label encoder, and training history saved successfully!")
