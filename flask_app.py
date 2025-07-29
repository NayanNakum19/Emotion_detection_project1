import os
import tempfile
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
from utils.feature_utils import extract_mfcc_cnn

# 🔧 Set ffmpeg converter path explicitly
AudioSegment.converter = r"C:\ffmpeg\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

# Suppress TensorFlow info/warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask
app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("models/cnn_emotion_model.h5")
labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

@app.route("/")
def index():
    # Render frontend page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check file
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]

    # Save raw upload to a temp webm file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        raw_path = tmp.name
        f.save(raw_path)

    print(f"📥 Received file: {raw_path} ({os.path.getsize(raw_path)} bytes)")

    # Convert webm/opus to wav
    try:
        audio = AudioSegment.from_file(raw_path, format="webm")
        wav_path = raw_path + "_converted.wav"
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio.export(wav_path, format="wav")
        print(f"✅ Converted to WAV: {wav_path} ({os.path.getsize(wav_path)} bytes)")
    except Exception as e:
        print("❌ Conversion error:", e)
        return jsonify({"error": f"Conversion to WAV failed: {e}"}), 500

    # Extract features
    mfcc = extract_mfcc_cnn(wav_path)
    if mfcc is None:
        return jsonify({"error": "Could not process audio"}), 500

    # Predict
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    pred = model.predict(mfcc)
    idx = int(np.argmax(pred))
    print(f"✅ Predicted: {labels[idx]}")

    return jsonify({
        "emotion": labels[idx],
        "confidences": {labels[i]: float(p) for i, p in enumerate(pred[0])}
    })

if __name__ == "__main__":
    # Run Flask (no reloader for stability)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
