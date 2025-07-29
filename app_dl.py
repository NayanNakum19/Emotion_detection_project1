import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go

from utils.feature_utils import extract_mfcc_cnn
from utils.plot_utils import (
    plot_waveform,
    plot_spectrogram,
    update_session_log,
    plot_emotion_trend,
)

# ------------------- Streamlit Page Setup -------------------
st.set_page_config(page_title="🎙 Emotion Detection", layout="centered")
st.title("🤖 Voice-Based Emotion Detection")
st.markdown("🎛 **You can either record from microphone or upload a `.wav` file.**")

# ------------------- Choose Input Method -------------------
input_method = st.radio("Select Input Method:", ["🎤 Microphone", "📂 Upload File"])

path = None  # will hold final audio path

# ========================= 🎤 Microphone Recording =========================
if input_method == "🎤 Microphone":
    # Find a valid input device
    def get_default_input_device():
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    return i
        except Exception as e:
            st.warning(f"⚠️ Could not list input devices: {e}")
        return None

    device_id = get_default_input_device()
    if device_id is None:
        st.error("❌ No input device found! Please connect a microphone.")
    else:
        duration = st.slider("⏱️ Select Recording Duration (seconds)", 1, 10, 3)
        if st.button("🎧 Record Now"):
            fs = 22050  # sample rate
            try:
                st.info("🎙 Recording in progress...")
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device_id)
                sd.wait()  # wait until recording is finished
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                    wav.write(tmpfile.name, fs, recording)
                    path = tmpfile.name
                st.success("✅ Recording complete!")
            except Exception as e:
                st.error(f"Microphone error: {e}")

# ========================= 📂 File Upload =========================
if input_method == "📂 Upload File":
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(uploaded_file.read())
            path = tmpfile.name
        st.success("✅ File uploaded successfully!")

# ========================= Process Audio if Available =========================
if path:
    # Plot waveform
    st.subheader("🎵 Voice Waveform")
    st.pyplot(plot_waveform(path))

    # Plot spectrogram
    st.subheader("🔊 Spectrogram")
    st.pyplot(plot_spectrogram(path))

    # Load model
    try:
        model = tf.keras.models.load_model("models/cnn_emotion_model.h5")
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    emotion_labels = [
        "neutral", "calm", "happy", "sad",
        "angry", "fearful", "disgust", "surprised"
    ]

    # Extract features
    mfcc = extract_mfcc_cnn(path)
    if mfcc is None:
        st.error("❌ Could not extract audio features.")
        st.stop()

    # Predict
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mfcc)
    emotion_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[emotion_index]

    # Display prediction
    st.markdown(f"### 🧠 Predicted Emotion: **{predicted_emotion.upper()}**")

    # Confidence bar chart
    confidences = prediction[0]
    fig = go.Figure([go.Bar(x=emotion_labels, y=confidences, marker_color="cyan")])
    fig.update_layout(title="🔍 Prediction Confidence", xaxis_title="Emotion", yaxis_title="Probability")
    st.plotly_chart(fig, use_container_width=True)

    # Update session log & plot trend
    update_session_log(predicted_emotion)
    trend_plot = plot_emotion_trend()
    if trend_plot:
        st.subheader("📈 Emotion Trend (Session)")
        st.pyplot(trend_plot)

    # Show animated emotion avatar GIF
    gif_map = {
        "happy": "https://media.giphy.com/media/1BdIPYgNlGyyQ/giphy.gif",
        "sad": "https://media.giphy.com/media/d2lcHJTG5Tscg/giphy.gif",
        "angry": "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",
        "calm": "https://media.giphy.com/media/l0ExdMHUDKteztyfe/giphy.gif",
        "neutral": "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif",
        "fearful": "https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif",
        "disgust": "https://media.giphy.com/media/Zau0yrl17uzdK/giphy.gif",
        "surprised": "https://media.giphy.com/media/l0IylOPCNkiqOgMyA/giphy.gif",
    }
    gif_url = gif_map.get(predicted_emotion, "")
    if gif_url:
        st.image(gif_url, width=300, caption=f"Emotion Avatar: {predicted_emotion.capitalize()}")
