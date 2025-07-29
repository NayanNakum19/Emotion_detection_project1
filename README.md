# Emotion_detection_project1
# 🎙️ Voice Emotion Detection with CNN & Streamlit

A real-time voice-based emotion recognition system using **Convolutional Neural Networks (CNN)**. Users can **record audio or upload `.wav` files**, visualize waveforms/spectrograms, and see emotion predictions with confidence levels and session tracking — all through an interactive **Streamlit UI**.

---

## 🚀 Features

- 🎧 **Record or Upload** audio input  
- 📊 **CNN-based prediction** on MFCC features  
- 📈 **Confidence chart** using Plotly bar graph  
- 🔉 **Waveform & Spectrogram visualizations**  
- 📆 **Emotion trend tracking** (logs over session)  
- 🤖 **Animated GIF/Emoji feedback** for emotions  
- ☁️ Easy to deploy on **Streamlit Cloud**, **Render**, or **Vercel**

---

## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow 2.15+
- Librosa, Sounddevice, Soundfile
- Streamlit
- Matplotlib, Plotly
- Scikit-learn

---

## 📦 Folder Structure

```

emotion\_detection/
│
├── app\_dl.py                    # Streamlit app (Record + Upload)
├── train\_cnn.py                 # CNN training script
├── models/
│   └── cnn\_emotion\_model.h5     # Trained model
├── data/
│   └── Audio\_Speech\_Actors...   # RAVDESS dataset
├── logs/
│   └── session\_emotions.csv     # Log for emotion trend
├── utils/
│   ├── feature\_utils.py         # MFCC feature extractor
│   └── plot\_utils.py            # Waveform/spectrogram/emotion trend
├── .streamlit/
│   └── config.toml              # Dark theme settings
├── .vscode/
│   └── launch.json              # Run config for Streamlit in VS Code
├── requirements.txt
└── README.md

````

---

## 🧪 Installation & Usage

```bash
# 1️⃣ Clone repo & create environment
git clone https://github.com/NayanNakum19/emotion_detection.git
cd emotion_detection
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate

# 2️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3️⃣ Run the app
streamlit run app_dl.py
```
