import librosa
import numpy as np

def extract_mfcc_cnn(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, max(pad_width, 0))), mode='constant')
        return mfccs[:, :max_pad_len]
    except Exception as e:
        print(f"MFCC Extraction Error: {e}")
        return None
'''
import librosa
import numpy as np

def extract_mfcc_cnn(path, max_pad_len=174):
    try:
        y, sr = librosa.load(path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print("Feature extraction error:", e)
        return None'''
