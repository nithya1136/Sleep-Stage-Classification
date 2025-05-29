import streamlit as st
import mne
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import tempfile
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sleep_stage_classifier.h5", custom_objects={'Attention': Attention})

# Preprocess uploaded EDF file
def preprocess_edf(uploaded_file, sampling_rate=100, epoch_sec=30):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        raw.resample(sampling_rate)

        eeg = raw.copy().pick_channels(['EEG Fpz-Cz']).get_data()[0]
        eog = raw.copy().pick_channels(['EOG horizontal']).get_data()[0]
        emg = raw.copy().pick_channels(['EMG submental']).get_data()[0]

        segment_len = sampling_rate * epoch_sec
        segments = len(eeg) // segment_len

        eeg_segments = np.array([eeg[i*segment_len:(i+1)*segment_len] for i in range(segments)])
        eog_segments = np.array([eog[i*segment_len:(i+1)*segment_len] for i in range(segments)])
        emg_segments = np.array([emg[i*segment_len:(i+1)*segment_len] for i in range(segments)])

        # Feature scaling
        eeg_scaled = StandardScaler().fit_transform(eeg_segments).reshape(-1, segment_len, 1)
        eog_scaled = StandardScaler().fit_transform(eog_segments).reshape(-1, segment_len, 1)
        emg_scaled = StandardScaler().fit_transform(emg_segments).reshape(-1, segment_len, 1)

        return eeg_scaled, eog_scaled, emg_scaled

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

    finally:
        os.remove(tmp_path)

# Mapping from predicted class index to label
stage_map = {
    0: "Wake (W)",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

# Streamlit UI
st.set_page_config(page_title="Sleep Stage Predictor", layout="wide")
st.title("🛌 Sleep Stage Predictor from EDF File")
st.write("Upload an EDF file containing EEG, EOG, and EMG signals (e.g., Sleep-EDF format) to get predicted sleep stages for 30-second epochs.")

uploaded_file = st.file_uploader("📂 Upload your PSG EDF file", type=["edf"])

if uploaded_file is not None:
    st.info("Reading and preprocessing the file...")

    eeg, eog, emg = preprocess_edf(uploaded_file)

    if eeg is not None:
        model = load_model()
        st.info("Running predictions...")
        predictions = model.predict([eeg, eog, emg])
        predicted_classes = np.argmax(predictions, axis=1)
        stage_names = [stage_map[i] for i in predicted_classes]

        st.success("✅ Prediction complete!")

        st.subheader("Predicted Sleep Stages (per 30s epoch):")
        st.write(stage_names)

        # Show distribution as bar chart
        unique, counts = np.unique(predicted_classes, return_counts=True)
        stage_dist = {stage_map[u]: c for u, c in zip(unique, counts)}
        st.subheader("📊 Sleep Stage Distribution")
        st.bar_chart(stage_dist)

        # Optionally allow download of results
        if st.button("📥 Download Predictions as CSV"):
            import pandas as pd
            df = pd.DataFrame({
                "Epoch": np.arange(len(stage_names)),
                "Predicted Sleep Stage": stage_names
            })
            st.download_button("Download CSV", df.to_csv(index=False), file_name="predicted_sleep_stages.csv", mime="text/csv")
