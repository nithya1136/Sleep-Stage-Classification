import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# -------- Custom Attention Layer --------
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

# -------- Load test data --------
X_test_eeg = np.load('X_test_eeg.npy')
X_test_eog = np.load('X_test_eog.npy')
X_test_emg = np.load('X_test_emg.npy')
y_test = np.load('y_test.npy')

# -------- Load trained model --------
model = load_model('sleep_stage_classifier.h5', custom_objects={'Attention': Attention})

# -------- SHAP Explanation --------
print("Explaining with SHAP...")
# Use subset of test data as background
background = [X_test_eeg[:100], X_test_eog[:100], X_test_emg[:100]]

# Create SHAP GradientExplainer
explainer = shap.GradientExplainer(model, background)

# Pick one sample to explain
sample_idx = 0
eeg_sample = X_test_eeg[sample_idx:sample_idx+1]
eog_sample = X_test_eog[sample_idx:sample_idx+1]
emg_sample = X_test_emg[sample_idx:sample_idx+1]

shap_values = explainer.shap_values([eeg_sample, eog_sample, emg_sample])

# Optional: save or visualize SHAP values
# Example: shap.image_plot(shap_values[0], eeg_sample)

# -------- Integrated Gradients Explanation --------
print("Explaining with Integrated Gradients...")
# Choose class index for explanation, e.g., class 4 (REM)
score = CategoricalScore([3])
modifier = ReplaceToLinear()
saliency = Saliency(model, model_modifier=modifier)

# Compute saliency for EEG only (you can repeat for EOG/EMG)
eeg_attributions = saliency(score, [eeg_sample, eog_sample, emg_sample])

# Plot EEG attribution
plt.figure(figsize=(12, 4))
plt.plot(eeg_attributions[0][0].squeeze(), label='EEG Attribution')
plt.title("Integrated Gradients - EEG Attribution for Class REM")
plt.xlabel("Time Steps")
plt.ylabel("Attribution")
plt.legend()
plt.tight_layout()
plt.savefig("integrated_gradients_eeg.png")
plt.show()