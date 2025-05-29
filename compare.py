import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten, Dense,
                                     Dropout, Concatenate, Bidirectional, LSTM, Layer)
from tensorflow.keras.utils import to_categorical
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
# -------- Load test data --------
X_test_eeg = np.load('X_test_eeg.npy')
X_test_eog = np.load('X_test_eog.npy')
X_test_emg = np.load('X_test_emg.npy')
y_test = np.load('y_test.npy')

# -------- Load trained model --------
model = load_model('sleep_stage_classifier.h5', custom_objects={'Attention': Attention})

# -------- Predict labels --------
y_pred_probs = model.predict([X_test_eeg, X_test_eog, X_test_emg])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
from sklearn.metrics import accuracy_score
print("Test Accuracy:", accuracy_score(y_true, y_pred))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["W", "N1", "N2", "N3", "R"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
for i in range(10):  # first 10 test samples
    print(f"Sample {i}: True = {y_true[i]}, Predicted = {y_pred[i]}")