import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

# Re-define Attention layer (must be identical to training file)
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

# Load test data
X_test_eeg = np.load('X_test_eeg.npy')
X_test_eog = np.load('X_test_eog.npy')
X_test_emg = np.load('X_test_emg.npy')
y_test = np.load('y_test.npy')

# Load model
model = load_model("sleep_stage_classifier.h5", custom_objects={'Attention': Attention})

# Predict
y_pred_prob = model.predict([X_test_eeg, X_test_eog, X_test_emg])
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['W', 'N1', 'N2', 'N3', 'REM'], 
            yticklabels=['W', 'N1', 'N2', 'N3', 'REM'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
