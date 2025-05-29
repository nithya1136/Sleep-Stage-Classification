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

# Attention Layer 
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

#  Model Architecture 
def build_model(input_shape):
    def branch(input_signal):
        x = Conv1D(64, 5, activation='relu', padding='same')(input_signal)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        return x

    input_eeg = Input(shape=input_shape)
    input_eog = Input(shape=input_shape)
    input_emg = Input(shape=input_shape)

    eeg_branch = branch(input_eeg)
    eog_branch = branch(input_eog)
    emg_branch = branch(input_emg)

    merged = Concatenate(axis=-1)([eeg_branch, eog_branch, emg_branch])
    x = Bidirectional(LSTM(64, return_sequences=True))(merged)
    x = Attention()(x)
    x = Dropout(0.5)(x)
    output = Dense(5, activation='softmax')(x)

    model = Model(inputs=[input_eeg, input_eog, input_emg], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#  Load and Preprocess Data 
def load_sleepedf_data(data_dir, sampling_rate=100, epoch_sec=30):
    eeg_list, eog_list, emg_list, labels = [], [], [], []
    label_map = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # Combine stage 3 & 4
        'Sleep stage R': 4
    }

    for file in os.listdir(data_dir):
        if file.endswith("PSG.edf"):
            psg_path = os.path.join(data_dir, file)
            hypnogram_path = os.path.join(data_dir, file.replace("PSG.edf", "Hypnogram.edf").replace("SC", "SC").replace("E0", "EC"))

            try:
                raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
                annotations = mne.read_annotations(hypnogram_path)
                raw.set_annotations(annotations)
                raw.resample(sampling_rate)

                eeg = raw.copy().pick_channels(['EEG Fpz-Cz']).get_data()[0]
                eog = raw.copy().pick_channels(['EOG horizontal']).get_data()[0]
                emg = raw.copy().pick_channels(['EMG submental']).get_data()[0]

                events, _ = mne.events_from_annotations(raw, event_id=label_map)

                for i in range(len(events)):
                    start_sample = events[i][0]
                    label = events[i][2]
                    end_sample = start_sample + epoch_sec * sampling_rate

                    if end_sample <= len(eeg):
                        eeg_list.append(eeg[start_sample:end_sample])
                        eog_list.append(eog[start_sample:end_sample])
                        emg_list.append(emg[start_sample:end_sample])
                        labels.append(label)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    eeg_list, eog_list, emg_list, labels = shuffle(eeg_list, eog_list, emg_list, labels)
    eeg_list = np.array(eeg_list)
    eog_list = np.array(eog_list)
    emg_list = np.array(emg_list)
    labels = np.array(labels)

    #  Feature Scaling 
    eeg_list = StandardScaler().fit_transform(eeg_list).reshape(-1, eeg_list.shape[1], 1)
    eog_list = StandardScaler().fit_transform(eog_list).reshape(-1, eog_list.shape[1], 1)
    emg_list = StandardScaler().fit_transform(emg_list).reshape(-1, emg_list.shape[1], 1)

    #  One-hot encoding 
    labels = to_categorical(labels, num_classes=5)

    return eeg_list, eog_list, emg_list, labels

#  Main Execution 
if __name__ == "__main__":
    data_dir = "C:/Users/ASUS/OneDrive/Desktop/Sleep/data"
    sampling_rate = 100
    epoch_sec = 30

    print("Loading and preprocessing data...")
    """
    eeg, eog, emg, labels = load_sleepedf_data(data_dir, sampling_rate, epoch_sec)
    
    np.save('eeg.npy', eeg)
    np.save('eog.npy', eog)
    np.save('emg.npy', emg)
    np.save('labels.npy', labels)

    print("Data saved to .npy files!")
    """
    eeg = np.load('eeg.npy')
    eog = np.load('eog.npy')
    emg = np.load('emg.npy')
    labels = np.load('labels.npy')

    print("Loaded EEG shape:", eeg.shape)
    print("Loaded EOG shape:", eog.shape)
    print("Loaded EMG shape:", emg.shape)
    print("Loaded labels shape:", labels.shape)
    # Train-test split
    X_train_eeg, X_test_eeg, X_train_eog, X_test_eog, X_train_emg, X_test_emg, y_train, y_test = train_test_split(
        eeg, eog, emg, labels, test_size=0.2, random_state=42
    )
    np.save('X_test_eeg.npy', X_test_eeg)
    np.save('X_test_eog.npy', X_test_eog)
    np.save('X_test_emg.npy', X_test_emg)
    np.save('y_test.npy', y_test)

    # Compute class weights
    y_labels_flat = np.argmax(y_train, axis=1)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_labels_flat), y=y_labels_flat)
    class_weights = dict(enumerate(weights))
    print("Class Weights:", class_weights)

    input_shape = X_train_eeg.shape[1:]  # (time, channels)
    model = build_model(input_shape)
    model.summary()

    print("Training the model...")
    model.fit(
        [X_train_eeg, X_train_eog, X_train_emg],
        y_train,
        validation_data=([X_test_eeg, X_test_eog, X_test_emg], y_test),
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )

    print("Evaluating the model...")
    loss, acc = model.evaluate([X_test_eeg, X_test_eog, X_test_emg], y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    print("Saving model...")
    model.save("sleep_stage_classifier.h5")
