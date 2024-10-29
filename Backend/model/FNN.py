import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


# Set up directory for outputs
current_directory = os.path.abspath(os.path.dirname(__file__))
outputs_directory = os.path.join(current_directory, 'outputs')
os.makedirs(outputs_directory, exist_ok=True)

# Load and preprocess data
data = pd.read_csv(r"C:\Users\Simon Risk\OneDrive\Desktop\speech_app\speech-app-main\all_word_audio_features.csv")

data = data.sample(frac=1, random_state=42).reset_index(drop=True)
data = data.drop(columns=['filename', 'start', 'end'])  # Drop unnecessary columns

# Clean and preprocess tempo column
data['tempo'] = data['tempo'].str.strip('[]').astype(float)

# Check for duplicates or nulls
def check_duplicates_or_nulls(data):
    duplicates = data.duplicated().sum() > 0
    nulls = data.isnull().sum().sum() > 0
    return duplicates or nulls

if check_duplicates_or_nulls(data):
    print('Duplicates or null values found in data')
else:
    print('No duplicates or null values found in data')

# Split data into features and target
X = np.array(data.drop(columns=['word']), dtype=float)  # Exclude the 'word' column
y = data['word']  # Use 'word' as the target variable

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Define Keras model
def create_model(input_dim, hidden_dim, output_dim):
    model = Sequential()
    model.add(Dense(hidden_dim, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.5))  # Dropout layer with 50% drop rate
    model.add(Dense(hidden_dim * 2, activation='relu'))  # Increase hidden layer size
    model.add(Dropout(0.5))  # Dropout layer with 50% drop rate
    model.add(Dense(hidden_dim, activation='relu'))  # Another layer
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))  # Output layer
    model.add(BatchNormalization())
    return model



# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = len(np.unique(y))  # Number of classes
model = create_model(input_dim, hidden_dim, output_dim)
learning_rate = 0.001  # You can adjust this value was
optimizer = Adam(learning_rate=learning_rate)

# Compile model with the new optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Training function with Early Stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train model with the new callback
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping, reduce_lr])


# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy and classification report
accuracy = accuracy_score(y_test, y_pred_classes)
report = classification_report(y_test, y_pred_classes, target_names=encoder.classes_)
print("Keras Model Accuracy:", accuracy)
print("Keras Model Classification Report:\n", report)

# Save model and preprocessing objects
model.save(os.path.join(outputs_directory, 'keras_model_final.h5'))
joblib.dump(scaler, os.path.join(outputs_directory, 'scaler.pkl'))
joblib.dump(encoder, os.path.join(outputs_directory, 'encoder.pkl'))
print("Model and preprocessing objects saved.")
