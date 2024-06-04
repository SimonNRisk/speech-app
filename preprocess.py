import os
import librosa
import librosa.display
import pandas as pd
from pydub import AudioSegment
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_file):
    sound = AudioSegment.from_mp3(mp3_file)
    wav_file = tempfile.mktemp(suffix='.wav')
    sound.export(wav_file, format="wav")
    return wav_file

# Function to extract features from an audio file for LSTM
def extract_features(audio_file):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)

        # Compute features over time
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Combine features into a dictionary
        features = {
            'chroma_stft': chroma_stft.T,
            'rms': rms.T,
            'spectral_centroid': spectral_centroid.T,
            'spectral_bandwidth': spectral_bandwidth.T,
            'rolloff': rolloff.T,
            'zero_crossing_rate': zero_crossing_rate.T,
            'mfcc': mfcc.T
        }

        return features

    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return None

# Function to create and save a spectrogram
def create_spectrogram(audio_file, output_folder):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {os.path.basename(audio_file)}')
        plt.tight_layout()

        output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(audio_file))[0]}_spectrogram.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Spectrogram saved to {output_path}")

    except Exception as e:
        print(f"Error creating spectrogram for {audio_file}: {e}")

# Main function to process all audio files in a folder
def process_audio_folder(folder_path, output_folder):
    all_features = []
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.mp3'):
                try:
                    file_path = convert_mp3_to_wav(file_path)
                except Exception as e:
                    print(f"Error converting {file_path} to WAV: {e}")
                    continue

            features = extract_features(file_path)
            if features:
                all_features.append(features)
                filenames.append(filename)
                create_spectrogram(file_path, output_folder)
    
    return all_features, filenames

# Specify the word folder to process
word_folder_path = '/Users/simonrisk/Desktop/speech_therapy/archive/augmented_dataset/augmented_dataset/tree/'
spectrogram_output_folder = '/Users/simonrisk/Desktop/speech_therapy/spectrograms/tree'
csv_output_folder = '/Users/simonrisk/Desktop/speech_therapy/csvs'

# Ensure the output folders exist
os.makedirs(spectrogram_output_folder, exist_ok=True)
os.makedirs(csv_output_folder, exist_ok=True)

# Process the specified word folder
print(f"Processing word folder: {word_folder_path}")
all_features, filenames = process_audio_folder(word_folder_path, spectrogram_output_folder)

# Write results to a CSV file
print("Writing results to CSV file...")
csv_file_path = os.path.join(csv_output_folder, 'tree.csv')

# Flattening and creating DataFrame
flattened_data = []
for idx, features in enumerate(all_features):
    num_frames = features['mfcc'].shape[0]
    for frame in range(num_frames):
        frame_data = {key: features[key][frame] for key in features}
        frame_data['filename'] = filenames[idx]
        flattened_data.append(frame_data)

df = pd.DataFrame(flattened_data)
df.to_csv(csv_file_path, index=False)
print("CSV file generation completed.")

# Indicate completion
print("Feature extraction, spectrogram generation, and CSV generation completed.")
