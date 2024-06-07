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

# Function to create and save a spectrogram with increased Mel bands
def create_spectrogram(audio_file, output_folder, n_mels=128, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(audio_file, sr=None)

        # Apply a band-pass filter to focus on relevant frequencies (300 Hz to 3400 Hz)
        y = librosa.effects.preemphasis(y)
        y = librosa.effects.trim(y, top_db=30)[0]
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', fmax=450)
        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(audio_file))[0]}_spectrogram.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Spectrogram saved to {output_path}")

    except Exception as e:
        print(f"Error creating spectrogram for {audio_file}: {e}")

# Main function to process all audio files in a folder
def process_audio_folder(folder_path, output_folder, n_mels=128, n_fft=2048, hop_length=512):
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
                create_spectrogram(file_path, output_folder, n_mels, n_fft, hop_length)
    
    return all_features, filenames

# Function to process word data
def process_word_data(word_folder_path, spectrogram_output_folder, csv_output_file):
    # Ensure the output folders exist
    os.makedirs(spectrogram_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

    # Process the specified word folder with increased Mel bands
    print(f"Processing word folder: {word_folder_path}")
    all_features, filenames = process_audio_folder(word_folder_path, spectrogram_output_folder, n_mels=128, n_fft=2048, hop_length=512)

    # Write results to a CSV file
    print("Writing results to CSV file...")

    # Flattening and creating DataFrame
    flattened_data = []
    for idx, features in enumerate(all_features):
        num_frames = features['mfcc'].shape[0]
        for frame in range(num_frames):
            frame_data = {key: features[key][frame] for key in features}
            frame_data['filename'] = filenames[idx]
            flattened_data.append(frame_data)

    df = pd.DataFrame(flattened_data)
    df.to_csv(csv_output_file, index=False)
    print("CSV file generation completed.")

    # Indicate completion
    print("Feature extraction, spectrogram generation, and CSV generation completed.")

# Specify paths for "tree"
tree_word_folder_path = '/Users/simonrisk/Desktop/speech_therapy/archive/augmented_dataset/augmented_dataset/tree/'
tree_spectrogram_output_folder = '/Users/simonrisk/Desktop/speech_therapy/spectrograms/tree'
tree_csv_output_file = '/Users/simonrisk/Desktop/speech_therapy/csvs/tree.csv'

# Specify paths for "three"
three_word_folder_path = '/Users/simonrisk/Desktop/speech_therapy/archive/augmented_dataset/augmented_dataset/three/'
three_spectrogram_output_folder = '/Users/simonrisk/Desktop/speech_therapy/spectrograms/three'
three_csv_output_file = '/Users/simonrisk/Desktop/speech_therapy/csvs/three.csv'

# Process "tree" data
process_word_data(tree_word_folder_path, tree_spectrogram_output_folder, tree_csv_output_file)

# Process "three" data
process_word_data(three_word_folder_path, three_spectrogram_output_folder, three_csv_output_file)
