import os
import librosa
import librosa.display
import pandas as pd
from pydub import AudioSegment
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# Function to convert MP3 to WAV, not used as of June 7
def convert_mp3_to_wav(mp3_file):
    sound = AudioSegment.from_mp3(mp3_file) #sound = the object taken from the mp3
    wav_file = tempfile.mktemp(suffix='.wav') #creates new temporary file with suffix.wav
    sound.export(wav_file, format="wav") #exports sound to wave_file under wav format
    return wav_file

# Function to extract features from an audio file for LSTM
def extract_features(audio_file, sr=22050):
    try:
        # Load the audio file with the desired sampling rate
        y, sr = librosa.load(audio_file, sr=sr)

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
def create_spectrogram(audio_file, output_folder, sr=22050, n_mels=256, n_fft=2048, hop_length=512, fmax=8000, word_type="tree"):
    try:
        y, sr = librosa.load(audio_file, sr=sr)

        # Apply a band-pass filter to focus on relevant frequencies (300 Hz to 3400 Hz)
        y = librosa.effects.preemphasis(y)
        y = librosa.effects.trim(y, top_db=30)[0]
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax)
        S_DB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(12, 6))
        librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None, fmax=fmax)
        
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])  # Remove the whitespace

        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        if word_type == "tree":
            output_path = os.path.join(output_folder, f"tree_spectrogram_{base_name}.png")
        else:
            output_path = os.path.join(output_folder, f"three_spectrogram_{base_name}.png")
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Spectrogram saved to {output_path}")

    except Exception as e:
        print(f"Error creating spectrogram for {audio_file}: {e}")

# Main function to process all audio files in a folder
def process_audio_folder(folder_path, output_folder, sr=22050, n_mels=256, n_fft=2048, hop_length=256, fmax=8000, word_type="tree"):
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

            features = extract_features(file_path, sr=sr)
            if features:
                all_features.append(features)
                filenames.append(filename)
                create_spectrogram(file_path, output_folder, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmax=fmax, word_type=word_type)
    
    return all_features, filenames

# Function to process word data
def process_word_data(word_folder_path, spectrogram_output_folder, csv_output_file, sr=22050, fmax=8000, word_type="tree"):
    # Ensure the output folders exist
    os.makedirs(spectrogram_output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

    # Process the specified word folder with increased Mel bands
    print(f"Processing word folder: {word_folder_path}")
    all_features, filenames = process_audio_folder(word_folder_path, spectrogram_output_folder, sr=sr, n_mels=256, n_fft=2048, hop_length=256, fmax=fmax, word_type=word_type)

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
tree_word_folder_path = '/Users/simonrisk/Desktop/speech_therapy_1/archive/augmented_dataset/augmented_dataset/tree/'
tree_spectrogram_output_folder = '/Users/simonrisk/Desktop/speech_therapy_1/spectrograms/tree'
tree_csv_output_file = '/Users/simonrisk/Desktop/speech_therapy_1/csvs/tree.csv'

# Specify paths for "three"
three_word_folder_path = '/Users/simonrisk/Desktop/speech_therapy_1/archive/augmented_dataset/augmented_dataset/three/'
three_spectrogram_output_folder = '/Users/simonrisk/Desktop/speech_therapy_1/spectrograms/three'
three_csv_output_file = '/Users/simonrisk/Desktop/speech_therapy_1/csvs/three.csv'

# Desired fmax value and sampling rate
desired_fmax = 8000  # Adjust this value as needed
desired_sr = 88200  # Set the desired sampling rate #was half of this

# Process "tree" data
process_word_data(tree_word_folder_path, tree_spectrogram_output_folder, tree_csv_output_file, sr=desired_sr, fmax=desired_fmax, word_type="tree")

# Process "three" data
process_word_data(three_word_folder_path, three_spectrogram_output_folder, three_csv_output_file, sr=desired_sr, fmax=desired_fmax, word_type="three")
