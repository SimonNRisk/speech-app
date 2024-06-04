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

# Function to extract features from an audio file
def extract_features(audio_file):
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=None)

        # Compute chroma feature from the waveform and sample rate
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        # Compute Root Mean Square (RMS) energy for each frame
        rms = librosa.feature.rms(y=y)
        # Compute spectral centroid, which indicates where the center of mass for a sound is located
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        # Compute spectral bandwidth, which is a measure of the width of the band of frequencies
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        # Compute spectral rolloff point, which is the frequency below which a specified percentage of the total spectral energy lies
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        # Compute zero crossing rate, which is the rate at which the signal changes sign
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        # Compute harmony and perceptr (percussive) components of the audio
        harmony, perceptr = librosa.effects.hpss(y)
        # Compute tempo (beats per minute)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Compute Mel-frequency cepstral coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        
        # Aggregate the features into a dictionary
        features = {
            'filename': os.path.basename(audio_file),  # Name of the audio file
            'length': len(y) / sr,  # Length of the audio in seconds
            'chroma_stft_mean': chroma_stft.mean() if chroma_stft.size else 0,  # Mean of chroma STFT
            'chroma_stft_var': chroma_stft.var() if chroma_stft.size else 0,  # Variance of chroma STFT
            'rms_mean': rms.mean() if rms.size else 0,  # Mean of RMS energy
            'rms_var': rms.var() if rms.size else 0,  # Variance of RMS energy
            'spectral_centroid_mean': spectral_centroid.mean() if spectral_centroid.size else 0,  # Mean of spectral centroid
            'spectral_centroid_var': spectral_centroid.var() if spectral_centroid.size else 0,  # Variance of spectral centroid
            'spectral_bandwidth_mean': spectral_bandwidth.mean() if spectral_bandwidth.size else 0,  # Mean of spectral bandwidth
            'spectral_bandwidth_var': spectral_bandwidth.var() if spectral_bandwidth.size else 0,  # Variance of spectral bandwidth
            'rolloff_mean': rolloff.mean() if rolloff.size else 0,  # Mean of spectral rolloff point
            'rolloff_var': rolloff.var() if rolloff.size else 0,  # Variance of spectral rolloff point
            'zero_crossing_rate_mean': zero_crossing_rate.mean() if zero_crossing_rate.size else 0,  # Mean of zero crossing rate
            'zero_crossing_rate_var': zero_crossing_rate.var() if zero_crossing_rate.size else 0,  # Variance of zero crossing rate
            'harmony_mean': harmony.mean() if harmony.size else 0,  # Mean of harmonic component
            'harmony_var': harmony.var() if harmony.size else 0,  # Variance of harmonic component
            'perceptr_mean': perceptr.mean() if perceptr.size else 0,  # Mean of percussive component
            'perceptr_var': perceptr.var() if perceptr.size else 0,  # Variance of percussive component
            'tempo': tempo,  # Tempo (beats per minute)
        }
        
        # Add MFCCs (Mel-frequency cepstral coefficients) mean and variance for the first 20 coefficients
        for i in range(1, 21):
            features[f'mfcc{i}_mean'] = mfcc[i-1].mean() if mfcc.shape[0] >= i else 0  # Mean of MFCC
            features[f'mfcc{i}_var'] = mfcc[i-1].var() if mfcc.shape[0] >= i else 0  # Variance of MFCC
        
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
    results = []
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
                results.append(features)
                create_spectrogram(file_path, output_folder)
    
    return results

# Specify the word folder to process
word_folder_path = '/Users/simonrisk/Desktop/speech_therapy/words/rabbit'  # Example: word = rabbit
spectrogram_output_folder = '/Users/simonrisk/Desktop/speech_therapy/spectrograms/rabbit'

# Ensure the output folder exists
os.makedirs(spectrogram_output_folder, exist_ok=True)

# Process the specified word folder
print(f"Processing word folder: {word_folder_path}")
word_results = process_audio_folder(word_folder_path, spectrogram_output_folder)

# Write results to a CSV file
print("Writing results to CSV file...")
df = pd.DataFrame(word_results)
csv_file_path = '/Users/simonrisk/Desktop/speech_therapy/csvs/rabbit.csv'
df.to_csv(csv_file_path, index=False)
print("CSV file generation completed.")

# Indicate completion
print("Feature extraction, spectrogram generation, and CSV generation completed.")
