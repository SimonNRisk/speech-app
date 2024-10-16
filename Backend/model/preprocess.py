import os
import librosa
import pandas as pd
from pydub import AudioSegment
import numpy as np
import soundfile as sf

#function to extract features from an audio segment
def extract_features_from_segment(y, sr, start_time, end_time):
    segment = y[start_time:end_time]

    chroma_stft = librosa.feature.chroma_stft(y=segment, sr=sr)
    rms = librosa.feature.rms(y=segment)
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)
    harmony, perceptr = librosa.effects.hpss(segment)
    tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)

    features = {
        'chroma_stft_mean': chroma_stft.mean() if chroma_stft.size else 0,
        'chroma_stft_var': chroma_stft.var() if chroma_stft.size else 0,
        'rms_mean': rms.mean() if rms.size else 0,
        'rms_var': rms.var() if rms.size else 0,
        'spectral_centroid_mean': spectral_centroid.mean() if spectral_centroid.size else 0,
        'spectral_centroid_var': spectral_centroid.var() if spectral_centroid.size else 0,
        'spectral_bandwidth_mean': spectral_bandwidth.mean() if spectral_bandwidth.size else 0,
        'spectral_bandwidth_var': spectral_bandwidth.var() if spectral_bandwidth.size else 0,
        'rolloff_mean': rolloff.mean() if rolloff.size else 0,
        'rolloff_var': rolloff.var() if rolloff.size else 0,
        'zero_crossing_rate_mean': zero_crossing_rate.mean() if zero_crossing_rate.size else 0,
        'zero_crossing_rate_var': zero_crossing_rate.var() if zero_crossing_rate.size else 0,
        'harmony_mean': harmony.mean() if harmony.size else 0,
        'harmony_var': harmony.var() if harmony.size else 0,
        'perceptr_mean': perceptr.mean() if perceptr.size else 0,
        'perceptr_var': perceptr.var() if perceptr.size else 0,
        'tempo': tempo,
    }

    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = mfcc[i-1].mean() if mfcc.shape[0] >= i else 0
        features[f'mfcc{i}_var'] = mfcc[i-1].var() if mfcc.shape[0] >= i else 0

    return features

#function to load audio file
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
    except sf.LibsndfileError:
        print(f"LibsndfileError: {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
    return y, sr

#function to extract features from an audio file
def extract_features(audio_file, segment_duration=1):
    try:
        y, sr = load_audio(audio_file)
        if y is None:
            return []

        total_duration = len(y) / sr
        segment_length = int(sr * segment_duration)

        features_list = []

        for start in range(0, len(y), segment_length):
            end = start + segment_length
            if end <= len(y):
                segment_features = extract_features_from_segment(y, sr, start, end)
                all_features = segment_features
                all_features['filename'] = os.path.basename(audio_file)
                all_features['start'] = start / sr
                all_features['end'] = end / sr
                features_list.append(all_features)

        return features_list

    except Exception as e:
        print(f"Error extracting features from {audio_file}: {e}")
        return []
    
#list of words
words = ["three", "tree"]

#base folder containing word subfolders
#base_folder_path = '/Users/simonrisk/Desktop/speech_therapy/archive/augmented_dataset/augmented_dataset'
base_folder_path = '/Users/simonrisk/Desktop/speech_therapy/external_data/'

#function to process a folder of audio files
def process_audio_folder(folder_path, word_label):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            features_list = extract_features(file_path)
            for features in features_list:
                features['word'] = word_label
                results.append(features)
    return results


all_results = []

for word in words:
    print(f"Processing word: {word}")
    folder_path = os.path.join(base_folder_path, word)
    word_results = process_audio_folder(folder_path, word)
    all_results.extend(word_results)
    print(f"Completed processing word: {word}")

print("Writing results to CSV file...")
df = pd.DataFrame(all_results)
csv_file_path = '/Users/simonrisk/Desktop/speech_therapy/file_features.csv'
df.to_csv(csv_file_path, index=False)
print("CSV file generation completed.")

df.head()

#get summary statistics
df.describe(include='all')

