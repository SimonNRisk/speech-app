from pydub import AudioSegment
import numpy as np
import os
import random

# Function to generate white noise
def generate_white_noise(duration_ms, volume=-20.0):
    # Create an array of white noise samples using NumPy
    sample_rate = 44100  # 44.1 kHz standard audio sample rate
    num_samples = int((duration_ms / 1000) * sample_rate)
    samples = np.random.normal(0, 1, num_samples).astype(np.float32)
    
    # Convert the NumPy array to an AudioSegment object
    white_noise = AudioSegment(
        samples.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=4,  # 32-bit float (4 bytes)
        channels=1
    )
    
    # Adjust volume
    white_noise = white_noise + volume
    return white_noise

# Function to add background noise
def add_background_noise(audio_segment, noise_level=-20):
    # Generate white noise for the length of the audio
    noise = generate_white_noise(len(audio_segment), volume=noise_level)
    
    # Overlay the noise with the original audio
    return audio_segment.overlay(noise)

# Function to change the pitch of the audio
def change_pitch(audio_segment, semitones):
    # The pitch can be changed by speeding up/down the audio
    new_sample_rate = int(audio_segment.frame_rate * (2.0 ** (semitones / 12.0)))
    
    # Apply the new frame rate
    return audio_segment._spawn(audio_segment.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(44100)

# Folder containing your MP3 files
input_directory = r"C:\Users\Simon Risk\OneDrive\Desktop\speech_app\input\single_mp3s"
output_directory = r"C:\Users\Simon Risk\OneDrive\Desktop\speech_app\input\diverse_single_mp3s"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through each file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".mp3"):
        # Load the MP3 file
        filepath = os.path.join(input_directory, filename)
        audio = AudioSegment.from_mp3(filepath)
        
        # Add background noise
        audio_with_noise = add_background_noise(audio, noise_level=-15)  # Adjust noise level as needed
        
        # Change pitch (can be positive or negative for higher or lower pitch)
        pitch_shifted_audio = change_pitch(audio_with_noise, semitones=random.choice([-2, 2]))
        
        # Export the altered audio file
        output_filepath = os.path.join(output_directory, f"altered_{filename}")
        pitch_shifted_audio.export(output_filepath, format="mp3")

        print(f"Processed and saved: {output_filepath}")

print("All audio files processed and altered successfully!")
