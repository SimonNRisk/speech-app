from pydub import AudioSegment

# Load the audio file
audio = AudioSegment.from_mp3(r"C:\Users\Simon Risk\OneDrive\Desktop\speech_app\input\p_30779624_145.mp3")

# Duration of each "Three" section in milliseconds
# Assuming you want equal intervals (for simplicity, this is approximate)
duration_per_word = len(audio) // 60

# Path to save the split files
output_directory = r"C:\Users\Simon Risk\OneDrive\Desktop\speech_app\input\single_mp3s"

# Split and export 60 segments
for i in range(60):
    # Define start and end of each segment
    start_time = i * duration_per_word
    end_time = start_time + duration_per_word
    
    # Extract the segment
    segment = audio[start_time:end_time]
    
    # Export the segment as an MP3 file
    segment.export(f"{output_directory}/three_{i+1}.mp3", format="mp3")

print("Audio segments successfully saved.")
