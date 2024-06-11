import os

def remove_old_spectrograms(folder_path, keyword):
    """
    Removes files from the specified folder that do not contain the specified keyword in their name.
    
    :param folder_path: Path to the folder containing spectrograms.
    :param keyword: Keyword that should be present in the filenames to keep them.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') and keyword not in filename:
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Removed {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

# Specify paths for "tree" and "three" spectrogram folders
three_spectrogram_folder = '/Users/simonrisk/Desktop/speech_therapy_1/spectrograms/three'
tree_spectrogram_folder = '/Users/simonrisk/Desktop/speech_therapy_1/spectrograms/tree'

# Remove old spectrograms that do not have "tree" in their name
remove_old_spectrograms(three_spectrogram_folder, 'tree')
remove_old_spectrograms(tree_spectrogram_folder, 'three')
