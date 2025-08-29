import librosa
import librosa.display
import matplotlib.pyplot as plt

# Replace 'your_audio_file.m4a' with the actual path to your .m4a file
audio_path = 'audio.wav'

# Load the audio file
y, sr = librosa.load(audio_path)

# You can now perform various audio analyses with 'y' and 'sr'
# For example, to visualize the waveform:
plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform of the M4A file')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Or compute a Mel-spectrogram:
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_dB = librosa.power_to_db(S, ref=librosa.util.amin)

plt.figure(figsize=(12, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-spectrogram of the M4A file')
plt.tight_layout()
plt.show()