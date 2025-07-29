import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load audio
audio_path = "L.zwo - House On Fire.wav"
y, sr = librosa.load(audio_path, sr=None)
hop_length = 512
frame_length = 2048

# Compute RMS energy
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
frames = range(len(rms))
times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

# Compute spectral flux
S = np.abs(librosa.stft(y, hop_length=hop_length))
flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
flux = np.pad(flux, (1, 0))

# Compute Tempo
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f"Estimated tempo: {tempo[0]:.2f} BPM")

# Plot energy & Flux
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
plt.plot(times, rms, label="RMS energy", color="red")
plt.title("Track Energy Curve (RMS)")
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(times, flux[:len(times)], label= "Spectral Flux", color="blue")
plt.title("Spectral Flux Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Flux")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()