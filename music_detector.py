"""
Libraries:
- librosa: audio/music analysis (converts audio to numbers)
- numpy: handles arrays of numbers from music (music can contains millions of data points, numpy utilizes arrays that are much faster than
py arrays)
- pandas: data organizer for data points
- scikit-learn: used to train models
- matplotlib: plot graphs to show accuracy + used for debugging
- jupyter: splits code into runable individual chunks (good for learning / debugging)
"""

import librosa
import os.path
import math
import numpy as np
import matplotlib.pyplot as plt

total_tempo_ai = 0
total_spectral_centroids_ai = 0
total_ats_ai = 0
mfcc_shapes_ai = []
sample_rate = 0


for i in range(
    1, 6
):  # go through the 5 files and load them through librosa => returns audio time series + sampling rate
    # audio time series = a sequence of data points representing sound, plotted as amplitude (loudness) against time
    y, sr = librosa.load(os.path.join(f"data\\ai\\ai_audio_{i}.wav"))
    sample_rate = sr
    total_ats_ai += y

    secs = y.size // sample_rate

    avg_abs_amp = np.sum(np.abs(y))/y.size
    print(f"Average Absolute Amplitude For AI Song #{i}: {avg_abs_amp}")
    """
    Onset Strength => How much a sound has changed compared to the moment before
    - Characteristics used to measure the change = Frequency, Amplitude, Tone (texture of the sound)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # tempo measures how often a pattern of changes in the sound occur over time, which is why onset_strength is needed
    t = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
    print(f"Tempo For AI Song #{i}: {t}")

    total_tempo_ai += t

    """
    Spectral Centroids: Shows which frequencies have the highest amplitudes
    - Low frequencies have the highest amplitudes => Sound is rather dark / bassy
    - High frequencies have the highest amplitudes => Sound is bright / sharp
     
    Uses the weighted sum of all frequencies multiplied by their amplitudes divided by the total amplitude
    - Result is measured in Hertz (Hz) (Lower => Darker, Higher => Brighter)

    | Frequency (Hz) | Amplitude |
    | -------------- | --------- |
    | 100            | 1         |
    | 500            | 2         |
    | 1000           | 5         | => High freq, high amp

    | Frequency (Hz) | Amplitude |
    | -------------- | --------- |
    | 100            | 5         | => Low freq, high amp
    | 500            | 2         |
    | 1000           | 1         | 
    """
    s_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    avg_s_cent = math.floor(np.sum(s_cent) / s_cent.size)
    print(f"Average Spectral Centroid For AI Song #{i}: {avg_s_cent}")
    
    total_spectral_centroids_ai += s_cent

    """
    [Expand on this definition / meaning]
    MFCC (Mel-Frequency Cepstral Coefficients): Describes the timbre / shape of a sound. 
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_shapes_ai.append(mfcc.shape)
    print(f"MFCC For AI Song #{i}: {mfcc.shape}\n")

print(
    "\nAverage absolute amp:",
    (np.sum(np.abs(total_ats_ai)) / total_ats_ai.size),
    "\nSampling Rate:",
    sr,
    "\nTempo:",
    math.floor(total_tempo_ai / 6),
    "\nSpectral Centroid:",
    math.floor((np.sum(total_spectral_centroids_ai) / total_spectral_centroids_ai.size)),
    "\nMFCC:",
    mfcc_shapes_ai,
)

print(total_ats_ai.size)
fig, ax = plt.subplots()
# Sample = Measurement of amplitude at a very specific point in time. The # of samples is dependent on the sampling rate (# of samples taken per second)
time = (
    total_ats_ai.size // sample_rate
)  # Seconds = total # of samples / sample_rate (# of samples per second) => equals to 30 seconds

# Plots average absolute amplitude per second
ax.plot(
    np.arange(time),
    [
        np.mean(np.abs(total_ats_ai[i * sr : (i + 1) * sr])) for i in range(time)
    ],  # Y-Axis (average of absolute amplitude per sample group (dependent on sample rate)). Absolute amplitude is more accurate to represent loudness + Positive/Negative amplitudes can cancel out, which can make the graph misleading
)
plt.show()


# for i in range(1, 6):
#     y, sr = librosa.load(os.path.join(f"data\\human\\human_audio_{i}.wav"))

#     onset_env = librosa.onset.onset_strength(y=y, sr=sr)
#     t = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
#     s_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr)

#     print(
#         "\nAmp over time:",
#         y,
#         "\nSampling Rate:",
#         sr,
#         "\nTempo:",
#         t,
#         "\nSpectral Centroid:",
#         s_cent,
#         "\nMFCC:",
#         mfccs,
#     )
