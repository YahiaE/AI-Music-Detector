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
import numpy as np

total_tempo = 0
total_spectral_centroids = 0
total_ats = 0
mfccs_shapes = []


for i in range(
    1, 6
):  # go through the 5 files and load them through librosa => returns audio time series + sampling rate
    # audio time series = a sequence of data points representing sound, plotted as amplitude (loudness) against time
    y, sr = librosa.load(os.path.join(f"data\\ai\\ai_audio_{i}.wav"))
    total_ats += y
    """
    Onset Strength => How much a sound has changed compared to the moment before
    - Characteristics used to measure the change = Frequency, Amplitude, Tone (texture of the sound)
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # tempo measures how often a pattern of changes in the sound occur over time, which is why onset_strength is needed
    t = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]

    total_tempo += t

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
    total_spectral_centroids += s_cent

    """
    [Expand on this definition / meaning]
    MFCC (Mel-Frequency Cepstral Coefficients): Describes the timbre / shape of a sound. 
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs_shapes.append(mfccs.shape)

print(
    "\nAvergage amp over time:",
    (np.sum(total_ats) / total_ats.size),
    "\nSampling Rate:",
    sr,
    "\nTempo:",
    total_tempo / 6,
    "\nSpectral Centroid:",
    (np.sum(total_spectral_centroids) / total_spectral_centroids.size),
    "\nMFCC:",
    mfccs_shapes,
)


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
