'''
Libraries: 
- librosa: audio/music analysis (converts audio to numbers)
- numpy: handles arrays of numbers from music (music can contains millions of data points, numpy utilizes arrays that are much faster than
py arrays)
- pandas: data organizer for data points
- scikit-learn: used to train models
- matplotlib: plot graphs to show accuracy + used for debugging
- jupyter: splits code into runable individual chunks (good for learning / debugging)
'''
import librosa
import os.path

for i in range(1,6): # go through the 5 files and load them through librosa => returns audio time series + sampling rate
    # audio time series = a sequence of data points representing sound, plotted as amplitude (loudness) against time
    y, sr = librosa.load(os.path.join(f'data\\ai\\ai_audio_{i}.wav'))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr) # detects how strong of a change of frequency has occurred throughout the sound
    # tempo measures how frequent sudden changes of frequency happen, which is why onset_strength is needed
    t = librosa.feature.tempo(onset_envelope=onset_env,sr=sr) 
    print("\nAmp over time:", y, "\nSampling Rate:", sr, "\nTempo:", t)


