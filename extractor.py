import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

from audio_features import *

hop_length = 512 # no overelap
frame_size = 512 # analysis window of 23ms
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, 'audio_samples')
data = np.array([])

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    for filename in os.listdir(folder_path):
        label = folder.split('/')[-1]

        audio_path = os.path.join(folder_path, filename)
        y, sr = librosa.load(audio_path)
        S = librosa.stft(y, n_fft=frame_size, hop_length=hop_length)
        M = np.abs(S) # vetor de magnitudes
        f_bins = librosa.fft_frequencies(sr=sr, n_fft = frame_size) # vetor de frequencias

        features = np.array([])

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)

        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=5)
        s_centroid = spectral_centroid(M, f_bins)
        s_rolloff = spectral_rolloff(M, f_bins)
        s_flux = spectral_flux(M)
        zc = time_domain_zero_crossings(y, frame_size)
        low_energy_feature = low_energy(y, frame_size)

        for mfcc in mfccs:
            texture_mean, texture_var = texture_window(mfcc)
            features = np.append(features, texture_mean)
            features = np.append(features, texture_var)

        centroid_mean, centroid_var = texture_window(s_centroid)
        features = np.append(features, centroid_mean)
        features = np.append(features, centroid_var)

        rolloff_mean, rolloff_var = texture_window(s_rolloff)
        features = np.append(features, rolloff_mean)
        features = np.append(features, rolloff_var)

        flux_mean, flux_var = texture_window(s_flux)
        features = np.append(features, flux_mean)
        features = np.append(features, flux_var)

        zc_mean, zc_var = texture_window(zc)
        features = np.append(features, zc_mean)
        features = np.append(features, zc_var)

        features = np.append(features, low_energy_feature)

        if data.size == 0:
            data = features
        else:
            data = np.vstack((data, features))

normalized_data = normalize(data, axis=0)
np.savetxt('features.txt', normalized_data, fmt='%.12f')