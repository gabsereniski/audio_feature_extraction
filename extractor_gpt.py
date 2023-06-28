import os
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from audio_features import *

hop_length = 512  # no overlap
frame_size = 512  # analysis window of 23ms
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, 'audio_samples')
data = np.array([])

print("running...")


def process_audio(folder_path, filename):
    label = folder_path.split('/')[-1]
    
    audio_path = os.path.join(folder_path, filename)
    y, sr = librosa.load(audio_path)
    S = librosa.stft(y, n_fft=frame_size, hop_length=hop_length)
    M = np.abs(S)  # vetor de magnitudes
    f_bins = librosa.fft_frequencies(sr=sr, n_fft=frame_size)  # vetor de frequencias
    
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
    features = np.append(features, label)

    return features


def process_folder(folder):
    folder_path = os.path.join(root_dir, folder)
    features_list = []

    for filename in os.listdir(folder_path):
        features = process_audio(folder_path, filename)
        features_list.append(features)

    print(f"finished {folder}")
    return features_list


folders = os.listdir(root_dir)

# Parallelize the execution of process_folder using joblib
all_features = joblib.Parallel(n_jobs=-1)(joblib.delayed(process_folder)(folder) for folder in folders)

for folder_result in all_features:
    data = np.vstack((data, np.vstack(folder_result))) if data.size != 0 else np.vstack(folder_result)

features = data[:, :-1]
labels = data[:, -1]

normalized_features = normalize(features, axis=0)
normalized_data = np.column_stack((normalized_features, labels))

np.savetxt('features.txt', normalized_data, fmt=['%.12f'] * (normalized_data.shape[1]-1) + ['%s'], delimiter=' ')
