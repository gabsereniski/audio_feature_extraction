import os
import librosa
import numpy as np
from audio_features import *
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()
print("Número de núcleos do processador:", num_cores)

hop_length = 512  # no overlap
frame_size = 512  # analysis window of 23ms
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, 'audio_samples')
data = np.array([])

print("running...")


def process_audio_file(file_path):
    label = os.path.basename(os.path.dirname(file_path))
    
    y, sr = librosa.load(file_path)
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
    features = np.append(features, str(label))  # Convert label to string

    return features


def process_folder(folder_path):
    folder_features = []
    for filename in os.listdir(folder_path):
        audio_path = os.path.join(folder_path, filename)
        features = process_audio_file(audio_path)
        folder_features.append(features)
    print(f"finished {folder_path}")
    return folder_features

# Collect the results from the worker processes
results = Parallel(n_jobs=num_cores)(delayed(process_folder)(os.path.join(root_dir, folder)) for folder in os.listdir(root_dir))

# Flatten the results list
folder_features = [item for sublist in results for item in sublist]

# Convert the features list to numpy array
data = np.array(folder_features)

# Check if data is 1-dimensional
if data.ndim == 1:
    features = data  # Just an array of features
else:
    features = data[:, :-1]  # All columns except the last
    labels = data[:, -1]  # Last column

# Normalize only the features
normalized_features = normalize(features, axis=0)

# Combine the normalized data with the label column
normalized_data = np.column_stack((normalized_features, labels))

np.savetxt('features.txt', normalized_data, fmt='%s')