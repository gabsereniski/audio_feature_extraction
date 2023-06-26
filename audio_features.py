import numpy as np
import librosa

def spectral_centroid(magnitude, freq):
    centroids = np.zeros(magnitude.shape[1])
    
    for i in range(magnitude.shape[1]):
        num = np.sum(freq * magnitude[:, i])
        den = np.sum(magnitude[:, i])
        
        if den != 0:
            centroids[i] = num / den
        else:
            centroids[i] = 0
    
    return centroids

def spectral_rolloff(magnitude, freq, pcent = 0.85):
    n_frames = magnitude.shape[1]
    rolloff = np.zeros(n_frames)

    for i in range(n_frames):
        f_mag = magnitude[:, i]
        sum = np.sum(f_mag)
        accumulated_energy = 0.0

        for j, mag in enumerate(f_mag):
            accumulated_energy += mag
            if accumulated_energy >= pcent * sum:
                rolloff[i] = freq[j]
                break
    
    return rolloff

def spectral_flux(magnitude):
    normal_mag = librosa.amplitude_to_db(magnitude)
    n_frames = magnitude.shape[1]
    sf = np.zeros(n_frames)
    
    for i in range(n_frames):
        sf[i] = np.sum((normal_mag[:, i] - normal_mag[:, i-1])**2)
    return sf

def _sign(arg):
    if arg >= 0:
        return 1
    return 0

def time_domain_zero_crossings(signal, frame_size):
    sign_array_function = np.vectorize(_sign)
    sign_array = sign_array_function(signal)
    zt = np.zeros((len(signal) + frame_size - 1) // frame_size)
    
    for t in range(0, len(signal), frame_size):
        frame = signal[t : t + frame_size]
        diff = np.abs(np.diff(sign_array[t : t + frame_size]))
        crossings = np.sum(diff)
        zt[t // frame_size] = 0.5 * crossings
    return zt

def low_energy(signal, frame_size):
    rms = np.zeros((len(signal) + frame_size - 1) // frame_size)
    
    for t in range(0, len(signal), frame_size):
        frame = signal[t : t + frame_size]
        square = np.square(frame)
        mean_square = np.mean(square)
        rms[t // frame_size] = np.sqrt(mean_square)
    
    avg_rms = np.mean(rms)
    low_energy_count = np.sum(rms < avg_rms)
    total_windows = len(rms)
    low_energy_percentage = low_energy_count / total_windows * 100
    
    return low_energy_percentage

def texture_window(analysis_windows, n=43):
    
    w = np.lib.stride_tricks.sliding_window_view(analysis_windows, window_shape=(n,))
    means = np.mean(w, axis=1)
    variances = np.var(w, axis=1)
    
    return np.mean(means), np.var(variances)