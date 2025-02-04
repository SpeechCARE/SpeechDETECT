import numpy as np
from scipy.signal import welch
from itertools import permutations

from helper import frame_signal

# Fractal Dimension
def hfd(audio_frame, k_max):
    black_eps = 1e-20
    L = []
    x = np.asarray(audio_frame)
    N = x.size
    for k in range(1, k_max):
        Lk = 0
        for m in range(k):
            idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=np.int32)
            Lmk = np.sum(np.abs(x[m + idxs * k] - x[m + (idxs - 1) * k])) / len(idxs) / k
            Lk += Lmk / k
        L.append(np.log(Lk / (m + 1)+black_eps))
    L = np.array(L)
    (p, r1, r2, s) = np.linalg.lstsq(np.vstack([np.log(np.arange(1, k_max)), np.ones(k_max - 1)]).T, L, rcond=None)
    return p[0]

def calculate_hfd_per_frame(data, fs, window_length_ms, window_step_ms, k_max, windowing_function="hamming"):
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Normalize data
    # data = data / np.max(np.abs(data))

    # Calculate window length and step in samples
    window_length_samples = int(fs * window_length_ms / 1000)
    window_step_samples = int(fs * window_step_ms / 1000)

    if windowing_function == "hamming":
        windowing_function = np.hamming(window_length_samples)

    # Pad the end of the data array with zeros if necessary
    padding_size = window_length_samples - (len(data) % window_step_samples)
    if padding_size != window_length_samples:
        data = np.append(data, np.zeros(padding_size))

    # Initialize an array to store HFD values for each frame
    hfd_values = []

    # Calculate HFD for each frame
    for start in range(0, len(data) - window_length_samples + 1, window_step_samples):
        frame = data[start:start + window_length_samples] * windowing_function
        hfd_value = hfd(frame, k_max)
        hfd_values.append(hfd_value)

    return np.array(hfd_values)


# Shannon Entropy
def calculate_frequency_entropy(signal, sample_rate, frame_length_ms, frame_step_ms, windowing_function="hamming"):
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)
    num_frames = int(np.ceil(float(len(signal)) / frame_step_samples))
    if windowing_function == "hamming":
        windowing_function = np.hamming(frame_length_samples)

    # Pad the signal at the end to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    pad_signal = np.append(signal, np.zeros(pad_signal_length - len(signal)))

    entropy_values = []

    # Calculate entropy for each frame
    for i in range(0, len(pad_signal) - frame_length_samples + 1, frame_step_samples):
        frame = pad_signal[i:i + frame_length_samples] * windowing_function


        # Calculate the power spectral density of the signal
        frequencies, power_spectrum = welch(frame, fs=sample_rate)

        # Convert power spectrum to probability distribution
        power_spectrum_prob_dist = power_spectrum / (np.sum(power_spectrum) + 1)

        # Calculate the Shannon entropy
        entropy = -np.sum(power_spectrum_prob_dist * np.log2(power_spectrum_prob_dist + np.finfo(float).eps))  # Adding epsilon to avoid log(0)
        entropy_values.append(entropy)

    return np.array(entropy_values)

# Shannon Entropy
def calculate_amplitude_entropy(signal, sample_rate, frame_length_ms, frame_step_ms, windowing_function="hamming"):
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)
    num_frames = int(np.ceil(float(len(signal)) / frame_step_samples))
    if windowing_function == "hamming":
        windowing_function = np.hamming(frame_length_samples)


    # Pad the signal at the end to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    pad_signal = np.append(signal, np.zeros(pad_signal_length - len(signal)))

    entropy_values = []

    # Calculate entropy for each frame
    for i in range(0, len(pad_signal) - frame_length_samples + 1, frame_step_samples):
        frame = pad_signal[i:i + frame_length_samples] * windowing_function

        # Normalize frame values to form a probability distribution
        histogram, _ = np.histogram(frame, bins=256, range=(np.min(frame), np.max(frame)), density=True)
        probability_distribution = histogram / np.sum(histogram)

        # Remove zero entries for log calculation
        probability_distribution = probability_distribution[probability_distribution > 0]

        # Calculate Shannon entropy
        entropy = -np.sum(probability_distribution * np.log2(probability_distribution))
        entropy_values.append(entropy)

    return np.array(entropy_values)


# Multi Sclae permutation entropy
def permutation_entropy(time_series, m, tau):
    n = len(time_series)
    permutations_dict = {p: 0 for p in permutations(range(m))}

    for i in range(n - tau * (m - 1)):
        # Reconstruct phase space
        sorted_index_tuple = tuple(np.argsort(time_series[i:i + tau * m:tau]))
        permutations_dict[sorted_index_tuple] += 1

    # Calculate the probabilities of each permutation
    permutation_counts = np.array(list(permutations_dict.values()))
    permutation_probabilities = permutation_counts / sum(permutation_counts)

    # Calculate the Shannon entropy
    entropy = -np.sum(permutation_probabilities * np.log2(permutation_probabilities + np.finfo(float).eps))

    return entropy


# Function to coarse-grain the time series
def coarse_grain(time_series, scale):
    n = len(time_series)
    # Calculate the number of coarse-grained elements
    num_elements = n // scale
    # Initialize the coarse-grained time series
    coarse_grained_series = np.zeros(num_elements)
    # Calculate the average for each non-overlapping window of length 'scale'
    for i in range(num_elements):
        coarse_grained_series[i] = np.mean(time_series[i * scale : (i + 1) * scale])
    return coarse_grained_series


# Function to calculate multiscale permutation entropy
def multiscale_permutation_entropy_helper(time_series, m, tau, scales):
    mpe = []
    for s in scales:
        # Coarse-grain the time series at scale 's'
        cg_time_series = coarse_grain(time_series, s)
        # Calculate the permutation entropy at this scale
        pe = permutation_entropy(cg_time_series, m, tau)
        mpe.append(pe)
    return mpe


def multiscale_permutation_entropy(data, fs, window_length_ms, window_step_ms, m, tau, scales, windowing_function="hamming"):
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)

    entropies = np.zeros((scales, frames.shape[0]))
    for i, frame in enumerate(frames):
        entropies[:, i] = multiscale_permutation_entropy_helper(frame, m, tau, scales)


