import numpy as np
import librosa
import parselmouth
from scipy.signal import find_peaks

def calculate_APQ_from_peaks(data, fs, window_length_ms, window_step_ms, num_cycles=3):
    """
    Calculate the frame-based Amplitude Perturbation Quotient (APQ) using peak amplitudes.

    Parameters:
    frames (np.array): 2D array where each row represents a frame of the voice signal.
    num_cycles (int): The number of cycles over which to average the APQ within each frame.

    Returns:
    np.array: The calculated APQ values for each frame.
    """
    frame_length = int(window_length_ms * fs / 1000)
    hop_length = int(window_step_ms * fs / 1000)

    frames = librosa.util.frame(data, frame_length=frame_length, hop_length=hop_length)

    apq_values = []

    for frame in frames.T:
        # Find peaks in the frame
        peaks, _ = find_peaks(frame)
        peak_amplitudes = frame[peaks]

        # If there are not enough peaks to calculate APQ, skip this frame
        if len(peak_amplitudes) < num_cycles + 1:
            apq_values.append(0)
            continue

        # Calculate the absolute differences in amplitude between consecutive peaks
        amplitude_differences = np.abs(np.diff(peak_amplitudes))

        # Compute the average difference over num_cycles for the frame
        frame_apq_values = []
        for i in range(len(amplitude_differences) - num_cycles + 1):
            avg_diff = np.mean(amplitude_differences[i:i + num_cycles])
            avg_amp = np.mean(np.abs(peak_amplitudes[i:i + num_cycles]))
            apq = avg_diff / avg_amp if avg_amp != 0 else 0
            frame_apq_values.append(apq)

        # Calculate mean APQ for the frame and append to the list
        frame_apq_mean = np.mean(frame_apq_values) * 100 if frame_apq_values else 0  # Convert to percentage
        apq_values.append(frame_apq_mean)

    return np.array(apq_values)


def calculate_frame_based_APQ(data, fs, window_length_ms, window_step_ms, num_samples=3):
    """
    Calculate the frame-based Amplitude Perturbation Quotient (APQ).

    Parameters:
    frames (np.array): 2D array where each row represents a frame of the voice signal.
    num_samples (int): The number of samples over which to average the APQ within each frame.

    Returns:
    np.array: The calculated APQ values for each frame.
    """
    frame_length = int(window_length_ms * fs / 1000)
    hop_length = int(window_step_ms * fs / 1000)

    frames = librosa.util.frame(data, frame_length=frame_length, hop_length=hop_length)
    apq_values = []
    for frame in frames.T:
        # Ensure there are enough samples in the frame to calculate APQ
        if len(frame) < num_samples:
            raise ValueError(f"Frame too short to calculate APQ over {num_samples} samples.")

        # Calculate the absolute differences in amplitude between consecutive samples
        amplitude_differences = np.abs(np.diff(frame))

        # Initialize list to store APQ values for this frame
        frame_apq_values = []

        # Compute the average difference over num_samples for the frame
        for i in range(len(amplitude_differences) - num_samples + 1):
            local_diffs = amplitude_differences[i:i+num_samples]
            local_amps = frame[i:i+num_samples]
            local_diff_avg = np.mean(local_diffs)
            local_amp_avg = np.mean(np.abs(local_amps))
            apq = local_diff_avg / local_amp_avg if local_amp_avg != 0 else 0
            frame_apq_values.append(apq)

        # Calculate mean APQ for the frame and append to the list
        frame_apq_mean = np.mean(frame_apq_values) if frame_apq_values else 0
        apq_values.append(frame_apq_mean * 100)  # Convert to percentage

    return np.array(apq_values)

def shimmer(frame):
    # Calculate the amplitude differences between consecutive samples
    amplitude_differences = np.abs(np.diff(frame))

    # Avoid division by zero
    if np.all(frame[:-1] == 0):
        return 0

    # Compute shimmer
    shimmer = np.mean(amplitude_differences) / np.mean(np.abs(frame[:-1]))

    return shimmer

def analyze_audio_shimmer(data, fs, frame_length_ms=40, hop_length_ms=20):

    frame_length = int(frame_length_ms * fs / 1000)
    hop_length = int(hop_length_ms * fs / 1000)

    # Frame the signal
    frames = librosa.util.frame(data, frame_length=frame_length, hop_length=hop_length)

    # Calculate shimmer for each frame
    shimmer_values = [shimmer(frame) for frame in frames.T]

    return np.array(shimmer_values)


def calculate_frame_level_hnr(signal, sample_rate, frame_length_ms, frame_step_ms, windowing_function="hamming"):
    frame_length = int(frame_length_ms * sample_rate / 1000)  # Convert ms to samples
    frame_step = int(frame_step_ms * sample_rate / 1000)      # Convert ms to samples
    num_frames = int(np.ceil(float(len(signal) - frame_length) / frame_step))

    if windowing_function == "hamming":
        windowing_function = np.hamming(frame_length)

    # Pad the signal to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step + frame_length
    pad_signal = np.append(signal, np.zeros(pad_signal_length - len(signal)))

    hnr_values = []

    for i in range(0, len(pad_signal) - frame_length + 1, frame_step):
        frame = pad_signal[i:i + frame_length] * windowing_function

        snd_frame = parselmouth.Sound(frame, sampling_frequency=sample_rate)
        hnr = snd_frame.to_harmonicity()
        # Remove NaN values and avoid division by zero
        valid_hnr = hnr.values[~np.isnan(hnr.values)]
        # valid_hnr = valid_hnr[valid_hnr > 0]
        if len(valid_hnr) > 0:
            hnr_values.append(np.mean(valid_hnr))
        else:
            hnr_values.append(np.nan)  # Assign zero if no valid HNR values
    # NHR can be calculated as reciprocal of HNR
    nhr_values = [1/x if x != 0 else float('inf') for x in hnr_values]

    return np.array(hnr_values), np.array(nhr_values)
