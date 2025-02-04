import numpy as np


def mean_energy_concentration(data):
    energy = np.sum(np.square(data)) / len(data)
    return energy


def rms_amplitude(signal, sample_rate, frame_length_ms, frame_step_ms, windowing_function="hamming"):
    # Convert frame length and step from seconds to samples
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)

    # Calculate total number of frames
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length_samples)) / frame_step_samples))

    if windowing_function == "hamming":
        windowing_function = np.hamming(frame_length_samples)

    # Pad signal to make sure that all frames have equal number of samples
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    z = np.zeros((pad_signal_length - len(signal)))
    pad_signal = np.append(signal, z)

    # Initialize an array to hold RMS values for each frame
    rms_values = []

    # Loop over frames
    for i in range(0, len(pad_signal) - frame_length_samples, frame_step_samples):
        # Extract the current frame
        frame = pad_signal[i:i+frame_length_samples] * windowing_function

        # Calculate RMS amplitude for the current frame
        rms = np.sqrt(np.mean(frame**2))
        rms_values.append(rms)

    return np.array(rms_values)


def spl_per(data, fs, window_length_ms, window_step_ms, windowing_function="hamming"):
    # Convert to float if necessary
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # Calculate window length and step in samples
    window_length_samples = int(fs * window_length_ms / 1000)
    window_step_samples = int(fs * window_step_ms / 1000)


    if windowing_function == "hamming":
        windowing_function = np.hamming(window_length_samples)

    # Reference pressure in air (20 µPa)
    ref_pressure = 10e-6

    # Pad the end of the data array with zeros if necessary
    padding_size = window_length_samples - (len(data) % window_step_samples)
    if padding_size != window_length_samples:
        data = np.append(data, np.zeros(padding_size))

    # Initialize an array to store SPL values for each frame
    spl_values = []

    # Calculate SPL for each frame
    for start in range(0, len(data) - window_length_samples + 1, window_step_samples):
        frame = data[start:start + window_length_samples] * windowing_function
        rms = np.sqrt(np.mean(frame**2))
        if rms == 0:
            spl_values.append(np.nan)
        else:
            spl = 20 * np.log10(rms / ref_pressure)
            spl_values.append(spl)

    return np.array(spl_values)


def peak_amplitude(signal, sample_rate, frame_length_ms, frame_step_ms, windowing_function="hamming"):
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)
    num_frames = int(np.ceil(float(len(signal)) / frame_step_samples))

    if windowing_function == "hamming":
        windowing_function = np.hamming(frame_length_samples)

    # Pad the signal at the end to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    pad_signal = np.append(signal, np.zeros(pad_signal_length - len(signal)))

    peak_amplitudes = []

    for i in range(0, num_frames * frame_step_samples, frame_step_samples):
        frame = pad_signal[i:i + frame_length_samples] * windowing_function
        peak_amplitude = np.max(np.abs(frame))
        peak_amplitudes.append(peak_amplitude)

    return np.array(peak_amplitudes)


def ste_amplitude(signal, sample_rate, frame_length_ms, frame_step_ms, windowing_function="hamming"):
    # Convert frame length and step from seconds to samples
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)

    if windowing_function == "hamming":
        windowing_function = np.hamming(frame_length_samples)

    # Calculate total number of frames
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length_samples)) / frame_step_samples))

    # Pad signal to make sure that all frames have equal number of samples
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    z = np.zeros((pad_signal_length - len(signal)))
    pad_signal = np.append(signal, z)

    ste = np.array([np.sum(np.abs(pad_signal[i:i+frame_length_samples] * windowing_function)**2) for i in range(0, num_frames * frame_step_samples, frame_step_samples)])

    return ste


def intensity(signal, sample_rate, frame_length_ms, frame_step_ms, windowing_function="hamming", loudness=False):
    # Convert frame length and step from seconds to samples
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)

    # Calculate total number of frames
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length_samples)) / frame_step_samples))

    if windowing_function == "hamming":
        windowing_function = np.hamming(frame_length_samples)

    # Pad signal to make sure that all frames have equal number of samples
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    z = np.zeros((pad_signal_length - len(signal)))
    pad_signal = np.append(signal, z)
    # Normalize
    # pad_signal = pad_signal / np.max(pad_signal)

    # Initialize an array to hold RMS values for each frame
    intensity_values = []
    I0 = 10e-6

    # Loop over frames
    for i in range(0, len(pad_signal) - frame_length_samples, frame_step_samples):
        # Extract the current frame
        frame = pad_signal[i:i+frame_length_samples] * windowing_function

        # Calculate RMS amplitude for the current frame
        I = np.mean(frame**2)
        if loudness:
            I = (I/I0) ** (0.3)
        intensity_values.append(I)

    return np.array(intensity_values)