from helper import frame_signal
from scipy.signal import get_window

import parselmouth
import numpy as np
import librosa

def get_pitch(data, fs, frame_duration_ms=40, step_size_ms=10):

    pitch_frames = []
    frames = frame_signal(data, fs, frame_duration_ms, step_size_ms)
    for i, frame in enumerate(frames):
        snd = parselmouth.Sound(frame, fs)
        # Extract pitch
        pitch = snd.to_pitch()
        f0 = pitch.selected_array['frequency']
        f0 = f0[f0 > 0]  # Remove unvoiced frames
        if len(f0) > 0:
            estimated_f0 = np.mean(f0)  # Get a stable estimate of F0
            pitch_frames.append(estimated_f0)

        else:
            pitch_frames.append(np.nan)

    return np.array(pitch_frames)



def f0_estimation(signal, sample_rate, frame_length_ms, frame_step_ms):
    frame_length_samples = int(frame_length_ms * sample_rate / 1000)
    frame_step_samples = int(frame_step_ms * sample_rate / 1000)
    window = get_window('hann', frame_length_samples)
    f0_pyin, voiced_flag, voiced_probs = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate,
                                                      frame_length=frame_length_samples, hop_length=frame_step_samples)

    return f0_pyin


def calculate_time_varying_jitter(f0, fs, pyin_frame_length_ms=100, pyin_hop_length_ms=25, jitter_frame_length_ms=250, jitter_hop_length_ms=125):

    pyin_frame_length = int(pyin_frame_length_ms * fs / 1000)
    pyin_hop_length = int(pyin_hop_length_ms * fs / 1000)
    jitter_frame_length = int(jitter_frame_length_ms * fs / 1000)
    jitter_hop_length = int(jitter_hop_length_ms * fs / 1000)


    # Extract the F0 contour for the entire audio file with pyin_frame_length and pyin_hop_length
    # f0 =  get_pitch(data, fs, frame_duration_ms=pyin_frame_length, step_size_ms=pyin_hop_length)
    # # Time-varying jitter calculation
    # f0 = np.array(f0)
    jitter_values = []

    # Calculate jitter for each frame with jitter_frame_length and jitter_hop_length
    for i in range(0, len(f0) - (jitter_frame_length // jitter_hop_length), jitter_hop_length // pyin_hop_length):
        frame_f0 = f0[i:i + (jitter_frame_length // pyin_hop_length)]

        # Remove unvoiced regions and zero values
        frame_f0 = frame_f0[frame_f0 > 0]

        if len(frame_f0) > 1:
            # Calculate differences between consecutive F0 values
            f0_diffs = np.abs(np.diff(frame_f0))

            # Calculate jitter for the frame
            frame_jitter = np.mean(f0_diffs) / np.mean(frame_f0)
            jitter_values.append(frame_jitter)
        else:
            # Append NaN or zero if the frame has insufficient data
            jitter_values.append(np.nan)

    return jitter_values

def get_formants_frame_based(data, fs, frame_duration_ms=40, step_size_ms=10, formant_number=1):

    frames = frame_signal(data, fs, frame_duration_ms, step_size_ms)
    res = np.zeros((3, frames.shape[0]))

    for i, frame in enumerate(frames):
        snd = parselmouth.Sound(frame, fs)

        # Estimate formants using Parselmouth
        formants = snd.to_formant_burg(max_number_of_formants=5)
        tmp = []
        for n in formant_number:
            f = formants.get_value_at_time(n, snd.duration / 2)  # Get F3 frequency at the midpoint
            tmp.append(f)
        res[:, i] = np.array(tmp)
    return res