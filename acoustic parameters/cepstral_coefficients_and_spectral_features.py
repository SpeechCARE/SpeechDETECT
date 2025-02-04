import numpy as np
from helper import *
from scipy.fftpack import dct
from scipy.signal import find_peaks, periodogram
import parselmouth
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import scipy.fft
from scipy.linalg import solve_toeplitz
import numpy as np
import math
import numpy as np
import scipy
from spafe.fbanks import mel_fbanks, bark_fbanks
import librosa
from scipy import linalg

def amplitude_range(data, fs, window_length_ms, window_step_ms, windowing_function="hamming"):
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function)

    # Calculate amplitude range and standard deviation for each frame
    ranges = np.ptp(frames, axis=1)  # Peak-to-peak calculation along each frame
    std = np.std(frames, axis=1)  # Standard deviation along each frame

    return ranges, std


def compute_msc(signal, sample_rate, nfft=512, window_length_ms=90, window_step_ms=25, num_msc=13):
    # Frame blocking
    frames = frame_signal(signal, sample_rate, window_length_ms, window_step_ms)
    # FFT
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))  # Magnitude of the FFT
    pow_frames = ((1.0 / nfft) * ((mag_frames) ** 2))  # Power Spectrum
    # Calculate modulation spectrum for each frequency band (simplified approach)
    modulation_spectra = np.abs(np.fft.rfft(pow_frames, nfft, axis=0))
    # DCT to get MSC (using the same method as for MFCCs but on the modulation spectrum)
    msc = dct(modulation_spectra, type=2, axis=0, norm='ortho')[:num_msc]
    return msc  # Transpose so that rows are frames and columns are MSC coefficients


def calculate_centriod(x, fs):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/fs)[:length//2+1]) # positive frequencies
    if np.sum(magnitudes) == 0:
        return np.nan
    else:
        return np.sum(magnitudes*freqs) / np.sum(magnitudes)
    
def spectral_centriod(data, fs, window_length_ms, window_step_ms, windowing_function="hamming"):
    window_length = int(window_length_ms*fs/1000)
    window_step = int(window_step_ms*fs/1000)
    if windowing_function == "hamming":
        windowing_function = np.hamming(window_length)


    total_length = len(data)
    window_count = int((total_length-window_length)/window_step) + 1

    # Pad Signal
    pad_signal_length = window_count * window_step + window_length
    z = np.zeros((pad_signal_length - total_length))
    padded_signal = np.append(data, z)

    centriods  = []
    for k in range(window_count):
        starting_position = k*window_step

        data_vector = padded_signal[starting_position:(starting_position+window_length),] * windowing_function
        centriods.append(calculate_centriod(data_vector, fs))
    return centriods



def ltas(x, fs, window_length_ms, window_step_ms, units='db', graph=False):
    """Calculates the long-term average spectrum (LTAS) of a signal.

    Args:
      x: A NumPy array representing the input signal.
      fs: The sampling frequency of the signal in Hz.
      win: The length of the window used to calculate the STFT. Defaults to 4096.
      hop: The step size between windows. Defaults to half the window length.
      units: The units to use for the LTAS. Can be `db` or `none`. Defaults to `db`.
      graph: Whether to plot the LTAS. Defaults to `False`.

    Returns:
      A NumPy array representing the LTAS of the input signal and frequencies
    """

    win = int(window_length_ms / 1000 * fs)
    hop = int(window_step_ms / 1000 * fs)


    # Check input arguments.
    if hop is None:
        hop =   win // 2

    # Calculate the STFT.
    f, t, S = scipy.signal.stft(x, fs, window='hann', nperseg=win, noverlap=hop)

    # Calculate the power spectral density (PSD).
    PSD = np.abs(S)**2

    # Calculate the LTAS.
    ltas = np.mean(PSD, axis=1)

    # Convert to dB if specified.
    if units == 'db':
        ltas = 10 * np.log10(ltas)

    # Plot the LTAS if specified.
    if graph:
        import matplotlib.pyplot as plt
        plt.plot(f, ltas)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (dB)')
        plt.title('Long-term average spectrum')
        plt.show()

    return ltas, f

def harmonic_difference(signal, sample_rate, window_length_ms, window_step_ms, n_first_formant, n_second_fromant):
    # Estimate F0 using Parselmouth
    frames = frame_signal(signal, sample_rate, window_length_ms, window_step_ms)
    differences = np.zeros(len(frames))
    for i, frame in enumerate(frames):
        snd = parselmouth.Sound(frame, sample_rate)
        pitch = snd.to_pitch()
        f0 = pitch.selected_array['frequency']
        f0 = f0[f0 > 0]  # Remove unvoiced frames
        estimated_f0 = np.mean(f0)  # Get a stable estimate of F0
        if math.isnan(estimated_f0):
            differences[i] = np.nan
            continue


        # Estimate formants using Parselmouth
        formants = snd.to_formant_burg(max_number_of_formants=5)
        fn_freq = formants.get_value_at_time(n_second_fromant, snd.duration / 2)  # Get F3 frequency at the midpoint

        # Calculate the spectrum
        frequencies, spectrum = periodogram(signal, sample_rate)

        # Find harmonic peaks
        harmonic_freqs = estimated_f0 * np.arange(1, int(sample_rate / (2 * estimated_f0)) + 1)
        harmonic_amplitudes = np.interp(harmonic_freqs, frequencies, spectrum)

        # Find the peak harmonic amplitude in the vicinity of the third formant
        harmonic_fn_range = harmonic_amplitudes[(harmonic_freqs >= fn_freq - estimated_f0) & (harmonic_freqs <= fn_freq + estimated_f0)]
        An_amplitude = np.max(harmonic_fn_range) if len(harmonic_fn_range) > 0 else 0

        # Get the amplitude of H1
        H1_amplitude = harmonic_amplitudes[n_first_formant]

        # Calculate H1-A3 in dB
        H1_An_dB = 20 * np.log10(H1_amplitude) - 20 * np.log10(An_amplitude)

        differences[i] =  H1_An_dB
    return differences


def alpha_ratio(data, fs, window_length_ms, window_step_ms, lower_band, higher_band):

    window_length_samples = int(window_length_ms*fs/1000)
    window_step_samples = int(window_step_ms*fs/1000)

    f, lt = scipy.signal.welch(data,fs, nperseg=window_length_samples, noverlap=window_step_samples)

    idx = np.nonzero((f > lower_band[0]) & (f < lower_band[1]))
    frequency_vector = f[idx]
    spectrum = lt[idx]

    low_frequency_energy = np.sum(spectrum)

    idx = np.nonzero((f > higher_band[0]) & (f <= higher_band[1]))
    frequency_vector = f[idx]
    spectrum = lt[idx]

    high_frequency_energy = np.sum(spectrum)

    return (high_frequency_energy / low_frequency_energy)

def log_mel_spectrogram(data, fs, window_length_ms, window_step_ms, melbands=120, fmin=20, fmax=8000, window="hamming"):

    spectrogram = stft(data,fs,window_length_ms=window_length_ms,window_step_ms=window_step_ms, windowing_function=window)
    nyquist_frequency = fs // 2

    fmin_bin = int(np.floor((spectrogram.shape[1] / nyquist_frequency) * fmin))
    fmax_bin = int(np.ceil((spectrogram.shape[1] / nyquist_frequency) * fmax))

    # Adjust Mel scale range
    min_mel = freq2mel(fmin)
    max_mel = freq2mel(fmax)
    mel_idx = np.linspace(min_mel, max_mel, melbands)

    freq_idx = mel2freq(mel_idx)

    n_points = fmax_bin - fmin_bin + 1

    # melfilterbank = np.zeros((spectrogram.shape[1],melbands))
    # freqvec = np.linspace(fmin, fmax, n_points)
    # for k in range(melbands-2):
    #     if k>0:
    #         upslope = (freqvec-freq_idx[k])/(freq_idx[k+1]-freq_idx[k])
    #     else:
    #         upslope = 1 + 0*freqvec
    #     if k<melbands-3:
    #         downslope = 1 - (freqvec-freq_idx[k+1])/(freq_idx[k+2]-freq_idx[k+1])
    #     else:
    #         downslope = 1 + 0*freqvec
    #     triangle = np.max([0*freqvec,np.min([upslope,downslope],axis=0)],axis=0)
    #     melfilterbank[fmin_bin:fmax_bin+1,k] = triangle

    nfft = 2048
    melfilterbank, mel_freqs = mel_fbanks.mel_filter_banks(nfilts=26,
                                                 nfft=spectrogram.shape[1] * 2 - 1,
                                                 fs=fs,
                                                 low_freq = fmin,
                                                 high_freq= fmax)
    # print(spectrogram.shape)
    # print(melfilterbank.shape)

    logmelspectrogram = ((1 / 1) * np.matmul(np.abs(spectrogram)**2,melfilterbank.T)+1e-12)
    return logmelspectrogram.T


def mfcc(data, fs, window_length_ms, window_step_ms, melbands=120, fmin=20, fmax=8000, lifter=0, window="hamming"):
    # print(melbands)
    logmelspectrogram = log_mel_spectrogram(data, fs, window_length_ms, window_step_ms, melbands, fmin, fmax, window)
    logmelspectrogram = np.log(logmelspectrogram)
    mfcc = scipy.fft.dct(logmelspectrogram.T).T
    if lifter > 0:
        mfcc = lifter_ceps(mfcc.T, lifter).astype(float).T

    return mfcc
def lpc(data, fs, window_length_ms, window_step_ms, lpc_length=0, windowing_function="hamming"):

    if lpc_length == 0:
        lpc_length = int(1.25*fs/1000)


    # Convert frame length and step from seconds to samples
    window_length = int(window_length_ms * fs / 1000)
    window_step = int(window_step_ms * fs / 1000)

    # Calculate total number of frames
    num_frames = int(np.ceil(float(np.abs(len(data) - window_length)) / window_step))

    if windowing_function == "hamming":
        windowing_function = np.hamming(window_length)

    # Pad signal to make sure that all frames have equal number of samples
    pad_signal_length = num_frames * window_step + window_length
    z = np.zeros((pad_signal_length - len(data)))
    pad_signal = np.append(data, z)

    lpc = np.zeros((lpc_length +1 , num_frames))
    for k in range(num_frames):
        starting_position = k*window_step
        # n = np.linspace(0.5,window_length-0.5,num=window_length)

        datawin = pad_signal[starting_position:(starting_position+window_length)]
        if np.max(np.abs(datawin)) == 0:
            continue
        datawin = datawin/np.max(np.abs(datawin)) # normalize

        # The autocorrelation of a signal is the convolution with itself r_k = x_n * x_{-n}
        # which in turn is in the z-domain R(z) = |X(z)|^2. It follows that a simple way to
        # calculate the autocorrelation is to take the DFT, absolute, square, and inverse-DFT.
        try:
            X = scipy.fft.fft(datawin*windowing_function)
            autocovariance = np.real(scipy.fft.ifft(np.abs(X)**2))
            b = np.zeros([lpc_length+1,1])
            b[0] = 1.
            a = solve_toeplitz(autocovariance[0:lpc_length+1], b)
            a = a/a[0]
        except:
            print("singular principa minor")
        else:
            lpc[:, k] = a.reshape((-1))

    return lpc

def lpcc(data, fs, window_length_ms, window_step_ms, lpc_length=0, windowing_function="hamming"):

    if lpc_length == 0:
        lpc_length = int(1.25*fs/1000)

    total_length = len(data)
    # Convert frame length and step from seconds to samples
    window_length = int(window_length_ms * fs / 1000)
    window_step = int(window_step_ms * fs / 1000)

    if windowing_function == "hamming":
        windowing_function = np.hamming(window_length)

    window_count = int((total_length-window_length)/window_step) + 1
    lpccs = np.zeros((lpc_length +1 , window_count))
    for k in range(window_count):

        starting_position = k*window_step

        datawin = data[starting_position:(starting_position+window_length)]
        if np.max(np.abs(datawin)) == 0:
            continue
        datawin = datawin/np.max(np.abs(datawin)) # normalize

        # The autocorrelation of a signal is the convolution with itself r_k = x_n * x_{-n}
        # which in turn is in the z-domain R(z) = |X(z)|^2. It follows that a simple way to
        # calculate the autocorrelation is to take the DFT, absolute, square, and inverse-DFT.

        X = scipy.fft.fft(datawin*windowing_function)
        autocovariance = np.real(scipy.fft.ifft(np.abs(X)**2))
        b = np.zeros([lpc_length+1,1])
        b[0] = 1.
        a = solve_toeplitz(autocovariance[0:lpc_length+1], b)
        a = a/a[0]


        if np.sum(a) == 0:
          lpcc = tuple(np.zeros(len(a)))

        else:
          powerspectrum = np.abs(np.fft.fft(a))  ** 2
          lpcc = np.fft.ifft(np.log(powerspectrum))
        lpccs[:, k] = (np.abs(lpcc)).reshape((-1))

    return lpccs

def spectral_envelope(data, fs, window_length_ms, window_step_ms, spectrum_length=512, windowing_function="hamming"):
    frames = frame_signal(data, fs, window_length_ms, window_step_ms, windowing_function=windowing_function)
    nyquist_frequency = fs // 2
    
    len_spectrum = spectrum_length // 2 + 1
    frequency_step_Hz = 500
    frequency_step = int(len_spectrum*frequency_step_Hz/nyquist_frequency)
    frequency_bins = int(len_spectrum/frequency_step+.5)


    envelopes = np.zeros((frequency_bins , frames.shape[0]))

    for i, window in enumerate(frames):
        spectrum = scipy.fft.rfft(window,n=spectrum_length)

        # filterbank
        frequency_step_Hz = 500
        frequency_step = int(len(spectrum)*frequency_step_Hz/nyquist_frequency)
        frequency_bins = int(len(spectrum)/frequency_step+.5)

        slope = np.arange(.5,frequency_step+.5,1)/(frequency_step+1)
        backslope = np.flipud(slope)
        filterbank = np.zeros((len(spectrum),frequency_bins))
        filterbank[0:frequency_step,0] = 1
        filterbank[(-frequency_step):-1,-1] = 1
        for k in range(frequency_bins-1):
            idx = int((k+0.25)*frequency_step) + np.arange(0,frequency_step)
            filterbank[idx,k+1] = slope
            filterbank[idx,k] = backslope

        # smoothing and downsampling
        spectrum_smoothed = np.matmul(np.transpose(filterbank),np.abs(spectrum)**2)
        logspectrum_smoothed = 10*np.log10(spectrum_smoothed  + np.finfo(float).eps)
        envelopes[:, i] = logspectrum_smoothed

    return envelopes


def calculate_cpp(data, fs, window_length_ms, window_step_ms, window="hamming"):

    frame_length = int(window_length_ms * fs / 1000)
    frame_step = int(window_step_ms * fs / 1000)


    # Windowing
    frames = librosa.util.frame(data, frame_length=int(frame_length), hop_length=int(frame_step))
    if window == "hamming":
            window = np.hamming(int(frame_length))
    windowed_frames = frames * window.reshape(-1, 1)

    # Fourier-Transform and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(windowed_frames.T, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Log power spectrum
    log_pow_frames = np.log(pow_frames + np.finfo(float).eps)

    # Compute real cepstrum
    real_cepstrum = np.fft.irfft(log_pow_frames, NFFT)

    # Identify peak and calculate CPP
    peak = np.max(real_cepstrum, axis=1)
    baseline = np.mean(real_cepstrum, axis=1)
    cpp = peak - baseline

    return cpp


def hammIndex(x, fs):

    # edges frequencies
    f1 = 2000
    f2 = 5000

    # checking and normalizing
    norm_signal = Preprocessing(fs, x)

    # estimation Power Spectral Density
    fq, Pxx = scipy.signal.welch(norm_signal, fs, nperseg=2048)
    Pxx_dB = 10*np.log10(Pxx)

    # sample position estimation
    n1 = (np.abs(fq - f1)).argmin()
    n2 = (np.abs(fq - f2)).argmin()

    # extract interval from 0-2000 Hz
    seg1 = Pxx_dB[0:n1]

    # extract interval from 2000-5000 Hz
    seg2 = Pxx_dB[n1:n2]

    # computing max SPL (Sound Preasure Level)
    SPL02 = max(seg1)
    SPL25 = max(seg2)
    HammIndex = SPL02-SPL25

    return HammIndex,fq,Pxx_dB



def plp_features(signal, sample_rate, num_filters=20, N_fft=512, fmin=20, fmax=8000):
    # Pre-emphasis

    # Hamming window
    frames = librosa.util.frame(signal, frame_length=N_fft, hop_length=N_fft // 2)
    windowed_frames = frames * np.hamming(N_fft)[:, None]

    # FFT and power spectrum
    fft_frames = np.fft.rfft(windowed_frames.T, N_fft)
    power_frames = np.abs(fft_frames) ** 2
    power_frames_db = 10 * np.log10(power_frames + np.finfo(float).eps)  # Add epsilon to avoid log(0)

    # Trapezoidal filter bank
    # filter_bank = trapezoidal_filter_bank(num_filters, N_fft, sample_rate)
    filter_bank, bark_freqs = bark_fbanks.bark_filter_banks(nfilts=num_filters,
                                                    nfft=N_fft,
                                                    fs=sample_rate,
                                                    low_freq=fmin,
                                                    high_freq=fmax)

    filter_bank_energy = np.dot(power_frames_db, filter_bank.T)

    # equal loudness pre_emphasis
    E = lambda w: ((w**2 + 56.8 * 10**6) * w**4) / (
        (w**2 + 6.3 * 10**6)
        * (w**2 + 0.38 * 10**9)
        * (w**6 + 9.58 * 10**26)
    )

    filter_bank_energy_nonlinear = [E(w) for w in filter_bank_energy]
    # Non-linear transformation (cubic root)
    filter_bank_energy_nonlinear = np.power(np.abs(filter_bank_energy), 1/3)
    lpc_length = 8
    order = lpc_length + 1
    lpccs = np.zeros((filter_bank_energy_nonlinear.shape[0], order))
    
    for k, frame in enumerate(filter_bank_energy_nonlinear):

        X = scipy.fft.fft(frame)
        autocovariance = np.real(scipy.fft.ifft(np.abs(X)**2))
        b = np.zeros([lpc_length+1,1])
        b[0] = 1.
        a = solve_toeplitz(autocovariance[0:lpc_length+1], b)
        a = a/a[0]

        if np.sum(a) == 0:
          lpcc = tuple(np.zeros(len(a)))

        else:
          powerspectrum = np.abs(np.fft.fft(a))  ** 2
          lpcc = np.fft.ifft(np.log(powerspectrum))
        lpccs[k, :] = (np.abs(lpcc)).reshape((-1))

    return lpccs.T


def harmonicity(data, fs):
    # Load the speech signal into a Parselmouth Sound object
    snd = parselmouth.Sound(data, fs)

    # Use Parselmouth to calculate HNR
    hnr = snd.to_harmonicity()

    # Extract the HNR values
    hnr_values = hnr.values
    hnr_values = hnr_values[hnr_values > 0]

    # You can calculate the mean HNR over the entire signal
    mean_hnr = np.nanmean(hnr_values)  # Using nanmean to ignore NaN values

    return mean_hnr


def calculate_lsp_freqs_for_frames(signal, fs, window_length_ms, window_step_ms, order, window="hamming"):
    """Calculate the Line Spectral Pair (LSP) frequencies for each frame of the signal."""

    c = lpc(signal, fs, window_length_ms, window_step_ms, lpc_length=order, windowing_function=window)
    lsp_freqs = np.zeros((order, c.shape[1]))

    for i, frame in enumerate(c.T):
        out = poly2lsf(frame)
        lsp_freqs[:, i] = out

    return lsp_freqs


def calculate_frame_wise_zcr(signal, fs, window_length_ms, window_step_ms):
    """Calculate the Zero-Crossing Rate (ZCR) for each frame of an audio signal using librosa."""

    # Convert frame length and step from seconds to samples
    window_length = int(window_length_ms * fs / 1000)
    window_step = int(window_step_ms * fs / 1000)
    frames = librosa.util.frame(signal, frame_length=window_length, hop_length=window_step)
    zcr = np.array([librosa.zero_crossings(frame, pad=False).mean() for frame in frames.T])
    return zcr
