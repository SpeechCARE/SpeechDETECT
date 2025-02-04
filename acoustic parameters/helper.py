import numpy as np
import scipy

def frame_signal(signal, sample_rate, frame_length_ms, frame_step_ms, windowing_function="hamming"):
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

    # Initialize frames array
    frames = np.zeros((num_frames, frame_length_samples))

    # Fill frames array with signal segments
    for i in range(num_frames):
        start = i * frame_step_samples
        end = start + frame_length_samples
        frames[i] = pad_signal[start:end]
        frames[i] = frames[i] * windowing_function

    return frames

def freq2mel(f): return 2595*np.log10(1 + (f/700))
def mel2freq(m): return 700*(10**(m/2595) - 1)


def hz_to_bark(hz):
    return 6 * np.arcsinh(hz / 600)

def bark_to_hz(bark):
    return 600 * np.sinh(bark / 6)


def stft(data,fs,window_length_ms=30,window_step_ms=20,windowing_function="hamming"):
    window_length = int(window_length_ms*fs/1000)
    window_step = int(window_step_ms*fs/1000)
    if windowing_function == "hamming":
        # windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2
        windowing_function = np.hanning(window_length)

    total_length = len(data)
    window_count = int( (total_length-window_length)/window_step) + 1

    spectrum_length = int((window_length)/2)+1
    spectrogram = np.zeros((window_count,spectrum_length))

    for k in range(window_count):
        starting_position = k*window_step

        data_vector = data[starting_position:(starting_position+window_length),]
        window_spectrum = np.abs(scipy.fft.rfft(data_vector*windowing_function,n=window_length))

        spectrogram[k,:] = window_spectrum

    return spectrogram


def lifter_ceps(ceps: np.ndarray, lift: int = 3) -> np.ndarray:
    """
    Apply a cepstral lifter the the matrix of cepstra. This has the effect of
    increasing the magnitude of the high frequency DCT coeffs. the liftering is
    implemented as in [Ellis-plp]_.

    Args:
        ceps (numpy.ndarray) : the matrix of mel-cepstra, will be numframes * numcep in size.
        lift           (int) : the liftering coefficient to use. (Default is 3).

    Returns:
        (numpy.ndarray) liftered cepstra.

    Note:
        - The liftering is applied to matrix of cepstra (one per column).
        - If the lift is positive (Use values smaller than 10 for meaningful results), then
          the liftering uses the exponent. However, if the lift is negative (Use integers), then
          the sine curve liftering is used.

    References:
        .. [Ellis-plp] : Ellis, D. P. W., 2005, PLP and RASTA and MFCC, and inversion in Matlab,
                     <http://www.ee.columbia.edu/~dpwe/resources/matlab/rastamat/>
    """
    # if lift == 0 or lift > 10:
    #     return ceps

    if lift > 0:
        lift_vec = np.array([1] + [i**lift for i in range(1, ceps.shape[1])])
        lift_mat = np.diag(lift_vec)
        return np.dot(ceps, lift_mat)

    else:
        lift = int(-1 * lift)
        lift_vec = 1 + (lift / 2.0) * np.sin(
            np.pi * np.arange(1, 1 + ceps.shape[1]) / lift
        )
        return ceps * lift_vec
    


def Preprocessing(fs, x):

    # ensure only one channel
    if x.ndim>1:
        mono_signal = 1/2*(x[:,0] + x[:,1])
    else:
        mono_signal = x

    # normalizing [-1, 1]
    # norm_signal = mono_signal/max(mono_signal)
    norm_signal = mono_signal

    # check minimum signal duration
    T = 5 # seconds
    t = len(norm_signal)/fs # Signal duration in seconds

    # check signal duration
    if t<T:
        raise ValueError('Signal duration must be greater than 5s')

    # check sampling frequency
    if fs <12000:
        raise ValueError('Sampling Rate fs must be greater than 12KHz')

    return norm_signal


from scipy.signal import deconvolve
def poly2lsf(a):
    """Prediction polynomial to line spectral frequencies.

    converts the prediction polynomial specified by A,
    into the corresponding line spectral frequencies, LSF.
    normalizes the prediction polynomial by A(1).

    .. doctest::

        >>> from spectrum import poly2lsf
        >>> a = [1.0000,  0.6149, 0.9899, 0.0000 ,0.0031, -0.0082]
        >>> lsf = poly2lsf(a)
        >>> lsf =  array([0.7842, 1.5605, 1.8776, 1.8984, 2.3593])

    .. seealso:: lsf2poly, poly2rc, poly2qc, rc2is
    """

    #Line spectral frequencies are not defined for complex polynomials.
    if not np.any(a):
        return np.zeros(len(a) - 1)

    # Normalize the polynomial

    a = np.array(a)
    if a[0] != 1 and a[0] != 0:
        a/=a[0]

    if max(np.abs(np.roots(a))) >= 1.0:
        return np.zeros(len(a)-1)


    # Form the sum and differnce filters

    p  = len(a)-1   # The leading one in the polynomial is not used
    a1 = np.concatenate((a, np.array([0])))
    a2 = a1[-1::-1]
    P1 = a1 - a2        # Difference filter
    Q1 = a1 + a2        # Sum Filter

    # If order is even, remove the known root at z = 1 for P1 and z = -1 for Q1
    # If odd, remove both the roots from P1

    if p%2: # Odd order
        P, r = deconvolve(P1,[1, 0 ,-1])
        Q = Q1
    else:          # Even order
        P, r = deconvolve(P1, [1, -1])
        Q, r = deconvolve(Q1, [1,  1])

    rP  = np.roots(P)
    rQ  = np.roots(Q)

    aP  = np.angle(rP[1::2])
    aQ  = np.angle(rQ[1::2])

    lsf = sorted(np.concatenate((-aP,-aQ)))

    return lsf