import sys
from statistical_functions import *
from scipy.io import wavfile
import librosa


from frequency_parameters import *
from cepstral_coefficients_and_spectral_features import *
from voice_quality import *
from loudness_and_intensity import *
from complexity import *
from rhythmic_structure import *
from speech_fluency_and_speech_production_dynamics import *
import numpy as np
import inspect


def process_file(filepath):
    # fs, data = wavfile.read(filepath)
    data, fs = librosa.load(filepath, sr=16000)
    # fs, data = wavfile.read(filepath)
    # data = data / np.max(np.abs(data))

    window_length_ms = 50
    window_step_ms = 25

    F0 = get_pitch(data, fs, window_length_ms, window_step_ms)
    F0 = F0[~np.isnan(F0)]
    jitter = calculate_time_varying_jitter(F0, fs, window_length_ms, window_step_ms, window_length_ms*2, window_step_ms*2)
    F1 = get_formants_frame_based(data, fs, window_length_ms, window_step_ms, [1, 2, 3])
    print("PROCESSED FREQUENCY PARAMETERS...")
    # 
    msc = compute_msc(data, fs, nfft=512, window_length_ms=window_length_ms, window_step_ms=window_step_ms, num_msc=13)
    centroids = spectral_centriod(data, fs, window_length_ms, window_step_ms)
    LTAS = ltas(data, fs, window_length_ms, window_step_ms)[0]
    ALPHA_RATIO = alpha_ratio(data, fs, window_length_ms, window_step_ms, (0, 1000), (1000, 5000))
    LOG_MEL_SPECTROGRAM = log_mel_spectrogram(data, fs, window_length_ms, window_step_ms, melbands=8, fmin=20, fmax=6500)
    MFCC = mfcc(data, fs, window_length_ms, window_step_ms, melbands=26, lifter=20)[:15]
    LPC = lpc(data, fs, window_length_ms, window_step_ms)
    LPCC = lpcc(data, fs, window_length_ms, window_step_ms, lpc_length=8)
    ENVELOPE = spectral_envelope(data, fs, window_length_ms, window_step_ms)
    CPP = calculate_cpp(data, fs, window_length_ms, window_step_ms)
    HAMM_INDEX = hammIndex(data, fs)[0]
    PLP = plp_features(data, fs, num_filters=26, fmin=20, fmax=8000)
    HARMONICITY = harmonicity(data, fs)

    lspFreq = calculate_lsp_freqs_for_frames(data, fs, window_length_ms, window_step_ms, order=8)
    ZCR = calculate_frame_wise_zcr(data, fs, window_length_ms, window_step_ms)  # PCM zero-crossing rate for 100 frames
  
    print("PROCESSED SPECTRAL DOMAIN...")

    
    APQ = None
    SHIMMER = analyze_audio_shimmer(data, fs, window_length_ms, window_step_ms)
    HNR, NHR = calculate_frame_level_hnr(data, fs, window_length_ms, window_step_ms)
    APQ_range, APQ_std = amplitude_range(data, fs, window_length_ms, window_step_ms)

    print("PROCESSED VOICE QUALITY...")

    RMS = rms_amplitude(data, fs, window_length_ms, window_step_ms)
    SPL = spl_per(data, fs, window_length_ms, window_step_ms)
    PEAK = peak_amplitude(data, fs, window_length_ms, window_step_ms)
    STE = ste_amplitude(data, fs, window_length_ms, window_step_ms)
    INTENSITY = intensity(data, fs, window_length_ms, window_step_ms)
    print("PROCESSED LOUDNESS AND INTENSITY PARAMETERS...")


    HFD = calculate_hfd_per_frame(data, fs, window_length_ms, window_step_ms, 10)
    FREQ_ENTROPY = calculate_frequency_entropy(data, fs, window_length_ms, window_step_ms)
    AMP_ENTROPY = calculate_amplitude_entropy(data, fs, window_length_ms, window_step_ms)
    print("PROCESSED COMPLEXITY...")


    results = []
    # Appending the processed results to the list
    # Frequency parameters
    results.append(process_row(F0, 'F0'))
    results.append(process_row(np.array(jitter), 'Jitter'))
    results.append(process_row(F1[0, :], 'F1'))
    results.append(process_row(F1[1, :], 'F2'))
    results.append(process_row(F1[2, :], 'F3'))
    print(f"Frequency parameters: {length(results)}")

    # Spectral domain
    results.append(process_matrix(msc, 'MSC'))
    results.append(process_row(np.array(centroids), 'CENTRIODS'))
    results.append(process_row(LTAS, 'LTAS'))
    results.append(process_matrix(LOG_MEL_SPECTROGRAM, 'LOG_MEL_SPECTROGRAM'))
    results.append(process_matrix(MFCC, 'MFCC'))
    results.append(process_matrix(LPC, 'LPC'))
    results.append(process_matrix(LPCC, 'LPCC'))
    results.append(process_matrix(ENVELOPE, 'ENVELOPE'))
    results.append(process_row(CPP, 'CPP'))
    results.append(process_matrix(PLP, 'PLP'))
    results.append(process_matrix(lspFreq, "LSP"))
    print(f"Spectral Domain: {length(results)}")

    # Voice Quality
    # results.append(process_row(APQ, 'APQ'))
    results.append(process_row(APQ_std, 'APQ2'))
    results.append(process_row(SHIMMER, 'SHIMMER'))
    results.append(process_row(HNR, 'HNR'))
    results.append(process_row(NHR, 'NHR'))
    results.append({'ALPHA_RATIO': ALPHA_RATIO})
    results.append(process_row(HAMM_INDEX, 'HAMMARBERG_INDEX'))
    results.append({'HARMONICITY': HARMONICITY})



    print(f"Voice Quality: {length(results)}")
    
    # Loudness and intensity of the sound
    results.append(process_row(APQ_range, 'Amplitude_Range'))
    results.append(process_row(RMS, 'RMS'))
    results.append(process_row(SPL, 'SPL'))
    results.append(process_row(PEAK, 'PEAK'))
    results.append(process_row(STE, 'STE'))
    results.append(process_row(INTENSITY, 'INTENSITY'))
    print(f"Loudness: {length(results)}")

    # characterizing the complexity of voice signals
    results.append(process_row(HFD, 'HFD'))
    results.append(process_row(FREQ_ENTROPY, 'FREQ_ENTROPY'))
    results.append(process_row(AMP_ENTROPY, 'AMP_ENTROPY'))
    results.append(process_row(ZCR, "ZCR"))

    print(f"Complexity: {length(results)}")

    final_results = {}
    for res in results:
        final_results.update(res)


    print(len(final_results))

    return final_results


def process_file_model(filepath, vad_model, utils, transcription_model):

    p_results = {}
    pause_behavior = PauseBehavior(vad_model, utils, transcription_model)
    pause_behavior.configure(filepath)
    
    voiceProb_signal =  vad_model.audio_forward(pause_behavior.data, sr=16000)
    voiceProb_signal = np.array(voiceProb_signal[0])
    prob = process_row(voiceProb_signal, "voiceProb")

    # List of methods to exclude
    excluded_methods = ['__init__', 'configure']

    # Iterate through all methods of the class instance
    for name, method in inspect.getmembers(pause_behavior, predicate=inspect.ismethod):
        # Check if the method is not in the excluded list
        if name not in excluded_methods:
            # Invoke each method and store the result
            p_results[name] = method()


    s_results = {}
    speech_behavior = SpeechBehavior(vad_model, utils, transcription_model)
    # speech_behavior.configure(filepath)
    speech_behavior.data = pause_behavior.data
    speech_behavior.silence_ranges = pause_behavior.silence_ranges
    speech_behavior.speech_ranges = pause_behavior.speech_ranges
    speech_behavior.transcription_result = pause_behavior.transcription_result
    speech_behavior.text = pause_behavior.text
    
    alignment_error = speech_behavior.phoneme_alignment(filepath)
    print(f"alignment_error: {alignment_error}")

      # List of methods to exclude
    excluded_methods = ['__init__', 'configure', 'phoneme_alignment', 'relative_sentence_duration', 'regularity_of_segments', 'alternating_regularity']

    # Iterate through all methods of the class instance
    for name, method in inspect.getmembers(speech_behavior, predicate=inspect.ismethod):
        # Check if the method is not in the excluded list
        if name not in excluded_methods:
            # Invoke each method and store the result
            s_results[name] = method()
    for i, res in enumerate(speech_behavior.regularity_of_segments()):
        s_results[f"regularity_{i}"] = res
    for i, res in enumerate(speech_behavior.alternating_regularity()):
        s_results[f"PVI_{i}"] = res
        
    relative_sentence_duration = speech_behavior.relative_sentence_duration()
    s_results.update(process_row(np.array(relative_sentence_duration), "relative_sentence_duration"))

    p_results.update(prob)
    print(len(p_results), len(s_results))
    p_results.update(s_results)

    return p_results




def process_matrix(matrix, feature_name):
    matrix_results = {}
    for i, row in enumerate(matrix):
        result = process_row(row, feature_name, i)
        matrix_results.update(result)

    return matrix_results


def process_row(row, feature_name, index=-1):
    row_sma = sma(row)
    results = {}
    for func_name in function_names:
        func = getattr(sys.modules[__name__], func_name)
        try:
            result = func(row_sma)
        except:
            result = None
        if index >= 0:
            name = f"{feature_name}_sma[{index}]_{func_name}"
        else:
            name = f"{feature_name}_sma_{func_name}"

        results[name] = result

    row_sma_de = de(row_sma)
    for func_name in function_names:
        func = getattr(sys.modules[__name__], func_name)
        try:
            result = func(row_sma_de)
        except:
            result = None
        if index >= 0:
            name = f"{feature_name}_sma_de[{index}]_{func_name}"
        else:
            name = f"{feature_name}_sma_de_{func_name}"

        results[name] = result

    return results

def length(res):
    l = sum([len(elm) for elm in res])
    return l
        

# process_file("process_file")
