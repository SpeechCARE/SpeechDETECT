import wave
import soundfile as sf
import numpy as np
import librosa

def calculate_duration_ms(ranges, sr):
    durations = []
    for rng in ranges:
        duration = (rng[1] - rng[0]) / sr * 1000
        durations.append(duration)

    return durations


def remove_subranges(full_range_start, full_range_end, subranges):
    # Start with the full range
    remaining_ranges = [(full_range_start, full_range_end)]

    # Process each subrange
    for s, e in subranges:
        new_remaining_ranges = []
        for r_start, r_end in remaining_ranges:
            # If subrange is completely outside the current range, ignore it
            if e < r_start or s > r_end:
                new_remaining_ranges.append((r_start, r_end))
            else:
                # Add the part before the subrange
                if s > r_start:
                    new_remaining_ranges.append((r_start, s))
                # Add the part after the subrange
                if e < r_end:
                    new_remaining_ranges.append((e, r_end))
        remaining_ranges = new_remaining_ranges

    return remaining_ranges


# Function to read a specific segment from a WAV file
def read_wav_segment_new(file_path, sample_rate, start_time, end_time):
    
    data, sample_rate = sf.read(file_path, dtype='int16')
    start_frame = int(start_time * sample_rate)
    end_frame = int(end_time * sample_rate)

    return data[start_frame:end_frame].tobytes(), sample_rate
 

# Function to read a specific segment from a WAV file
def read_wav_segment(file_path, start_time, end_time):
    with wave.open(file_path, 'rb') as wav:
        # Get the sample rate (number of frames per second)
        sample_rate = wav.getframerate()

        # Calculate the start and end frame indices
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)

        # Set the file pointer to the start frame
        wav.setpos(start_frame)

        # Calculate the number of frames to read
        frame_count = end_frame - start_frame

        # Read the specified number of frames
        segment_data = wav.readframes(frame_count)

        return segment_data, sample_rate


def extract_syllables(phonetic_transcription):
    vowel_sounds = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
    syllables = []
    current_syllable = []

    for phone in phonetic_transcription:
        if len(current_syllable) == 0:
            start_time = phone["start"]

        phoneme = phone["name"]
        current_syllable.append(phoneme)

        # When a vowel is encountered, it marks the end of a syllable
        if phoneme in vowel_sounds:
            syllables.append({"syllable": ' '.join(current_syllable), "start": start_time, "end": phone["end"]})
            current_syllable = []

    # Adding any remaining consonants as a syllable (for cases like final consonants)
    if current_syllable:
        # syllables.append({"syllable": ' '.join(current_syllable), "start": start_time, "end": phone["end"]})
        if len(syllables):
            syllables[-1]["syllable"] += ' '.join(current_syllable)
            # print(phonetic_transcription[-1]["end"])
            syllables[-1]["end"] = phonetic_transcription[-1]["end"]
            # print(syllables)
        else:
          syllables.append({"syllable": ' '.join(current_syllable), "start": start_time, "end": phone["end"]})

    return syllables


def calculate_statistics(segments):
    mean = np.mean(segments)
    std_dev = np.std(segments)
    coefficient_of_variation = (std_dev / mean) * 100
    return mean, std_dev, coefficient_of_variation

def calculate_raw_pvi(segments):
    differences = np.abs(np.diff(segments))
    raw_pvi = np.mean(differences)
    return raw_pvi

def calculate_normalized_pvi(segments):
    nPVI_values = []
    for i in range(len(segments) - 1):
        dur_i, dur_i_1 = segments[i], segments[i + 1]
        nPVI = abs(dur_i - dur_i_1) / ((dur_i + dur_i_1) / 2) * 100
        nPVI_values.append(nPVI)
    return np.mean(nPVI_values) if nPVI_values else 0


def calculate_silence_segments(speech_segments):
    silence_segments = []
    for i in range(len(speech_segments) - 1):
        end_current_speech = speech_segments[i][1]
        start_next_speech = speech_segments[i + 1][0]
        silence_segments.append((end_current_speech, start_next_speech))
    return silence_segments

def calculate_alternating_durations(speech_segments, SAMPLING_RATE):
    speech_durations = calculate_duration_ms(speech_segments, SAMPLING_RATE)
    silence_segments = calculate_silence_segments(speech_segments)
    silence_durations = calculate_duration_ms(silence_segments, SAMPLING_RATE)

    alternating_durations = []
    for i in range(len(speech_durations)):
        alternating_durations.append(speech_durations[i])
        if i < len(silence_durations):
            alternating_durations.append(silence_durations[i])

    return alternating_durations
