import textstat
import whisperx
from pocketsphinx import Decoder, AudioFile, Segmenter

import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from speech_fluency_and_speech_production_dynamics_helper import *

nltk.download('cmudict')
nltk.download('punkt')


class SpeechBehavior:
    def __init__(self, vad_model, vad_model_utils, transcription_model):
        self.vad_model = vad_model
        self.vad_model_utils = vad_model_utils
        self.transcription_model = transcription_model
        self.transcription_result = []
        self.silence_ranges = []
        self.speech_ranges = []
        self.SAMPLING_RATE = 16000
        self.text = ""
        self.all_phonemes = []
        self.data = None

    def configure(self, filename):
        (get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = self.vad_model_utils

        self.data = read_audio(filename, sampling_rate=self.SAMPLING_RATE)
        # get speech timestamps from full audio file
        speech_timestamps = get_speech_timestamps(self.data, self.vad_model, sampling_rate=self.SAMPLING_RATE, threshold=0.5)
        # pprint(speech_timestamps)

        self.speech_ranges = []
        for timestamp in speech_timestamps:
            self.speech_ranges.append((timestamp["start"], timestamp["end"]))

        self.silence_ranges = remove_subranges(0, len(self.data), self.speech_ranges)

        device = "cuda"
        audio = whisperx.load_audio(filename)
        self.transcription_result = self.transcription_model.transcribe(audio, batch_size=8)
        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=self.transcription_result["language"], device=device)
        self.transcription_result = whisperx.align(self.transcription_result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        texts = []
        for segment in self.transcription_result["segments"]:
            texts.append(segment["text"].strip())

        self.text = " ".join(texts)

    def phoneme_alignment(self, file_name):
        
        arpabet = nltk.corpus.cmudict.dict()
        self.all_phonemes = []
        n_hypothesis = 0

        for segment in self.transcription_result["segments"]:
            text = segment["text"].translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower().strip()
            start_time = segment["start"]
            end_time = segment["end"]

            # data = audio.getfp().read()
            data, sr = read_wav_segment_new(file_name, self.SAMPLING_RATE, start_time, end_time)
            decoder = Decoder(samprate=sr, bestpath=False)
            try:
                decoder.set_align_text(text)
                decoder.start_utt()
                decoder.process_raw(data, full_utt=True)
                decoder.end_utt()
                decoder.set_alignment()
                decoder.start_utt()
                decoder.process_raw(data, full_utt=True)
                decoder.end_utt()
            except:
                # print(text)
                for word_dict in segment["words"]:
                    word = word_dict["word"].translate(str.maketrans('', '', string.punctuation.replace("'", ""))).lower().strip()
                    try:    
                        start, end = word_dict["start"], word_dict["end"]
                    except:
                        continue

                    try:
                        phonemes = arpabet[word][0]
                    except:
                        word_normalize = word.replace("'", "")
                        if word_normalize in arpabet:
                            phonemes = arpabet[word_normalize][0]
                        else:
                            phonemes = word_normalize

                    duration = (end - start) / len(phonemes)
                    # print(start, end, duration)

                    word_entry = {"text": word, "start": start, "end": end, "phonemes": []}

                    for i, phone in enumerate(phonemes):
                        phoneme_start = round(start+i*duration, 3)
                        phoneme_end = round(start+(i+1)*duration, 3)
                        # print(phoneme_start, phoneme_end)
                        stripped_phone = phone.translate(str.maketrans('', '', string.digits))
                        word_entry["phonemes"].append({"name": stripped_phone, "start": phoneme_start, "end": phoneme_end})

                    self.all_phonemes.append(word_entry)
                n_hypothesis += 1
            else:

                for word in decoder.get_alignment():
                    if word.name == "<sil>":
                        continue
                    word_entry = {"text": word.name, "start": round(start_time+word.start/100, 3), "end": round(start_time+(word.start+word.duration)/100, 3), "phonemes": []}
                    for phone in word:
                        name = phone.name
                        start = start_time + (phone.start / (100))
                        end = start_time + ((phone.start + phone.duration) / (100))
                        start = round(start, 3)
                        end = round(end, 3)
                        word_entry["phonemes"].append({"name": name, "start": start, "end": end})
                    self.all_phonemes.append(word_entry)

        return (len(self.transcription_result["segments"]), n_hypothesis)
    
    def articulation_rate(self):
        num_phonemes = 0
        duration = 0
        for word in self.all_phonemes:
            duration += (word["end"] - word["start"])
            num_phonemes += len(word["phonemes"])

        return num_phonemes / (duration + 0.1)
    
    def mean_inter_syllabic_pauses(self):
        num_silence = 0
        sum_silence = 0
        for word in self.all_phonemes:
            ranges = list(map(lambda x: (x["start"], x["end"]), word["phonemes"]))
            sil_ranges = remove_subranges(word["start"], word["end"], ranges)
            num_silence += len(sil_ranges)
            sum_silence += sum(list(map(lambda x: x[1] - x[0], sil_ranges)))
        return sum_silence / num_silence if num_silence > 0 else None
    
    def num_of_sylablles(self):
        total_duration = len(self.data) / self.SAMPLING_RATE
        all = 0
        for word in self.all_phonemes:
            all += len(extract_syllables(word["phonemes"]))
        return all / total_duration

    def syllabic_interval_duration(self):
        total_duration = len(self.data) / self.SAMPLING_RATE
        duration = 0
        for word in self.all_phonemes:
            for syllable in extract_syllables(word["phonemes"]):
                duration = duration + (syllable["end"] - syllable["start"])

        return duration / (total_duration + 0.1)

    def vowel_duration(self):
        total_duration = len(self.data) / self.SAMPLING_RATE
        vowel_sounds = {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'}
        vwl_dur = 0

        for word in self.all_phonemes:
            for phone in word["phonemes"]:
                if phone["name"] in vowel_sounds:
                    vwl_dur += (phone["end"] - phone["start"])

        return vwl_dur / (total_duration + 0.1)

    def phonation_time(self):
        time = sum(list(map(lambda x: (x["end"] - x["start"]), self.all_phonemes)))
        return time

    def percentage_phonation_time(self):
        phonation_time = self.phonation_time()
        total_duration = len(self.data) / self.SAMPLING_RATE
        return round(phonation_time / total_duration, 3) * 100

    def transformed_phonation_rate(self):
        phonation_rate = self.percentage_phonation_time() / 100
        transformed_rate = np.arcsin(np.sqrt(phonation_rate))

        return transformed_rate

    def tlt(self):
        locution_time_ms = (self.speech_ranges[-1][1] - self.speech_ranges[0][0]) / self.SAMPLING_RATE * 1000

        speech_durations = calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE)
        total_duration_speech_ms = sum(speech_durations)
        ratio = locution_time_ms / total_duration_speech_ms
        return ratio

    def token_duration(self):
        speech_durations = calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE)
        total_speech_duration_ms = sum(speech_durations)

        return total_speech_duration_ms

    def syllable_count(self):
        total_duration = len(self.data) / self.SAMPLING_RATE
        
        n = textstat.syllable_count(self.text.lower())
        return n / total_duration

    def count_tokens(self):
        total_duration = len(self.data) / self.SAMPLING_RATE

        tokens = word_tokenize(self.text)
        return len(tokens) / total_duration

    def speech_rate_words(self):
        number_of_words = len(self.text.split())

        speech_durations = calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE)
        total_speech_duration_ms = sum(speech_durations)
        total_speech_duration_min = total_speech_duration_ms / (60 * 1000)

        speech_rate_words = number_of_words / total_speech_duration_min

        return speech_rate_words

    def speech_rate_syllable(self):
        syllable_count = self.syllable_count()

        speech_durations = calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE)
        total_speech_duration_ms = sum(speech_durations)
        total_speech_duration_min = total_speech_duration_ms / (60 * 1000)

        speech_rate_syllables = syllable_count / total_speech_duration_min

        return speech_rate_syllables

    def phonation_to_syllable(self):
        return self.phonation_time() / self.syllable_count()

    def average_num_of_speech_segments(self):
        number_of_speech_segments = len(self.speech_ranges)
        total_duration_s = len(self.data) / self.SAMPLING_RATE

        average = number_of_speech_segments / total_duration_s
        return average

    def mean_words_in_utterance(self):
        total_number_of_words = 0
        number_of_utterances = len(self.transcription_result["segments"])
        for segment in self.transcription_result["segments"]:
            total_number_of_words += len(segment["text"].split())

        mean = total_number_of_words / number_of_utterances

        return mean

    def mean_length_sentence(self):
        total_duration = len(self.data) / self.SAMPLING_RATE

        sentences = sent_tokenize(self.text)
        mls = (self.count_tokens() * total_duration) / len(sentences)

        return mls

    def relative_sentence_duration(self):
        speech_durations = calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE)
        total_speech_duration_s = sum(speech_durations) / 1000

        rsd = []
        for segment in self.transcription_result["segments"]:
            duration = (segment["end"] - segment["start"])
            rsd.append(duration / total_speech_duration_s)

        return rsd


    def regularity_of_segments(self):
        # Convert segment tuples to durations
        speech_durations = calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE)
        silence_durations = calculate_duration_ms(self.silence_ranges, self.SAMPLING_RATE)

        # Adjusting the silence segments
        if self.silence_ranges:
            # Exclude silence before first speech segment
            if self.silence_ranges[0][1] <= self.speech_ranges[0][0]:
                silence_durations.pop(0)
            # Exclude silence after last speech segment
            if self.silence_ranges[-1][0] >= self.speech_ranges[-1][1]:
                silence_durations.pop()

        # Calculate statistics for speech and silence segments
        speech_mean, speech_std_dev, speech_cv = calculate_statistics(speech_durations)
        silence_mean, silence_std_dev, silence_cv = calculate_statistics(silence_durations)

        # Calculate PVI for speech segments
        speech_pvi = calculate_raw_pvi(speech_durations)
        # Calculate Normalized PVI for speech segments
        speech_npvi = calculate_normalized_pvi(speech_durations)

        # Calculate PVI for speech segments
        silence_pvi = calculate_raw_pvi(silence_durations)
        # Calculate Normalized PVI for speech segments
        silence_npvi = calculate_normalized_pvi(silence_durations)
        res = [
            speech_pvi, speech_npvi, speech_mean, speech_std_dev, speech_cv,
            silence_pvi, silence_npvi, silence_mean, silence_std_dev, silence_cv
            ]
        return res
    

    def alternating_regularity(self):
        alternating_durations = calculate_alternating_durations(self.speech_ranges, self.SAMPLING_RATE)

        # Calculate PVI for speech segments
        speech_pvi = calculate_raw_pvi(alternating_durations)
        # Calculate Normalized PVI for speech segments
        speech_npvi = calculate_normalized_pvi(alternating_durations)

        return [speech_pvi, speech_npvi]



