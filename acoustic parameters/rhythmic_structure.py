import textstat
import whisperx
import nltk
from nltk.tokenize import word_tokenize

import statistics

from speech_fluency_and_speech_production_dynamics_helper import calculate_duration_ms, remove_subranges

nltk.download('punkt')

class PauseBehavior:
    def __init__(self, vad_model, vad_model_utils, transcription_model):
        self.vad_model = vad_model
        self.vad_model_utils = vad_model_utils
        self.transcription_model = transcription_model
        self.silence_ranges = []
        self.speech_ranges = []
        self.SAMPLING_RATE = 16000
        self.text = ""
        self.data = None

        self.filler_words = {
        "um", "uh", "ah", "oh", "like", "you know", "so", "actually", "basically", "seriously",
        "literally", "i mean", "you see", "well", "okay", "right", "sort of", "kind of",
        "i guess", "you know what i mean", "believe me", "to be honest", "i think",
        "i suppose", "in a sense", "anyway", "and all that", "at the end of the day",
        "that said", "you know what?", "i feel like", "i don't know"
        }

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



    def count_pause_segments(self):
        total_duration = len(self.data) / self.SAMPLING_RATE
        return len(self.silence_ranges) / total_duration
    
    def pause_length(self):
        total_duration = len(self.data) / self.SAMPLING_RATE
        pause_len = sum(calculate_duration_ms(self.silence_ranges, self.SAMPLING_RATE))
        return pause_len / total_duration
    
    def vocalic_interval(self):
        total_duration = len(self.data) / self.SAMPLING_RATE
        speech_len = sum(calculate_duration_ms(self.speech_ranges, self.SAMPLING_RATE))
        return speech_len / total_duration
    

    def pause_lengths_avg(self):
        silence_durations = calculate_duration_ms(self.silence_ranges, self.SAMPLING_RATE)
        pause_mean = statistics.mean(silence_durations)
        if len(silence_durations) > 2:
            pause_std = statistics.stdev(silence_durations)
        else:
            pause_std = None

        return pause_mean #, pause_std
    
    def pasue_speech_ratio(self):
        ratio = len(self.silence_ranges) / len(self.speech_ranges)

        return ratio
    
    def pause_speech_duration_ratio(self):
        return self.pause_length() / self.vocalic_interval()
    
    def pause_totallength_ratio(self):
        pause_len = self.pause_length() / 1000
        total_duration_s = len(self.data) / self.SAMPLING_RATE
        return pause_len / total_duration_s
    
    def num_words_to_num_pauses(self):
        n_words = len(self.text.split())
        n_pauses = self.count_pause_segments()
        standardized_pause_rate = n_words / (n_pauses + 1)

        return standardized_pause_rate
    
    def pause_to_syllable(self):
        number_of_pauses = self.count_pause_segments()
        # Counting the syllables in the transcription
        total_syllables = textstat.syllable_count(self.text)
        pauses_per_syllable = number_of_pauses / (total_syllables + 1)

        return pauses_per_syllable
    
    def pause_to_tokens(self):
        number_of_pauses = self.count_pause_segments()

        tokens = word_tokenize(self.text)
        number_of_tokens = len(tokens)

        # Calculating pauses per syllable
        pauses_per_token = number_of_pauses / (number_of_tokens + 1)
        return pauses_per_token
    

    def hesitation_rate(self, num_pauses=0):
        num_fillerwords = 0
        lower_transcription = self.text.lower()

        for filler in self.filler_words:
            num_fillerwords += lower_transcription.count(filler)
        if num_pauses:
            num_fillerwords += num_pauses


        return num_fillerwords / (len(lower_transcription.split()) + 1)





    
