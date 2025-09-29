import os
import wave
import contextlib
import webrtcvad
import collections
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment, effects


def extract_audio(input_file, output_wav="output.wav"):
    if input_file.endswith((".mp4", ".mkv", ".avi")):
        video = VideoFileClip(input_file)
        if video.audio is None:
            raise ValueError(f"No audio track found in {input_file}")
        # Export audio to temp file first
        temp_wav = "temp.wav"
        video.audio.write_audiofile(temp_wav, fps=16000, nbytes=2, codec="pcm_s16le")
        # Convert to mono and apply preprocessing
        audio = AudioSegment.from_wav(temp_wav)
        audio = audio.set_channels(1)
        audio = effects.normalize(audio)  # normalize volume
        audio = audio.high_pass_filter(80)  # remove low-frequency music/bass
        audio.export(output_wav, format="wav")
        os.remove(temp_wav)
    else:
        # If already audio, convert/normalize with pydub
        audio = AudioSegment.from_file(input_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio = effects.normalize(audio)
        audio = audio.high_pass_filter(80)
        audio.export(output_wav, format="wav")
    return output_wav


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, "Audio must be mono"
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate == 16000
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n <= len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames, min_segment_s=0.3):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False

    segments = []
    segment_start = None

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                segment_start = ring_buffer[0][0].timestamp
                ring_buffer.clear()
        else:
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                segment_end = frame.timestamp + frame.duration
                if (segment_end - segment_start) >= min_segment_s:
                    segments.append({"start": round(segment_start, 2), "end": round(segment_end, 2)})
                ring_buffer.clear()

    if triggered:
        segment_end = frame.timestamp + frame.duration
        if (segment_end - segment_start) >= min_segment_s:
            segments.append({"start": round(segment_start, 2), "end": round(segment_end, 2)})

    return segments


def export_segments(audio_file, segments, out_dir="segments"):
    os.makedirs(out_dir, exist_ok=True)
    audio = AudioSegment.from_wav(audio_file)
    for i, seg in enumerate(segments, start=1):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        clip = audio[start_ms:end_ms]
        filename = os.path.join(out_dir, f"segment_{i:02d}.wav")
        clip.export(filename, format="wav")
        print(f"Saved {filename}")

def main(input_file):
    # Step 1
    wav_file = extract_audio(input_file, "processed.wav")

    # Step 2
    audio, sample_rate = read_wave(wav_file)
    vad = webrtcvad.Vad(3)  # max aggressiveness
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 700, vad, frames, min_segment_s=0.3)

    # Save timestamps
    with open("segments.json", "w") as f:
        json.dump(segments, f, indent=2)
    print("Speech segments saved to segments.json")

    # Step 3
    export_segments(wav_file, segments, "segments")

if __name__ == "__main__":
    main("game analysing.mp4") 
