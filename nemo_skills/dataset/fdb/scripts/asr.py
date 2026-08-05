import argparse
import json
import os
import tempfile
from glob import glob

import nemo.collections.asr as nemo_asr
import soundfile as sf
from tqdm import tqdm


MODEL_NAME = ""


def get_time_aligned_transcription(data_path, task, is_stereo=False):
    if task == "input_only":
        audio_paths = sorted(glob(f"{data_path}/**/input.wav", recursive=True))
        audio_paths += sorted(glob(f"{data_path}/**/clean_input.wav", recursive=True))
    else:
        audio_paths = sorted(glob(f"{data_path}/**/output.wav", recursive=True))

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    ).cuda()

    for audio_path in tqdm(audio_paths):
        print(audio_path)
        waveform, sample_rate = sf.read(audio_path)

        if waveform.ndim > 1 and is_stereo:
            waveform = waveform[:, 1]
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        offset = 0.0
        if task == "user_interruption":
            meta_path = audio_path.replace(f"{MODEL_NAME}output.wav", "interrupt.json")
            with open(meta_path, "r") as metadata_file:
                interrupt_metadata = json.load(metadata_file)
            _, end_interrupt = interrupt_metadata[0]["timestamp"]
            offset = end_interrupt
            waveform = waveform[int(end_interrupt * sample_rate) :]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            sf.write(temp_wav.name, waveform, sample_rate)
            asr_outputs = asr_model.transcribe([temp_wav.name], timestamps=True)
        os.unlink(temp_wav.name)

        word_timestamps = asr_outputs[0].timestamp["word"]
        chunks = []
        text = ""
        for word_timestamp in word_timestamps:
            start_time = word_timestamp["start"] + offset
            end_time = word_timestamp["end"] + offset
            word = word_timestamp["word"]
            text += word + " "
            chunks.append({"text": word, "timestamp": [start_time, end_time]})

        result_path = os.path.splitext(audio_path)[0] + ".json"
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as result_file:
            json.dump({"text": text.strip(), "chunks": chunks}, result_file, indent=4)
        print(f"Transcription saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe full audio, audio after an interruption, or input audio"
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default="full",
        choices=["full", "user_interruption", "input_only"],
    )
    parser.add_argument("--stereo", action="store_true")
    args = parser.parse_args()
    get_time_aligned_transcription(args.root_dir, args.task, args.stereo)
