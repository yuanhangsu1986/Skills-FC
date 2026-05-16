########################
# Eval script for turn-taking (TT), user back-channeling (BC), user barge-in (BI)
# V2: Reads from output.jsonl format with inline audio paths and timestamps

import argparse
import json
import os
import re
import string

import torch
import torchaudio
from jiwer import process_characters, process_words
from nemo.collections import asr as nemo_asr
from tqdm import tqdm

INF_LATENCY = 9999.0
FRAME_SIZE_SEC = 0.08  # 80ms per frame
DEFAULT_SEGMENT_BUFFER_SEC = 0.5  # Default segment buffer for WER calculation

LLM_JUDGE_PROMPT_TEMPLATE = """
I need your help to evaluate the performance of a speech-to-speech model. The model receives speech input from the user and responds with speech output.
Your task is to rate the model's responses based on the provided user input transcription [User] and the model's output transcription [Agent].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user's query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user's question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don't contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user's question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user's query in a highly effective and efficient manner, providing exactly the information needed.

Below is the conversation transcript:

{conversation}

After evaluating, please output "Rating: X" where X is your score (1-5), without anything else.
""".strip()


def normalize_text_for_wer(text):
    """Normalize text for WER calculation: remove punctuation, timestamps, lowercase.

    Removes:
    - Timestamp tags: <|t|>, <$t$>, <|0.00|>, <$0.00$>
    - Sentence boundary tokens: <s>, </s>
    - Special tokens: <SPECIAL_N>, <SPEC_N>
    - All punctuation including hyphens
    - Extra whitespace
    """
    if not text:
        return ""

    # Remove timestamp tags with values: <|0.00|>, <$0.00$>
    text = re.sub(r"<\|[\d\.]+\|>", "", text)
    text = re.sub(r"<\$[\d\.]+\$>", "", text)

    # Remove sentence boundary tokens
    text = re.sub(r"</?s>", "", text)

    # Remove special tokens
    text = re.sub(r"<SPECIAL_\d+>", "", text)
    text = re.sub(r"<SPEC_\d+>", "", text)

    # Remove any remaining angle bracket tokens
    text = re.sub(r"<[^>]+>", "", text)

    # Remove punctuation including hyphens
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Lowercase and normalize whitespace
    text = text.lower()
    text = " ".join(text.split())

    return text


def time_to_frame(time_sec):
    """Convert time in seconds to frame index."""
    return int(time_sec / FRAME_SIZE_SEC)


def frame_to_time(frame_idx):
    """Convert frame index to time in seconds."""
    return frame_idx * FRAME_SIZE_SEC


def extract_text_from_alignment(frame_alignment, field, start_frame=None, end_frame=None):
    """Extract concatenated decoded text from frame_alignment within optional frame bounds.

    Args:
        frame_alignment: dict with frame_idx, asr_stream_decoded, agent_stream_decoded, etc.
        field: "asr_stream_decoded" or "agent_stream_decoded"
        start_frame: optional start frame index (inclusive)
        end_frame: optional end frame index (exclusive)

    Returns:
        Concatenated text from the specified field within bounds
    """
    if not frame_alignment or field not in frame_alignment:
        return ""

    tokens = frame_alignment[field]
    frame_indices = frame_alignment.get("frame_idx", list(range(len(tokens))))

    if start_frame is None and end_frame is None:
        # Return all tokens concatenated
        return "".join(tokens)

    # Filter by frame range
    result_tokens = []
    for i, fidx in enumerate(frame_indices):
        if start_frame is not None and fidx < start_frame:
            continue
        if end_frame is not None and fidx >= end_frame:
            continue
        if i < len(tokens):
            result_tokens.append(tokens[i])

    return "".join(result_tokens)


def get_debug_info(entry):
    """Extract debug_info from entry, handling both direct and nested formats."""
    original = entry.get("original_entry", entry)
    return original.get("debug_info", {})


def compute_user_speech_wer(
    user_segments, frame_alignment, user_transcripts, audio_duration, segment_buffer_sec=DEFAULT_SEGMENT_BUFFER_SEC
):
    """Compute WER for user speech recognition from model's ASR output vs ground truth.

    Args:
        user_segments: list of user segment dicts with start/end times (from VAD)
        frame_alignment: debug_info frame_alignment dict
        user_transcripts: dict mapping (start, end) tuples to ground truth ASR text
        audio_duration: total audio duration in seconds

    Returns:
        dict with:
            per_segment: list of dicts with ref/hyp text, S/I/D counts, WER per segment
            total_wer: WER calculated from sum of all errors / sum of ref words
            out_of_bounds_words: list of (word, onset_time) tuples for words outside segments
            out_of_bounds_word_ratio: words outside segments / audio_duration
    """
    if not frame_alignment or "asr_stream_decoded" not in frame_alignment:
        return {
            "per_segment": [],
            "total_wer": None,
            "total_ref_words": 0,
            "total_substitutions": 0,
            "total_insertions": 0,
            "total_deletions": 0,
            "out_of_bounds_words": [],
            "out_of_bounds_word_ratio": None,
            "error": "no_frame_alignment",
        }

    per_segment = []
    total_ref_words = 0
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0

    for seg in user_segments:
        start_frame = time_to_frame(seg["start"])
        # Extend end boundary for WER calculation to capture trailing words
        end_frame = time_to_frame(seg["end"] + segment_buffer_sec)

        # Get model's ASR output for this segment from frame_alignment (hypothesis)
        hyp_text = extract_text_from_alignment(frame_alignment, "asr_stream_decoded", start_frame, end_frame)
        hyp_normalized = normalize_text_for_wer(hyp_text)

        # Get ground truth ASR transcription (reference)
        seg_key = (seg["start"], seg["end"])
        ref_text = user_transcripts.get(seg_key, "")
        ref_normalized = normalize_text_for_wer(ref_text)

        ref_word_count = len(ref_normalized.split()) if ref_normalized else 0
        hyp_word_count = len(hyp_normalized.split()) if hyp_normalized else 0

        # Calculate detailed WER metrics using jiwer
        if ref_normalized or hyp_normalized:
            if ref_normalized and hyp_normalized:
                result = process_words(ref_normalized, hyp_normalized)
                subs = result.substitutions
                ins = result.insertions
                dels = result.deletions
            elif ref_normalized:
                # Hypothesis empty - all deletions
                subs, ins, dels = 0, 0, ref_word_count
            else:
                # Reference empty - all insertions
                subs, ins, dels = 0, hyp_word_count, 0

            segment_wer = (
                (subs + ins + dels) / ref_word_count if ref_word_count > 0 else (1.0 if hyp_word_count > 0 else 0.0)
            )

            total_ref_words += ref_word_count
            total_substitutions += subs
            total_insertions += ins
            total_deletions += dels

            per_segment.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "reference": ref_normalized,
                    "hypothesis": hyp_normalized,
                    "ref_words": ref_word_count,
                    "substitutions": subs,
                    "insertions": ins,
                    "deletions": dels,
                    "wer": segment_wer,
                }
            )

    # Calculate out-of-bounds words with timestamps
    all_tokens = frame_alignment.get("asr_stream_decoded", [])
    frame_indices = frame_alignment.get("frame_idx", list(range(len(all_tokens))))

    # Build set of frames that are within user segments (with buffer on both ends)
    in_segment_frames = set()
    for seg in user_segments:
        start_frame = time_to_frame(max(0, seg["start"] - segment_buffer_sec))
        end_frame = time_to_frame(seg["end"] + segment_buffer_sec)
        # Use end_frame + 1 to include the end frame (range is exclusive of end)
        for f in range(start_frame, end_frame + 1):
            in_segment_frames.add(f)

    # Collect words outside segments with their timestamps
    out_of_bounds_words = []
    current_word = []
    current_word_start_frame = None

    for i, fidx in enumerate(frame_indices):
        if fidx not in in_segment_frames and i < len(all_tokens):
            token = all_tokens[i]
            if token and token.strip():
                # Check if this token starts a new word (contains space or is start)
                if current_word_start_frame is None:
                    current_word_start_frame = fidx
                current_word.append(token)
        else:
            # End of out-of-segment region - flush accumulated tokens
            if current_word:
                word_text = normalize_text_for_wer("".join(current_word))
                if word_text:
                    for w in word_text.split():
                        out_of_bounds_words.append(
                            {
                                "word": w,
                                "onset_time": frame_to_time(current_word_start_frame),
                            }
                        )
                current_word = []
                current_word_start_frame = None

    # Flush any remaining tokens
    if current_word:
        word_text = normalize_text_for_wer("".join(current_word))
        if word_text:
            for w in word_text.split():
                out_of_bounds_words.append(
                    {
                        "word": w,
                        "onset_time": frame_to_time(current_word_start_frame),
                    }
                )

    out_of_bounds_word_count = len(out_of_bounds_words)
    out_of_bounds_ratio = out_of_bounds_word_count / audio_duration if audio_duration > 0 else 0

    # Calculate total WER from summed errors
    total_errors = total_substitutions + total_insertions + total_deletions
    total_wer = total_errors / total_ref_words if total_ref_words > 0 else None

    return {
        "per_segment": per_segment,
        "total_wer": total_wer,
        "total_ref_words": total_ref_words,
        "total_substitutions": total_substitutions,
        "total_insertions": total_insertions,
        "total_deletions": total_deletions,
        "out_of_bounds_words": out_of_bounds_words,
        "out_of_bounds_word_count": out_of_bounds_word_count,
        "out_of_bounds_word_ratio": out_of_bounds_ratio,
    }


def compute_wer_with_details(reference, hypothesis, ignore_trailing_deletions=True):
    """Compute WER with detailed S/I/D counts, optionally ignoring trailing deletions.

    When TTS is truncated due to user interruption, the hypothesis may be
    shorter than the reference. If ignore_trailing_deletions=True, we ignore
    these trailing deletions.

    Args:
        reference: normalized reference text (speech2text model output)
        hypothesis: normalized hypothesis text (TTS audio transcribed)
        ignore_trailing_deletions: whether to ignore trailing deletions (TTS truncation)

    Returns:
        dict with wer, subs, ins, dels, ref_words, truncation_detected, truncated_words
    """
    ref_words = reference.split() if reference else []
    hyp_words = hypothesis.split() if hypothesis else []
    ref_word_count = len(ref_words)
    hyp_word_count = len(hyp_words)

    if not reference and not hypothesis:
        return {
            "wer": 0.0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "ref_words": 0,
            "truncation_detected": False,
            "truncated_words": [],
        }
    if not reference:
        return {
            "wer": 1.0,
            "substitutions": 0,
            "insertions": hyp_word_count,
            "deletions": 0,
            "ref_words": 0,
            "truncation_detected": False,
            "truncated_words": [],
        }
    if not hypothesis:
        return {
            "wer": 1.0,
            "substitutions": 0,
            "insertions": 0,
            "deletions": ref_word_count,
            "ref_words": ref_word_count,
            "truncation_detected": True,
            "truncated_words": ref_words,
        }

    # Use jiwer to get detailed metrics
    result = process_words(reference, hypothesis)
    subs = result.substitutions
    ins = result.insertions
    dels = result.deletions

    truncation_detected = False
    truncated_words = []

    # Check for trailing deletion (truncation) if hypothesis is shorter
    if ignore_trailing_deletions and hyp_word_count < ref_word_count:
        # Try trimming reference to hypothesis length
        trimmed_ref = " ".join(ref_words[:hyp_word_count])
        if trimmed_ref:
            trimmed_result = process_words(trimmed_ref, hypothesis)
            trimmed_errors = trimmed_result.substitutions + trimmed_result.insertions + trimmed_result.deletions
            original_errors = subs + ins + dels

            # If trimming significantly reduces errors, it's truncation
            if trimmed_errors < original_errors * 0.8:
                truncation_detected = True
                truncated_words = ref_words[hyp_word_count:]
                subs = trimmed_result.substitutions
                ins = trimmed_result.insertions
                dels = trimmed_result.deletions
                ref_word_count = hyp_word_count  # Adjust ref count for WER calculation

    total_errors = subs + ins + dels
    wer_value = total_errors / ref_word_count if ref_word_count > 0 else 0.0

    return {
        "wer": wer_value,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "ref_words": ref_word_count,
        "truncation_detected": truncation_detected,
        "truncated_words": truncated_words,
    }


def compute_agent_speech_quality(
    agent_segments, frame_alignment, agent_transcripts, segment_buffer_sec=DEFAULT_SEGMENT_BUFFER_SEC
):
    """Compute WER/CER for agent speech: TTS output vs speech2text model output.

    Args:
        agent_segments: list of agent segment dicts with start/end times
        frame_alignment: debug_info frame_alignment dict
        agent_transcripts: dict mapping (start, end) tuples to TTS audio ASR transcription

    Returns:
        dict with per_segment details including ref/hyp text, S/I/D counts, WER/CER
        Total WER/CER calculated from sum of all errors / sum of ref words/chars
    """
    if not frame_alignment or "agent_stream_decoded" not in frame_alignment:
        return {
            "per_segment": [],
            "total_wer": None,
            "total_cer": None,
            "total_ref_words": 0,
            "total_ref_chars": 0,
            "total_word_substitutions": 0,
            "total_word_insertions": 0,
            "total_word_deletions": 0,
            "total_char_substitutions": 0,
            "total_char_insertions": 0,
            "total_char_deletions": 0,
            "truncation_events": 0,
            "truncated_words": [],
            "error": "no_frame_alignment",
        }

    per_segment = []
    total_ref_words = 0
    total_ref_chars = 0
    total_word_subs = 0
    total_word_ins = 0
    total_word_dels = 0
    total_char_subs = 0
    total_char_ins = 0
    total_char_dels = 0
    truncation_events = 0
    all_truncated_words = []

    for seg in agent_segments:
        start_frame = time_to_frame(seg["start"])
        # Extend end boundary for WER calculation to capture trailing words
        end_frame = time_to_frame(seg["end"] + segment_buffer_sec)

        # Reference: speech2text model output from frame_alignment
        ref_text = extract_text_from_alignment(frame_alignment, "agent_stream_decoded", start_frame, end_frame)
        ref_normalized = normalize_text_for_wer(ref_text)

        # Hypothesis: ASR transcription of TTS audio output
        seg_key = (seg["start"], seg["end"])
        hyp_text = agent_transcripts.get(seg_key, "")
        hyp_normalized = normalize_text_for_wer(hyp_text)

        if ref_normalized or hyp_normalized:
            # Compute WER with truncation handling
            wer_result = compute_wer_with_details(ref_normalized, hyp_normalized, ignore_trailing_deletions=True)

            if wer_result["truncation_detected"]:
                truncation_events += 1
                all_truncated_words.extend(wer_result["truncated_words"])

            total_ref_words += wer_result["ref_words"]
            total_word_subs += wer_result["substitutions"]
            total_word_ins += wer_result["insertions"]
            total_word_dels += wer_result["deletions"]

            # Compute CER (character error rate)
            # If truncation was detected, compute CER only on the matched portion
            if wer_result["truncation_detected"] and wer_result["truncated_words"]:
                # Trim reference to match the non-truncated portion
                ref_words = ref_normalized.split()
                matched_ref_words = ref_words[: len(ref_words) - len(wer_result["truncated_words"])]
                ref_for_cer = " ".join(matched_ref_words)
            else:
                ref_for_cer = ref_normalized

            ref_chars = len(ref_for_cer) if ref_for_cer else 0
            hyp_chars = len(hyp_normalized) if hyp_normalized else 0

            if ref_for_cer and hyp_normalized:
                cer_result = process_characters(ref_for_cer, hyp_normalized)
                char_subs = cer_result.substitutions
                char_ins = cer_result.insertions
                char_dels = cer_result.deletions
            elif ref_for_cer:
                char_subs, char_ins, char_dels = 0, 0, ref_chars
            else:
                char_subs, char_ins, char_dels = 0, hyp_chars, 0

            total_ref_chars += ref_chars
            total_char_subs += char_subs
            total_char_ins += char_ins
            total_char_dels += char_dels

            segment_cer = (
                (char_subs + char_ins + char_dels) / ref_chars if ref_chars > 0 else (1.0 if hyp_chars > 0 else 0.0)
            )

            per_segment.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "reference": ref_normalized,
                    "hypothesis": hyp_normalized,
                    "ref_words": wer_result["ref_words"],
                    "word_substitutions": wer_result["substitutions"],
                    "word_insertions": wer_result["insertions"],
                    "word_deletions": wer_result["deletions"],
                    "wer": wer_result["wer"],
                    "ref_chars": ref_chars,
                    "char_substitutions": char_subs,
                    "char_insertions": char_ins,
                    "char_deletions": char_dels,
                    "cer": segment_cer,
                    "truncation_detected": wer_result["truncation_detected"],
                    "truncated_words": wer_result["truncated_words"],
                }
            )

    # Calculate total WER/CER from summed errors
    total_word_errors = total_word_subs + total_word_ins + total_word_dels
    total_wer = total_word_errors / total_ref_words if total_ref_words > 0 else None

    total_char_errors = total_char_subs + total_char_ins + total_char_dels
    total_cer = total_char_errors / total_ref_chars if total_ref_chars > 0 else None

    return {
        "per_segment": per_segment,
        "total_wer": total_wer,
        "total_cer": total_cer,
        "total_ref_words": total_ref_words,
        "total_ref_chars": total_ref_chars,
        "total_word_substitutions": total_word_subs,
        "total_word_insertions": total_word_ins,
        "total_word_deletions": total_word_dels,
        "total_char_substitutions": total_char_subs,
        "total_char_insertions": total_char_ins,
        "total_char_deletions": total_char_dels,
        "truncation_events": truncation_events,
        "truncated_words": all_truncated_words,
    }


def compute_tts_hallucinations(
    agent_segments, frame_alignment, agent_transcripts, segment_buffer_sec=DEFAULT_SEGMENT_BUFFER_SEC
):
    """Detect TTS hallucinations: words in TTS output not present in speech2text output.

    Args:
        agent_segments: list of agent segment dicts with start/end times
        frame_alignment: debug_info frame_alignment dict
        agent_transcripts: dict mapping (start, end) tuples to TTS audio ASR transcription

    Returns:
        dict with hallucinated words with onset times per segment and overall
    """
    if not frame_alignment or "agent_stream_decoded" not in frame_alignment:
        return {
            "per_segment": [],
            "hallucinations": [],
            "hallucination_rate": None,
            "error": "no_frame_alignment",
        }

    per_segment = []
    all_hallucinations = []
    total_hyp_words = 0

    for seg in agent_segments:
        start_frame = time_to_frame(seg["start"])
        # Extend end boundary for consistency with WER calculation
        end_frame = time_to_frame(seg["end"] + segment_buffer_sec)
        seg_start_time = seg["start"]

        # Reference: speech2text model output from frame_alignment
        ref_text = extract_text_from_alignment(frame_alignment, "agent_stream_decoded", start_frame, end_frame)
        ref_normalized = normalize_text_for_wer(ref_text)
        ref_words = set(ref_normalized.split()) if ref_normalized else set()

        # Hypothesis: ASR transcription of TTS audio output
        seg_key = (seg["start"], seg["end"])
        hyp_text = agent_transcripts.get(seg_key, "")
        hyp_normalized = normalize_text_for_wer(hyp_text)
        hyp_words = hyp_normalized.split() if hyp_normalized else []

        total_hyp_words += len(hyp_words)

        # Find hallucinated words with estimated onset times
        # Estimate onset time based on word position within segment
        segment_duration = seg["end"] - seg["start"]
        segment_hallucinations = []

        for i, word in enumerate(hyp_words):
            if word not in ref_words:
                # Estimate onset time based on word position
                word_ratio = i / len(hyp_words) if hyp_words else 0
                estimated_onset = seg_start_time + word_ratio * segment_duration

                hallucination_entry = {
                    "word": word,
                    "onset_time": round(estimated_onset, 3),
                    "segment_start": seg_start_time,
                    "segment_end": seg["end"],
                }
                segment_hallucinations.append(hallucination_entry)
                all_hallucinations.append(hallucination_entry)

        per_segment.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "count": len(segment_hallucinations),
                "hallucinations": segment_hallucinations,
            }
        )

    hallucination_rate = len(all_hallucinations) / total_hyp_words if total_hyp_words > 0 else 0.0

    return {
        "per_segment": per_segment,
        "hallucinations": all_hallucinations,
        "total_hallucinated": len(all_hallucinations),
        "total_agent_words": total_hyp_words,
        "hallucination_rate": hallucination_rate,
    }


def compute_token_balance(frame_alignment):
    """Compute token balance metrics from frame alignment.

    Both streams use <s> for BOS and </s> for EOS sentence boundary tokens.

    Balance metric is normalized to [-1, 1]:
    - 0 = perfectly balanced
    - Positive = more BOS tokens than EOS tokens (incomplete utterances)
    - Negative = more EOS tokens than BOS tokens

    Args:
        frame_alignment: debug_info frame_alignment dict

    Returns:
        dict with token counts and balance metrics for both streams
    """
    if not frame_alignment:
        return {
            "agent_bos_count": 0,
            "agent_eos_count": 0,
            "agent_balance": 0.0,
            "user_bos_count": 0,
            "user_eos_count": 0,
            "user_balance": 0.0,
            "error": "no_frame_alignment",
        }

    # Count agent stream BOS/EOS tokens (<s> and </s>)
    agent_tokens = frame_alignment.get("agent_stream_decoded", [])
    agent_text = "".join(agent_tokens) if agent_tokens else ""

    # Agent BOS: <s>
    agent_bos_count = agent_text.count("<s>")
    # Agent EOS: </s>
    agent_eos_count = agent_text.count("</s>")

    # Calculate agent balance
    agent_total = agent_bos_count + agent_eos_count
    agent_balance = (agent_bos_count - agent_eos_count) / agent_total if agent_total > 0 else 0.0

    # Count user stream (ASR) BOS/EOS tokens (<s> and </s>)
    user_tokens = frame_alignment.get("asr_stream_decoded", [])
    user_text = "".join(user_tokens) if user_tokens else ""

    # User BOS: <s>
    user_bos_count = user_text.count("<s>")
    # User EOS: </s>
    user_eos_count = user_text.count("</s>")

    # Calculate user balance
    user_total = user_bos_count + user_eos_count
    user_balance = (user_bos_count - user_eos_count) / user_total if user_total > 0 else 0.0

    return {
        "agent_bos_count": agent_bos_count,
        "agent_eos_count": agent_eos_count,
        "agent_balance": agent_balance,
        "user_bos_count": user_bos_count,
        "user_eos_count": user_eos_count,
        "user_balance": user_balance,
    }


def parse_float_list(arg):
    """Parse a string representation of a list of floats."""
    if arg.startswith("[") and arg.endswith("]"):
        arg = arg[1:-1]
    return [float(x.strip()) for x in arg.split(",")]


def remove_special_symbols(text):
    """Remove special symbols like <SPECIAL_12> from text."""
    text = re.sub(r"<SPECIAL_\d+>", "", text)
    return text.strip()


def parse_timestamped_text(text_with_timestamps, audio_duration=None):
    """Parse BOS <|t|> and EOS <$t$> timestamps from text.

    Args:
        text_with_timestamps: Text containing BOS <|t|> and EOS <$t$> markers
        audio_duration: Optional audio duration to use as end time for last segment without EOS
    """
    bos_pattern = r"<\|([\d\.]+)\|>"
    eos_pattern = r"<\$([\d\.]+)\$>"

    bos_matches = list(re.finditer(bos_pattern, text_with_timestamps))
    eos_matches = list(re.finditer(eos_pattern, text_with_timestamps))

    bos_timestamps = [float(match.group(1)) for match in bos_matches]
    eos_timestamps = [float(match.group(1)) for match in eos_matches]

    agent_segments = []

    if bos_timestamps and eos_timestamps:
        for i, start_time in enumerate(bos_timestamps):
            end_time = None
            eos_idx = None
            for j, eos_time in enumerate(eos_timestamps):
                if eos_time > start_time:
                    end_time = eos_time
                    eos_idx = j
                    break

            text = ""
            if end_time is not None and i < len(bos_matches) and eos_idx < len(eos_matches):
                bos_end_pos = bos_matches[i].end()
                eos_start_pos = eos_matches[eos_idx].start()
                text = text_with_timestamps[bos_end_pos:eos_start_pos].strip()
            elif i < len(bos_matches):
                # No matching EOS - extract text from BOS to next BOS or end of string
                bos_end_pos = bos_matches[i].end()
                if i < len(bos_matches) - 1:
                    next_bos_start_pos = bos_matches[i + 1].start()
                    text = text_with_timestamps[bos_end_pos:next_bos_start_pos].strip()
                else:
                    # Last BOS without EOS - extract to end of string
                    text = text_with_timestamps[bos_end_pos:].strip()
                    # Remove any trailing EOS tags that might be present
                    text = re.sub(r"<\$[\d\.]+\$>", "", text).strip()

            if end_time is not None:
                agent_segments.append({"start": start_time, "end": end_time, "text": text})
            else:
                # No EOS - use audio duration if available, otherwise default to +5s
                fallback_end = audio_duration if audio_duration is not None else start_time + 5.0
                agent_segments.append({"start": start_time, "end": fallback_end, "text": text})
    elif bos_timestamps:
        for i, timestamp in enumerate(bos_timestamps):
            text = ""
            if i < len(bos_matches):
                bos_end_pos = bos_matches[i].end()
                if i < len(bos_matches) - 1:
                    next_bos_start_pos = bos_matches[i + 1].start()
                    text = text_with_timestamps[bos_end_pos:next_bos_start_pos].strip()
                else:
                    text = text_with_timestamps[bos_end_pos:].strip()

            if i < len(bos_timestamps) - 1:
                agent_segments.append({"start": timestamp, "end": bos_timestamps[i + 1], "text": text})
            else:
                # Last segment - use audio duration if available, otherwise default to +5s
                fallback_end = audio_duration if audio_duration is not None else timestamp + 5.0
                agent_segments.append({"start": timestamp, "end": fallback_end, "text": text})

    return agent_segments


def load_results_jsonl(results_dir, prefer_cached=True):
    """Load entries from output.jsonl or output_with_eval.jsonl in results directory.

    Args:
        results_dir: Directory containing the JSONL files
        prefer_cached: If True, prefer output_with_eval.jsonl if it exists (has cached segmentation/transcription)
    """
    cached_path = os.path.join(results_dir, "output_with_eval.jsonl")
    original_path = os.path.join(results_dir, "output.jsonl")

    # Prefer cached file if it exists and prefer_cached is True
    if prefer_cached and os.path.exists(cached_path):
        jsonl_path = cached_path
        print(f"Using cached results file: {jsonl_path}")
    else:
        jsonl_path = original_path

    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                entries.append(data)

    return entries


def get_audio_dir(results_dir):
    """Get audio directory path from results directory."""
    return os.path.join(results_dir, "audio")


def _get_original_entry(entry):
    """Get the original entry, unwrapping if this is from output_with_eval.jsonl."""
    return entry.get("original_entry", entry)


def get_item_id(entry):
    """Extract item ID from entry."""
    entry = _get_original_entry(entry)
    audio_path = entry.get("audio_path", "")
    if audio_path:
        filename = os.path.basename(audio_path)
        if filename.endswith(".flac"):
            filename = filename[:-5]
        elif filename.endswith(".wav"):
            filename = filename[:-4]
        return filename
    return str(hash(json.dumps(entry, sort_keys=True)))[:8]


def get_audio_path(entry, audio_dir=None):
    """Get audio file path from entry, optionally remapping to audio_dir."""
    original = _get_original_entry(entry)
    audio = original.get("audio", {})
    if isinstance(audio, dict):
        path = audio.get("path", None)
        if path and audio_dir:
            # Remap to local audio directory
            filename = os.path.basename(path)
            return os.path.join(audio_dir, filename)
        return path
    return None


def get_generation_text(entry):
    """Get timestamped generation text from entry."""
    original = _get_original_entry(entry)
    generation = original.get("generation", "")
    if generation:
        return generation
    audio = original.get("audio", {})
    if isinstance(audio, dict):
        return audio.get("transcript", "")
    return ""


def get_cached_eval_data(entry):
    """Check if entry has cached segmentation and transcription from previous eval run.

    Returns tuple: (segmentation_data, transcription_data) or (None, None) if not cached.
    """
    # Check for eval data in the entry (from output_with_eval.jsonl format)
    segmentation = entry.get("segmentation", None)
    transcription = entry.get("transcription", None)

    if segmentation and transcription:
        return segmentation, transcription

    # Also check if this is an original_entry wrapper format
    original = entry.get("original_entry", None)
    if original:
        # This entry was already processed - use the eval data from parent
        segmentation = entry.get("segmentation", None)
        transcription = entry.get("transcription", None)
        if segmentation and transcription:
            return segmentation, transcription

    return None, None


def parse_cached_transcripts(transcription_dict):
    """Convert cached transcription format back to tuple-keyed dict.

    Cached format: {"1.234-5.678": "text"}
    Returns: {(1.234, 5.678): "text"}
    """
    result = {}
    for key, text in transcription_dict.items():
        try:
            start_str, end_str = key.split("-")
            result[(float(start_str), float(end_str))] = text
        except (ValueError, AttributeError):
            continue
    return result


def is_stopped_by_backchannel(agent_speech_segments, end_times, delay=0.99):
    """Check if agent's speech was interrupted by user backchanneling."""
    if not end_times or not agent_speech_segments:
        return [False] * len(end_times)

    agent_speech_segments = sorted(agent_speech_segments, key=lambda x: x["start"])

    bc_failure = []
    for t in end_times:
        if t == 0:
            bc_failure.append(False)
            continue

        overlapping = False
        for segment in agent_speech_segments:
            if t >= segment["start"] and t <= segment["end"]:
                overlapping = True
                agent_delayed_stop_time = t + delay
                agent_still_speaking = any(
                    s["start"] <= agent_delayed_stop_time <= s["end"] for s in agent_speech_segments
                )
                is_interrupted = not agent_still_speaking
                bc_failure.append(is_interrupted)
                break

        if not overlapping:
            bc_failure.append(False)

    return bc_failure


def find_user_barge_ins(user_turns, agent_turns, threshold_seconds=0.5):
    """Find user barge-in events during agent speech."""
    i, j = 0, 0
    success_barge_ins = []
    failed_barge_ins = []

    while i < len(user_turns) and j < len(agent_turns):
        u_start, u_end = user_turns[i]["start"], user_turns[i]["end"]
        a_start, a_end = agent_turns[j]["start"], agent_turns[j]["end"]

        if u_start > a_start and u_start < a_end:
            stop_duration_ms = round((a_end - u_start) * 1000)

            barge_in_info = {"stop_duration_ms": stop_duration_ms, "user": user_turns[i], "agent": agent_turns[j]}

            if stop_duration_ms < threshold_seconds * 1000:
                success_barge_ins.append(barge_in_info)
            else:
                failed_barge_ins.append(barge_in_info)

        if u_end < a_end:
            i += 1
        else:
            j += 1

    return success_barge_ins, failed_barge_ins


def init_vad_model():
    """Initialize Silero VAD model."""
    vad_model, utils = torch.hub.load("snakers4/silero-vad", model="silero_vad", force_reload=False)
    vad_model = vad_model.to("cuda")
    get_speech_timestamps, _, _, _, _ = utils
    return vad_model, get_speech_timestamps


def init_asr_model(model_name="nvidia/parakeet-tdt-0.6b-v2"):
    """Initialize NeMo ASR model."""
    print(f"Loading ASR model: {model_name}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name).cuda()
    print("ASR model loaded successfully.")
    return asr_model


def transcribe_segment(audio, start_time, end_time, sample_rate, asr_model, temp_dir="/tmp"):
    """Transcribe a specific segment of audio using NeMo ASR."""
    import tempfile

    try:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment_audio = audio[:, start_sample:end_sample]

        if segment_audio.shape[1] < 160:
            return "", 0.0

        if sample_rate != 16000:
            segment_audio = torchaudio.functional.resample(segment_audio, sample_rate, 16000)
            sample_rate = 16000

        if segment_audio.shape[0] > 1:
            segment_audio = torch.mean(segment_audio, dim=0, keepdim=True)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir) as tmp_file:
            tmp_path = tmp_file.name
            torchaudio.save(tmp_path, segment_audio.cpu(), sample_rate)

        asr_outputs = asr_model.transcribe([tmp_path], timestamps=True)
        os.remove(tmp_path)

        if not asr_outputs:
            return "", 0.0

        result = asr_outputs[0]
        text = result.text if hasattr(result, "text") else ""

        end_timestamp = 0.0
        if hasattr(result, "timestamp") and result.timestamp and "word" in result.timestamp:
            word_timestamps = result.timestamp["word"]
            if word_timestamps and len(word_timestamps) > 0:
                last_word = word_timestamps[-1]
                if isinstance(last_word, dict) and "end" in last_word:
                    end_timestamp = last_word["end"]
        elif hasattr(result, "end_time"):
            end_timestamp = result.end_time

        return text.strip(), end_timestamp

    except Exception as e:
        print(f"Error transcribing segment [{start_time:.3f}s - {end_time:.3f}s]: {str(e)}")
        return "", 0.0


def compute_barge_in_metrics(success_barge_ins, failed_barge_ins):
    """Compute barge-in metrics including success rate and counts."""
    total_barge_ins = len(success_barge_ins) + len(failed_barge_ins)
    success_count = len(success_barge_ins)

    metrics = {"total_count": total_barge_ins, "success_count": success_count, "has_barge_ins": total_barge_ins > 0}

    if metrics["has_barge_ins"]:
        metrics["success_rate"] = (success_count / total_barge_ins) * 100

    if success_count > 0:
        metrics["avg_latency_ms"] = sum(bi["stop_duration_ms"] for bi in success_barge_ins) / success_count

    return metrics


def compute_turn_taking_metrics(
    agent_segments, user_segments, tt_latency_threshold_sec, tt_precision_buffer_sec, tt_recall_buffer_sec
):
    """Compute turn-taking metrics using precision and recall."""
    if not agent_segments or not user_segments:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "avg_latency": INF_LATENCY,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }

    tp = 0
    fp = 0
    fn = 0
    tp_latencies = []

    for agent_seg in agent_segments:
        found_tp = False
        min_latency = INF_LATENCY

        for user_seg in user_segments:
            gap = agent_seg["start"] - user_seg["end"]
            if (
                gap >= -tt_precision_buffer_sec
                and gap <= tt_latency_threshold_sec
                and agent_seg["start"] >= user_seg["start"]
            ):
                found_tp = True
                min_latency = min(min_latency, max(gap, 0))

        if found_tp:
            tp += 1
            tp_latencies.append(min_latency)
        else:
            fp += 1

    for user_seg in user_segments:
        found_tp = False
        for agent_seg in agent_segments:
            gap = agent_seg["start"] - user_seg["end"]
            if gap >= -tt_recall_buffer_sec and gap <= tt_latency_threshold_sec:
                found_tp = True
                break
        if not found_tp:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency = sum(tp_latencies) / len(tp_latencies) if tp_latencies else INF_LATENCY

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency": avg_latency,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def print_detailed_utterance(metrics_dict):
    """Print detailed information for a single utterance."""
    user_transcripts = metrics_dict.get("user_transcripts", {})
    agent_transcripts = metrics_dict.get("agent_transcripts", {})

    print(f"\n{'=' * 80}")
    print(f"Utterance: {metrics_dict['item_id']}")
    print(f"{'=' * 80}")

    print("\nMetrics:")
    print("  Turn-taking:")
    print(f"    - Precision: {metrics_dict['tt_precision']:.3f}")
    print(f"    - Recall: {metrics_dict['tt_recall']:.3f}")
    print(f"    - F1: {metrics_dict['tt_f1']:.3f}")
    print(f"    - Latency: {metrics_dict['tt_latency']:.3f}s ({metrics_dict['tt_latency'] * 1000:.1f}ms)")

    if metrics_dict["barge_in_metrics"]["has_barge_ins"]:
        print("  Barge-in:")
        print(
            f"    - Success rate: {metrics_dict['barge_in_metrics']['success_rate']:.1f}% ({metrics_dict['barge_in_metrics']['success_count']}/{metrics_dict['barge_in_metrics']['total_count']})"
        )
        if "avg_latency_ms" in metrics_dict["barge_in_metrics"]:
            print(f"    - Average latency: {metrics_dict['barge_in_metrics']['avg_latency_ms']:.1f}ms")
    else:
        print("  Barge-in: No barge-ins detected")

    print("\nConversation flow:")
    all_segments = []
    for seg in metrics_dict["user_segments"]:
        all_segments.append({"type": "User", "start": seg["start"], "end": seg["end"]})
    for seg in metrics_dict["agent_segments"]:
        all_segments.append({"type": "Agent", "start": seg["start"], "end": seg["end"]})

    all_segments.sort(key=lambda x: x["start"])

    for seg in all_segments:
        seg_key = (seg["start"], seg["end"])
        duration = seg["end"] - seg["start"]

        if seg["type"] == "User":
            transcript = user_transcripts.get(seg_key, "")
            if transcript:
                print(
                    f"  \033[94mUser\033[0m  [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s): {transcript}"
                )
            else:
                print(f"  \033[94mUser\033[0m  [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s)")
        else:
            transcript = agent_transcripts.get(seg_key, "")
            if transcript:
                cleaned_text = remove_special_symbols(transcript)
                print(
                    f"  \033[92mAgent\033[0m [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s): {cleaned_text}"
                )
            else:
                print(f"  \033[92mAgent\033[0m [{seg['start']:7.3f}s - {seg['end']:7.3f}s] ({duration:.3f}s)")

    if metrics_dict["barge_in_metrics"]["has_barge_ins"]:
        print("\nBarge-in events:")
        if metrics_dict["success_barge_ins"]:
            print(f"  Successful ({len(metrics_dict['success_barge_ins'])}):")
            for bi in metrics_dict["success_barge_ins"]:
                print(f"    User barged in at {bi['user']['start']:.3f}s during agent speech")
                print(f"    Agent stopped in {bi['stop_duration_ms']:.1f}ms")

        if metrics_dict["failed_barge_ins"]:
            print(f"  Failed ({len(metrics_dict['failed_barge_ins'])}):")
            for bi in metrics_dict["failed_barge_ins"]:
                print(f"    User barged in at {bi['user']['start']:.3f}s during agent speech")
                print(f"    Agent took {bi['stop_duration_ms']:.1f}ms to stop (too slow)")

    print(f"{'=' * 80}\n")


def print_bottom_percentile_utterances(all_metrics, percentile=5):
    """Print utterances in the bottom percentile for barge-in accuracy and turn-taking recall."""
    import numpy as np

    if not all_metrics:
        print("No metrics to analyze.")
        return

    print(f"\n{'#' * 80}")
    print(f"BOTTOM {percentile}% PERCENTILE ANALYSIS")
    print(f"{'#' * 80}\n")

    utterances_with_barge_ins = [m for m in all_metrics if m["barge_in_metrics"]["has_barge_ins"]]
    barge_in_rates = [m["barge_in_metrics"]["success_rate"] for m in utterances_with_barge_ins]
    tt_recalls = [m["tt_recall"] for m in all_metrics]

    if barge_in_rates:
        barge_in_threshold = np.percentile(barge_in_rates, percentile)
        print(f"Barge-in success rate {percentile}th percentile threshold: {barge_in_threshold:.1f}%")
        print(f"  (Based on {len(utterances_with_barge_ins)} utterances with barge-ins)\n")
    else:
        barge_in_threshold = None
        print("No utterances with barge-ins detected.\n")

    tt_recall_threshold = np.percentile(tt_recalls, percentile)
    print(f"Turn-taking recall {percentile}th percentile threshold: {tt_recall_threshold:.3f}")
    print(f"  (Based on {len(all_metrics)} utterances)\n")

    low_barge_in_utterances = []
    if barge_in_threshold is not None:
        low_barge_in_utterances = [
            m for m in utterances_with_barge_ins if m["barge_in_metrics"]["success_rate"] <= barge_in_threshold
        ]
        low_barge_in_utterances.sort(key=lambda m: m["barge_in_metrics"]["success_rate"])

    low_tt_recall_utterances = [m for m in all_metrics if m["tt_recall"] <= tt_recall_threshold]
    low_tt_recall_utterances.sort(key=lambda m: m["tt_recall"])

    if low_barge_in_utterances:
        print(f"\n{'-' * 80}")
        print(f"UTTERANCES WITH LOW BARGE-IN SUCCESS RATE (≤ {barge_in_threshold:.1f}%)")
        print(f"Found {len(low_barge_in_utterances)} utterance(s)")
        print(f"{'-' * 80}")

        for m in low_barge_in_utterances:
            print_detailed_utterance(m)

    if low_tt_recall_utterances:
        print(f"\n{'-' * 80}")
        print(f"UTTERANCES WITH LOW TURN-TAKING RECALL (≤ {tt_recall_threshold:.3f})")
        print(f"Found {len(low_tt_recall_utterances)} utterance(s)")
        print(f"{'-' * 80}")

        for m in low_tt_recall_utterances:
            print_detailed_utterance(m)

    print(f"\n{'#' * 80}")
    print(f"END OF BOTTOM {percentile}% PERCENTILE ANALYSIS")
    print(f"{'#' * 80}\n")


def print_metrics(metrics_dict, verbose=False):
    """Print all evaluation metrics for a conversation."""
    if not verbose:
        return

    barge_in_stats = []
    if metrics_dict["barge_in_metrics"]["has_barge_ins"]:
        barge_in_stats.append(
            f"   - Barge-in success rate: {metrics_dict['barge_in_metrics']['success_rate']:.1f}% ({metrics_dict['barge_in_metrics']['success_count']}/{metrics_dict['barge_in_metrics']['total_count']})"
        )
        if "avg_latency_ms" in metrics_dict["barge_in_metrics"]:
            barge_in_stats.append(
                f"   - Average barge-in latency: {metrics_dict['barge_in_metrics']['avg_latency_ms']:.1f} ms"
            )
    else:
        barge_in_stats.append("   - No barge-ins detected")

    segment_info = ""
    if verbose:
        user_transcripts = metrics_dict.get("user_transcripts", {})
        agent_transcripts = metrics_dict.get("agent_transcripts", {})

        all_segments = []
        for seg in metrics_dict["user_segments"]:
            all_segments.append({"type": "User", "start": seg["start"], "end": seg["end"]})
        for seg in metrics_dict["agent_segments"]:
            all_segments.append({"type": "Agent", "start": seg["start"], "end": seg["end"]})

        all_segments.sort(key=lambda x: x["start"])

        user_to_agent_latencies = {}
        for user_seg in metrics_dict["user_segments"]:
            next_agent = None
            min_latency = float("inf")
            for agent_seg in metrics_dict["agent_segments"]:
                if agent_seg["start"] >= user_seg["end"]:
                    latency = agent_seg["start"] - user_seg["end"]
                    if latency < min_latency:
                        min_latency = latency
                        next_agent = agent_seg

            if next_agent:
                user_to_agent_latencies[(user_seg["start"], user_seg["end"])] = min_latency

        def format_segment(seg):
            seg_key = (seg["start"], seg["end"])
            transcript = ""

            if seg["type"] == "User":
                if seg_key in user_transcripts:
                    transcript = f" ({user_transcripts[seg_key]})"

                if seg_key in user_to_agent_latencies:
                    latency = user_to_agent_latencies[seg_key]
                    return f"   \033[94m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s], \033[93m{latency:.3f}s\033[0m{transcript}"
                else:
                    return (
                        f"   \033[94m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s]{transcript}"
                    )
            else:
                if seg_key in agent_transcripts:
                    cleaned_text = remove_special_symbols(agent_transcripts[seg_key])
                    words = cleaned_text.split()
                    estimate_sec_per_word = 0.3
                    estimated_duration = len(words) * estimate_sec_per_word
                    transcript = f" ({cleaned_text}) [\033[95mest. {estimated_duration:.2f}s\033[0m]"
                return f"   \033[92m{seg['type']:5s}\033[0m [{seg['start']:6.3f}s - {seg['end']:6.3f}s]{transcript}"

        segments_str = "\n".join(format_segment(seg) for seg in all_segments)

        barge_in_segments = []
        if metrics_dict["barge_in_metrics"]["has_barge_ins"]:
            if metrics_dict["success_barge_ins"]:
                barge_in_segments.append("   Successful barge-ins:")
                for bi in metrics_dict["success_barge_ins"]:
                    barge_in_segments.append(f"     User: [{bi['user']['start']:.3f}s - {bi['user']['end']:.3f}s]")
                    barge_in_segments.append(f"     Agent: [{bi['agent']['start']:.3f}s - {bi['agent']['end']:.3f}s]")
                    barge_in_segments.append(f"     Stop duration: {bi['stop_duration_ms']:.3f} ms")

            if metrics_dict["failed_barge_ins"]:
                barge_in_segments.append("   Failed barge-ins:")
                for bi in metrics_dict["failed_barge_ins"]:
                    barge_in_segments.append(f"     User: [{bi['user']['start']:.3f}s - {bi['user']['end']:.3f}s]")
                    barge_in_segments.append(f"     Agent: [{bi['agent']['start']:.3f}s - {bi['agent']['end']:.3f}s]")
                    barge_in_segments.append(f"     Stop duration: {bi['stop_duration_ms']:.3f} ms")

        segment_info = f"""
4. Speech segments (chronological order):
{segments_str}
5. Barge-in details:
{chr(10).join(barge_in_segments) if barge_in_segments else "   No barge-ins detected"}"""

    output = f"""
Evaluation metrics for conversation {metrics_dict["item_id"]}:
1. Turn-taking metrics:
   - Average latency: {metrics_dict["tt_latency"]:.3f} seconds
   - Precision: {metrics_dict["tt_precision"]:.3f}
   - Recall: {metrics_dict["tt_recall"]:.3f}
   - F1: {metrics_dict["tt_f1"]:.3f}
2. Barge-in statistics:
{chr(10).join(barge_in_stats)}
3. Backchanneling failures: {metrics_dict["bc_failure"]}{segment_info}
{"-" * 50}"""

    print(output)


def main(args):
    print(f"Loading results from: {args.results_dir}")
    # Load entries, preferring cached results unless force_recompute is set
    entries = load_results_jsonl(args.results_dir, prefer_cached=not args.force_recompute)
    audio_dir = get_audio_dir(args.results_dir)
    print(f"Loaded {len(entries)} entries")
    print(f"Audio directory: {audio_dir}")

    # Initialize models (may not be needed if all entries have cached data)
    vad_model, get_speech_timestamps = init_vad_model()

    asr_model = None
    if not args.disable_transcription:
        asr_model = init_asr_model(args.asr_model_name)

    # Metrics accumulators
    count = 0
    all_tt_latencies = []
    all_tt_precisions = []
    all_tt_recalls = []
    all_tt_f1s = []
    all_barge_in_success_rates = []
    all_barge_in_latencies = []
    all_bc_accuracies = []
    all_metrics_dicts = []

    # Speech quality metrics accumulators
    all_user_speech_wer = []
    all_agent_speech_wer = []
    all_agent_speech_cer = []
    all_hallucination_rates = []
    all_out_of_bounds_ratios = []

    # For per-sample results output
    per_sample_results = []

    for idx, entry in enumerate(tqdm(entries, desc="Processing entries")):
        item_id = get_item_id(entry)
        audio_path = get_audio_path(entry, audio_dir=audio_dir)
        generation_text = get_generation_text(entry)

        # Check for cached segmentation and transcription data (unless force_recompute)
        cached_segmentation, cached_transcription = get_cached_eval_data(entry)
        use_cached = not args.force_recompute and cached_segmentation is not None and cached_transcription is not None

        # Get audio filename for logging
        audio_filename = os.path.basename(audio_path) if audio_path else "unknown"

        if use_cached:
            print(f"\nProcessing: {item_id}")
            print(f"  Audio: {audio_filename}")
            print("  Using cached segmentation/transcription")
            # Extract segments from cached data
            user_segments = cached_segmentation.get("user_segments", [])
            agent_segments = cached_segmentation.get("agent_segments", [])
            use_vad_for_agent = cached_segmentation.get("used_audio_segmentation_for_agent", False)

            # Extract transcripts from cached data
            user_transcripts = parse_cached_transcripts(cached_transcription.get("user", {}))
            agent_transcripts = parse_cached_transcripts(cached_transcription.get("agent", {}))

            print(f"  Cached: {len(user_segments)} user segments, {len(agent_segments)} agent segments")

            # Get original entry if this is a wrapper format
            original_entry = entry.get("original_entry", entry)
            debug_info = None  # Will be retrieved later if needed
        else:
            if not audio_path or not os.path.exists(audio_path):
                print(f"Skipping {item_id}: audio file not found at {audio_path}")
                continue

            print(f"\nProcessing: {item_id}")
            print(f"  Audio: {audio_filename} ({audio_path})")

            # Load stereo audio: channel 0 = user, channel 1 = agent
            audio, audio_sr = torchaudio.load(audio_path)
            if audio.shape[0] < 2:
                print(f"  Warning: Expected stereo audio, got {audio.shape[0]} channel(s). Skipping.")
                continue

            user_audio = audio[0:1, :]
            agent_audio = audio[1:2, :]

            # Resample to 16kHz for VAD
            user_audio_16k = torchaudio.functional.resample(user_audio, audio_sr, 16000)
            agent_audio_16k = torchaudio.functional.resample(agent_audio, audio_sr, 16000)

            # Get estimated audio duration from debug_info for timestamp parsing
            debug_info = get_debug_info(entry)
            total_frames = debug_info.get("total_frames", 0) if debug_info else 0
            estimated_audio_duration = total_frames * FRAME_SIZE_SEC if total_frames > 0 else None

            # Get agent segments from timestamps or VAD fallback
            agent_segments = []
            use_vad_for_agent = args.use_audio_segmentation

            if not use_vad_for_agent and generation_text:
                agent_segments = parse_timestamped_text(generation_text, audio_duration=estimated_audio_duration)
                if agent_segments:
                    print(f"  Parsed {len(agent_segments)} agent segments from timestamps")

            if not agent_segments:
                if args.use_audio_segmentation:
                    print("  Using VAD for agent segmentation (--use_audio_segmentation)")
                else:
                    print("  No timestamps found, using VAD for agent segmentation")
                agent_vad_results = get_speech_timestamps(
                    agent_audio_16k.to("cuda"),
                    vad_model,
                    sampling_rate=16000,
                    min_silence_duration_ms=args.vad_min_silence_duration_ms,
                )
                agent_segments = [{"start": s["start"] / 16000, "end": s["end"] / 16000} for s in agent_vad_results]
                use_vad_for_agent = True
                print(f"  VAD detected {len(agent_segments)} agent segments")

            # Get user segments via VAD
            user_vad_results = get_speech_timestamps(
                user_audio_16k.to("cuda"),
                vad_model,
                sampling_rate=16000,
                min_silence_duration_ms=args.vad_min_silence_duration_ms,
            )
            user_segments = [{"start": s["start"] / 16000, "end": s["end"] / 16000} for s in user_vad_results]
            print(f"  VAD detected {len(user_segments)} user segments")

            # Initialize transcripts (will be filled below if not cached)
            user_transcripts = {}
            agent_transcripts = {}
            original_entry = entry

        # Get frame_alignment for speech quality metrics (debug_info already retrieved earlier)
        if not debug_info:
            debug_info = get_debug_info(entry)
        frame_alignment = debug_info.get("frame_alignment", {}) if debug_info else {}

        # Get audio duration - from audio file or estimate from segments
        audio_duration = 0.0
        if not use_cached and audio_path and os.path.exists(audio_path):
            audio_info = torchaudio.info(audio_path)
            audio_duration = audio_info.num_frames / audio_info.sample_rate
        else:
            # Estimate from segments
            all_ends = [s["end"] for s in user_segments + agent_segments]
            if all_ends:
                audio_duration = max(all_ends)

        # Compute metrics
        tt_metrics = compute_turn_taking_metrics(
            agent_segments,
            user_segments,
            args.tt_latency_threshold_sec,
            args.tt_precision_buffer_sec,
            args.tt_recall_buffer_sec,
        )
        tt_latency = tt_metrics["avg_latency"]

        success_barge_ins, failed_barge_ins = find_user_barge_ins(
            user_segments, agent_segments, args.barge_in_threshold_sec
        )
        barge_in_metrics = compute_barge_in_metrics(success_barge_ins, failed_barge_ins)

        end_time = args.end_time
        if end_time is not None:
            bc_failure = is_stopped_by_backchannel(agent_segments, end_time, args.barge_in_threshold_sec)
        else:
            bc_failure = []

        # Store metrics
        all_tt_latencies.append(tt_latency)
        all_tt_precisions.append(tt_metrics["precision"])
        all_tt_recalls.append(tt_metrics["recall"])
        all_tt_f1s.append(tt_metrics["f1"])

        if barge_in_metrics["has_barge_ins"]:
            all_barge_in_success_rates.append(barge_in_metrics["success_rate"])
            if "avg_latency_ms" in barge_in_metrics:
                all_barge_in_latencies.append(barge_in_metrics["avg_latency_ms"])

        bc_accuracy = sum(1 for x in bc_failure if not x) / len(bc_failure) if bc_failure else 0
        all_bc_accuracies.append(bc_accuracy)

        # Transcriptions - skip if using cached data
        if not use_cached and asr_model is not None:
            # Always transcribe agent segments with ASR to get actual TTS output
            # (The reference from frame_alignment is what model intended to say,
            #  the ASR transcription is what was actually spoken by TTS)
            print(f"  Transcribing {len(agent_segments)} agent segments with ASR...")
            for seg in agent_segments:
                transcript, _ = transcribe_segment(agent_audio, seg["start"], seg["end"], audio_sr, asr_model)
                agent_transcripts[(seg["start"], seg["end"])] = transcript

            # Transcribe user segments with ASR
            print(f"  Transcribing {len(user_segments)} user segments...")
            for seg in user_segments:
                transcript, _ = transcribe_segment(user_audio, seg["start"], seg["end"], audio_sr, asr_model)
                user_transcripts[(seg["start"], seg["end"])] = transcript

        # Compute speech quality metrics if frame_alignment is available
        user_speech_metrics = compute_user_speech_wer(
            user_segments, frame_alignment, user_transcripts, audio_duration, args.segment_buffer_sec
        )
        agent_speech_metrics = compute_agent_speech_quality(
            agent_segments, frame_alignment, agent_transcripts, args.segment_buffer_sec
        )
        hallucination_metrics = compute_tts_hallucinations(
            agent_segments, frame_alignment, agent_transcripts, args.segment_buffer_sec
        )
        token_balance_metrics = compute_token_balance(frame_alignment)

        # Accumulate speech quality metrics
        if user_speech_metrics.get("total_wer") is not None:
            all_user_speech_wer.append(user_speech_metrics["total_wer"])
        if user_speech_metrics.get("out_of_bounds_word_ratio") is not None:
            all_out_of_bounds_ratios.append(user_speech_metrics["out_of_bounds_word_ratio"])
        if agent_speech_metrics.get("total_wer") is not None:
            all_agent_speech_wer.append(agent_speech_metrics["total_wer"])
        if agent_speech_metrics.get("total_cer") is not None:
            all_agent_speech_cer.append(agent_speech_metrics["total_cer"])
        if hallucination_metrics.get("hallucination_rate") is not None:
            all_hallucination_rates.append(hallucination_metrics["hallucination_rate"])

        metrics_dict = {
            "item_id": item_id,
            "tt_latency": tt_latency,
            "tt_accuracy": tt_metrics["f1"],
            "tt_precision": tt_metrics["precision"],
            "tt_recall": tt_metrics["recall"],
            "tt_f1": tt_metrics["f1"],
            "barge_in_metrics": barge_in_metrics,
            "bc_failure": bc_failure,
            "user_segments": user_segments,
            "agent_segments": agent_segments,
            "success_barge_ins": success_barge_ins,
            "failed_barge_ins": failed_barge_ins,
            "user_transcripts": user_transcripts,
            "agent_transcripts": agent_transcripts,
            "user_speech_metrics": user_speech_metrics,
            "agent_speech_metrics": agent_speech_metrics,
            "hallucination_metrics": hallucination_metrics,
            "token_balance_metrics": token_balance_metrics,
        }

        all_metrics_dicts.append(metrics_dict)
        print_metrics(metrics_dict, verbose=args.verbose)
        count += 1

        # Build per-sample result for output
        if args.save_per_sample_results:
            # Convert tuple keys to string for JSON serialization
            user_transcripts_json = {f"{k[0]:.3f}-{k[1]:.3f}": v for k, v in user_transcripts.items()}
            agent_transcripts_json = {f"{k[0]:.3f}-{k[1]:.3f}": v for k, v in agent_transcripts.items()}

            sample_result = {
                "original_entry": original_entry,
                "eval_metrics": {
                    "turn_taking": {
                        "latency": tt_latency,
                        "precision": tt_metrics["precision"],
                        "recall": tt_metrics["recall"],
                        "f1": tt_metrics["f1"],
                    },
                    "barge_in": {
                        "has_barge_ins": barge_in_metrics["has_barge_ins"],
                        "success_rate": barge_in_metrics.get("success_rate", 0),
                        "success_count": barge_in_metrics.get("success_count", 0),
                        "total_count": barge_in_metrics.get("total_count", 0),
                        "avg_latency_ms": barge_in_metrics.get("avg_latency_ms", None),
                    },
                    "backchanneling": {"failures": bc_failure},
                    "user_speech": {
                        "total_wer": user_speech_metrics.get("total_wer"),
                        "total_ref_words": user_speech_metrics.get("total_ref_words", 0),
                        "total_substitutions": user_speech_metrics.get("total_substitutions", 0),
                        "total_insertions": user_speech_metrics.get("total_insertions", 0),
                        "total_deletions": user_speech_metrics.get("total_deletions", 0),
                        "per_segment": user_speech_metrics.get("per_segment", []),
                        "out_of_bounds_words": user_speech_metrics.get("out_of_bounds_words", []),
                        "out_of_bounds_word_count": user_speech_metrics.get("out_of_bounds_word_count", 0),
                        "out_of_bounds_word_ratio": user_speech_metrics.get("out_of_bounds_word_ratio"),
                    },
                    "agent_speech": {
                        "total_wer": agent_speech_metrics.get("total_wer"),
                        "total_cer": agent_speech_metrics.get("total_cer"),
                        "total_ref_words": agent_speech_metrics.get("total_ref_words", 0),
                        "total_ref_chars": agent_speech_metrics.get("total_ref_chars", 0),
                        "total_word_substitutions": agent_speech_metrics.get("total_word_substitutions", 0),
                        "total_word_insertions": agent_speech_metrics.get("total_word_insertions", 0),
                        "total_word_deletions": agent_speech_metrics.get("total_word_deletions", 0),
                        "total_char_substitutions": agent_speech_metrics.get("total_char_substitutions", 0),
                        "total_char_insertions": agent_speech_metrics.get("total_char_insertions", 0),
                        "total_char_deletions": agent_speech_metrics.get("total_char_deletions", 0),
                        "per_segment": agent_speech_metrics.get("per_segment", []),
                        "truncation_events": agent_speech_metrics.get("truncation_events", 0),
                        "truncated_words": agent_speech_metrics.get("truncated_words", []),
                    },
                    "tts_hallucinations": {
                        "hallucination_rate": hallucination_metrics.get("hallucination_rate"),
                        "total_hallucinated": hallucination_metrics.get("total_hallucinated", 0),
                        "total_agent_words": hallucination_metrics.get("total_agent_words", 0),
                        "hallucinations": hallucination_metrics.get("hallucinations", []),
                        "per_segment": hallucination_metrics.get("per_segment", []),
                    },
                    "token_balance": {
                        "agent_bos_count": token_balance_metrics.get("agent_bos_count", 0),
                        "agent_eos_count": token_balance_metrics.get("agent_eos_count", 0),
                        "agent_balance": token_balance_metrics.get("agent_balance", 0.0),
                        "user_bos_count": token_balance_metrics.get("user_bos_count", 0),
                        "user_eos_count": token_balance_metrics.get("user_eos_count", 0),
                        "user_balance": token_balance_metrics.get("user_balance", 0.0),
                    },
                },
                "segmentation": {
                    "user_segments": [{"start": s["start"], "end": s["end"]} for s in user_segments],
                    "agent_segments": [{"start": s["start"], "end": s["end"]} for s in agent_segments],
                    "used_audio_segmentation_for_agent": use_vad_for_agent,
                },
                "transcription": {"user": user_transcripts_json, "agent": agent_transcripts_json},
            }
            per_sample_results.append(sample_result)

    # Compute and print average metrics
    _valid_tt_latencies = [x for x in all_tt_latencies if x != INF_LATENCY]
    avg_metrics = {
        "avg_tt_latency": sum(_valid_tt_latencies) / len(_valid_tt_latencies) if _valid_tt_latencies else 0,
        "avg_tt_precision": sum(all_tt_precisions) / len(all_tt_precisions) * 100 if all_tt_precisions else 0,
        "avg_tt_recall": sum(all_tt_recalls) / len(all_tt_recalls) * 100 if all_tt_recalls else 0,
        "avg_tt_f1": sum(all_tt_f1s) / len(all_tt_f1s) * 100 if all_tt_f1s else 0,
        "avg_barge_in_success_rate": sum(all_barge_in_success_rates) / len(all_barge_in_success_rates)
        if all_barge_in_success_rates
        else 0,
        "avg_barge_in_latency": sum(all_barge_in_latencies) / len(all_barge_in_latencies)
        if all_barge_in_latencies
        else 0,
        "avg_bc_accuracy": sum(all_bc_accuracies) / len(all_bc_accuracies) * 100 if all_bc_accuracies else 0,
        "num_audios_evaluated": count,
        # Speech quality metrics
        "avg_user_speech_wer": sum(all_user_speech_wer) / len(all_user_speech_wer) * 100
        if all_user_speech_wer
        else None,
        "avg_out_of_bounds_ratio": sum(all_out_of_bounds_ratios) / len(all_out_of_bounds_ratios)
        if all_out_of_bounds_ratios
        else None,
        "avg_agent_speech_wer": sum(all_agent_speech_wer) / len(all_agent_speech_wer) * 100
        if all_agent_speech_wer
        else None,
        "avg_agent_speech_cer": sum(all_agent_speech_cer) / len(all_agent_speech_cer) * 100
        if all_agent_speech_cer
        else None,
        "avg_hallucination_rate": sum(all_hallucination_rates) / len(all_hallucination_rates) * 100
        if all_hallucination_rates
        else None,
    }

    # Format optional metrics for display
    user_wer_str = (
        f"{avg_metrics['avg_user_speech_wer']:.1f}%" if avg_metrics["avg_user_speech_wer"] is not None else "N/A"
    )
    out_bounds_str = (
        f"{avg_metrics['avg_out_of_bounds_ratio']:.3f}"
        if avg_metrics["avg_out_of_bounds_ratio"] is not None
        else "N/A"
    )
    agent_wer_str = (
        f"{avg_metrics['avg_agent_speech_wer']:.1f}%" if avg_metrics["avg_agent_speech_wer"] is not None else "N/A"
    )
    agent_cer_str = (
        f"{avg_metrics['avg_agent_speech_cer']:.1f}%" if avg_metrics["avg_agent_speech_cer"] is not None else "N/A"
    )
    halluc_str = (
        f"{avg_metrics['avg_hallucination_rate']:.1f}%" if avg_metrics["avg_hallucination_rate"] is not None else "N/A"
    )

    avg_metrics_str = f"""
{"=" * 50}
Average Metrics:
1. Turn-taking:
   - Average latency: {avg_metrics["avg_tt_latency"] * 1000:.1f} ms
   - Precision: {avg_metrics["avg_tt_precision"]:.1f}%
   - Recall: {avg_metrics["avg_tt_recall"]:.1f}%
   - F1: {avg_metrics["avg_tt_f1"]:.1f}%
2. User barge-in:
   - Average success rate: {avg_metrics["avg_barge_in_success_rate"]:.1f}%
   - Average latency: {avg_metrics["avg_barge_in_latency"]:.1f} ms
3. Back-channeling:
   - Average accuracy: {avg_metrics["avg_bc_accuracy"]:.1f}%
4. User speech quality:
   - Average WER: {user_wer_str}
   - Out-of-bounds word ratio: {out_bounds_str} words/sec
5. Agent speech quality:
   - Average WER: {agent_wer_str}
   - Average CER: {agent_cer_str}
   - TTS hallucination rate: {halluc_str}
6. Number of audios evaluated: {avg_metrics["num_audios_evaluated"]}
{"=" * 50}"""

    print(avg_metrics_str)

    if args.show_bottom_percentile:
        print_bottom_percentile_utterances(all_metrics_dicts, percentile=args.percentile_threshold)

    # Save dataset-level metrics to JSON file (in results_dir)
    if args.output_file:
        # If not an absolute path, save in results_dir
        if not os.path.isabs(args.output_file):
            output_file_path = os.path.join(args.results_dir, args.output_file)
        else:
            output_file_path = args.output_file

        dataset_metrics = {
            "dataset_metrics": {
                "turn_taking": {
                    "avg_latency_ms": avg_metrics["avg_tt_latency"] * 1000,
                    "avg_precision": avg_metrics["avg_tt_precision"],
                    "avg_recall": avg_metrics["avg_tt_recall"],
                    "avg_f1": avg_metrics["avg_tt_f1"],
                },
                "barge_in": {
                    "avg_success_rate": avg_metrics["avg_barge_in_success_rate"],
                    "avg_latency_ms": avg_metrics["avg_barge_in_latency"],
                },
                "backchanneling": {"avg_accuracy": avg_metrics["avg_bc_accuracy"]},
                "user_speech": {
                    "avg_wer": avg_metrics["avg_user_speech_wer"],
                    "out_of_bounds_word_ratio": avg_metrics["avg_out_of_bounds_ratio"],
                },
                "agent_speech": {
                    "avg_wer": avg_metrics["avg_agent_speech_wer"],
                    "avg_cer": avg_metrics["avg_agent_speech_cer"],
                    "hallucination_rate": avg_metrics["avg_hallucination_rate"],
                },
                "num_samples_evaluated": avg_metrics["num_audios_evaluated"],
            },
            "args": vars(args),
        }
        with open(output_file_path, "w") as f:
            json.dump(dataset_metrics, f, indent=2)
        print(f"\nDataset metrics saved to: {output_file_path}")

    # Save per-sample results to JSONL
    if args.save_per_sample_results and per_sample_results:
        output_jsonl_path = os.path.join(args.results_dir, "output_with_eval.jsonl")
        with open(output_jsonl_path, "w") as f:
            for result in per_sample_results:
                f.write(json.dumps(result) + "\n")
        print(f"Per-sample results saved to: {output_jsonl_path}")

    # Generate LLM judge input if requested
    if args.generate_llm_judge_input and per_sample_results:
        generate_llm_judge_input(per_sample_results, args.results_dir)


def format_conversation_for_llm_judge(result: dict, use_full_agent: bool) -> str:
    """Format conversation turns for LLM judge prompt."""
    transcription = result.get("transcription", {})
    user_trans = transcription.get("user", {})
    agent_trans = transcription.get("agent", {})

    agent_quality = result.get("eval_metrics", {}).get("agent_speech", {})
    per_segment = agent_quality.get("per_segment", [])

    # Build mapping of agent segments: start -> {full, sounded}
    agent_content = {}
    for seg in per_segment:
        start = seg.get("start", 0)
        agent_content[start] = {
            "full": seg.get("reference", ""),
            "sounded": seg.get("hypothesis", ""),
        }

    turns = []

    for time_range, text in user_trans.items():
        start = float(time_range.split("-")[0])
        turns.append({"start": start, "role": "User", "text": text})

    for time_range, text in agent_trans.items():
        start = float(time_range.split("-")[0])
        matched_text = None
        for seg_start, content in agent_content.items():
            if abs(seg_start - start) < 0.1:
                matched_text = content["full"] if use_full_agent else content["sounded"]
                break
        turns.append({"start": start, "role": "Agent", "text": matched_text or text})

    turns.sort(key=lambda x: x["start"])
    return "\n".join(f"[{t['role']}]: {t['text']}" for t in turns)


def generate_llm_judge_input(per_sample_results: list, results_dir: str):
    """Generate llm_judge_input.jsonl for nemo-skills generation."""
    output_entries = []

    for result in per_sample_results:
        original = result.get("original_entry", result)
        audio_path = original.get("audio_path", "")
        item_id = os.path.basename(audio_path) if audio_path else str(hash(json.dumps(original, sort_keys=True)))[:8]

        # Create entry for "full" evaluation (ignoring barge-ins)
        conv_full = format_conversation_for_llm_judge(result, use_full_agent=True)
        prompt_full = LLM_JUDGE_PROMPT_TEMPLATE.format(conversation=conv_full)
        output_entries.append(
            {
                "item_id": f"{item_id}_full",
                "category": "open",  # Required for AudioMetrics judge evaluation
                "subset_for_metrics": "full",  # Group metrics by eval type
                "messages": [{"role": "user", "content": prompt_full}],
            }
        )

        # Create entry for "sounded" evaluation (with barge-in effects)
        conv_sounded = format_conversation_for_llm_judge(result, use_full_agent=False)
        prompt_sounded = LLM_JUDGE_PROMPT_TEMPLATE.format(conversation=conv_sounded)
        output_entries.append(
            {
                "item_id": f"{item_id}_sounded",
                "category": "open",  # Required for AudioMetrics judge evaluation
                "subset_for_metrics": "sounded",  # Group metrics by eval type
                "messages": [{"role": "user", "content": prompt_sounded}],
            }
        )

    output_path = os.path.join(results_dir, "llm_judge_input.jsonl")
    with open(output_path, "w") as f:
        for entry in output_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"LLM judge input saved to: {output_path} ({len(output_entries)} prompts)")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate conversation behavior from output.jsonl format")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Directory containing output.jsonl and audio/ subdirectory"
    )
    parser.add_argument(
        "--barge_in_threshold_sec",
        type=float,
        default=1.5,
        help="Buffering time for the agent to stop after user barges in",
    )
    parser.add_argument(
        "--tt_latency_threshold_sec",
        type=float,
        default=1.5,
        help="Threshold in seconds for considering turn-taking accurate",
    )
    parser.add_argument(
        "--tt_precision_buffer_sec", type=float, default=0.5, help="Buffer time in seconds for precision calculation"
    )
    parser.add_argument(
        "--tt_recall_buffer_sec", type=float, default=0.5, help="Buffer time in seconds for recall calculation"
    )
    parser.add_argument(
        "--end_time",
        type=lambda x: None if x is None or x.lower() == "none" else parse_float_list(x),
        default=None,
        help="End time of backchanneling. Format: '[1,10,15,20]' or '1,10,15,20' or None",
    )
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed segment information")
    parser.add_argument(
        "--vad_min_silence_duration_ms",
        type=int,
        default=1500,
        help="Minimum silence duration in milliseconds for VAD",
    )
    parser.add_argument(
        "--segment_buffer_sec",
        type=float,
        default=0.5,
        help="Buffer in seconds to extend segment boundaries for WER calculation and out-of-bounds detection",
    )
    parser.add_argument(
        "--disable_transcription",
        action="store_true",
        default=False,
        help="Disable ASR transcription of user segments (enabled by default)",
    )
    parser.add_argument(
        "--asr_model_name",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="Name of the ASR model for user transcription",
    )
    parser.add_argument(
        "--show_bottom_percentile",
        action="store_true",
        default=True,
        help="Show analysis of utterances in the bottom percentile",
    )
    parser.add_argument(
        "--percentile_threshold",
        type=float,
        default=5.0,
        help="Percentile threshold for identifying low-quality utterances",
    )
    parser.add_argument(
        "--use_audio_segmentation",
        action="store_true",
        default=False,
        help="Use VAD+ASR for both channels instead of text timestamps. "
        "This ignores <|t|> markers and derives all timing from audio analysis.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics.json",
        help="Filename for dataset-level metrics JSON (saved in results_dir)",
    )
    parser.add_argument(
        "--save_per_sample_results",
        action="store_true",
        default=False,
        help="Save per-sample segmentation, ASR, and metrics to output_with_eval.jsonl",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        default=False,
        help="Force recompute segmentation and ASR even if cached results exist",
    )
    parser.add_argument(
        "--generate_llm_judge_input",
        action="store_true",
        default=False,
        help="Generate llm_judge_input.jsonl for LLM-as-judge evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
