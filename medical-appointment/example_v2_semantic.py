import logging
import os
import re
import tempfile

import numpy as np
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import decode_audio


logger = logging.getLogger(__name__)

MODEL_SIZE = "medium.en"

logger.info("Loading Whisper model %s", MODEL_SIZE)

_model = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    compute_type="float16",
)

logger.info("Loading semantic model")

_semantic_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)


def normalize(text: str) -> str:
    text = text.lower()

    replacements = {
        "milligrams": "mg",
        "milligram": "mg",
        "micrograms": "mcg",
        "microgram": "mcg",
        "grams": "g",
        "gram": "g",
        "milliliters": "ml",
        "milliliter": "ml",
        "weeks": "week",
        "days": "day",
        "months": "month",
        "hours": "hour",
        "twice daily": "twice a day",
        "once daily": "once a day",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(r"[^a-z0-9.%/]+", " ", text)

    return " ".join(text.split())


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "to", "of", "for", "and", "or",
    "in", "on", "at", "it", "this", "that", "there",
    "any", "do", "does", "did", "will", "would",
    "should", "could", "can", "have", "has", "had",
    "with", "from", "as", "about", "right", "yes",
    "no", "patient", "doctor",
}


def tokens(text):
    return [
        x for x in normalize(text).split()
        if len(x) > 1 and x not in STOPWORDS
    ]


def extract_numbers(text):
    return set(
        re.findall(
            r"\b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\b",
            normalize(text),
        )
    )


def transcribe(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(audio_bytes)
        f.flush()

        segments, _ = _model.transcribe(
            f.name,
            language="en",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=True,
        )

        result = []

        for seg in segments:
            words = []

            if seg.words:
                for word in seg.words:
                    if word.start is None or word.end is None:
                        continue

                    words.append({
                        "text": word.word.strip(),
                        "start": float(word.start),
                        "end": float(word.end),
                    })

            result.append({
                "text": seg.text.strip(),
                "start": float(seg.start),
                "end": float(seg.end),
                "words": words,
            })

        return result


def candidate_windows(segments):
    windows = []
    n = len(segments)

    for size in (1, 2):
        for i in range(n):
            j = i + size

            if j > n:
                continue

            group = segments[i:j]

            windows.append({
                "text": " ".join(x["text"] for x in group),
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "segments": group,
            })

    return windows


def lexical_score(question, passage):
    q = set(tokens(question))
    p = set(tokens(passage))

    if not q:
        return 0.0

    return len(q & p) / len(q)


def semantic_scores(question, windows):
    texts = [question] + [w["text"] for w in windows]

    emb = _semantic_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    q = emb[0]
    passages = emb[1:]

    return passages @ q


def number_conflict(question, passage):
    qnums = extract_numbers(question)

    if not qnums:
        return False

    pnums = extract_numbers(passage)

    return not qnums.issubset(pnums)


def choose_span(window, question):
    qtokens = set(tokens(question))

    matching = []

    for seg in window["segments"]:
        for word in seg["words"]:
            nw = normalize(word["text"])

            if nw in qtokens:
                matching.append(word)

    duration = window["end"] - window["start"]

    if not matching:
        if duration <= 7.0:
            return window["start"], window["end"]

        midpoint = (window["start"] + window["end"]) / 2.0

        return (
            max(window["start"], midpoint - 2.5),
            min(window["end"], midpoint + 2.5),
        )

    first = matching[0]
    last = matching[-1]

    start = max(
        window["start"],
        first["start"] - 0.8,
    )

    end = min(
        window["end"],
        last["end"] + 1.2,
    )

    if end - start < 1.2:
        centre = (start + end) / 2.0
        start = max(window["start"], centre - 0.8)
        end = min(window["end"], centre + 0.8)

    return float(start), float(end)


def answer_from_transcript(segments, question):
    windows = candidate_windows(segments)

    if not windows:
        return False, None

    semantic = semantic_scores(question, windows)

    ranked = []

    for sem, window in zip(semantic, windows):
        lex = lexical_score(question, window["text"])

        score = (
            0.78 * float(sem)
            + 0.22 * lex
        )

        if number_conflict(question, window["text"]):
            score -= 0.35

        ranked.append((score, float(sem), lex, window))

    ranked.sort(key=lambda x: x[0], reverse=True)

    best_score, best_sem, best_lex, best = ranked[0]

    if number_conflict(question, best["text"]):
        return False, None

    # Conservative threshold because hard negatives are the largest
    # classification weakness in V1.
    if best_score < 0.43:
        return False, None

    # A purely semantic but weakly related match is risky.
    if best_sem < 0.34 and best_lex < 0.20:
        return False, None

    start, end = choose_span(best, question)

    return True, (start, end)


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    audio_bytes = decode_audio(request.audio_base64)

    try:
        segments = transcribe(audio_bytes)

        if os.getenv("DUMP_TRANSCRIPTS") == "1":
            os.makedirs("transcript_dumps", exist_ok=True)

            out = os.path.join(
                "transcript_dumps",
                request.audio_filename + ".txt",
            )

            with open(out, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(
                        f"{seg['start']:.2f}-{seg['end']:.2f}  "
                        f"{seg['text']}\n"
                    )

        logger.info(
            "%s: %d segments, %d questions",
            request.audio_filename,
            len(segments),
            len(request.questions),
        )

        answers = []
        starts = []
        ends = []

        for question in request.questions:
            try:
                answer, span = answer_from_transcript(
                    segments,
                    question,
                )
            except Exception:
                logger.exception(
                    "Question failure: %s",
                    question,
                )

                answer = False
                span = None

            answers.append(bool(answer))

            if answer and span is not None:
                starts.append(float(span[0]))
                ends.append(float(span[1]))
            else:
                starts.append(None)
                ends.append(None)

        return ASRQuestionResponseDto(
            answers=answers,
            evidence_start=starts,
            evidence_end=ends,
        )

    except Exception:
        logger.exception(
            "Conversation failure: %s",
            request.audio_filename,
        )

        n = len(request.questions)

        return ASRQuestionResponseDto(
            answers=[False] * n,
            evidence_start=[None] * n,
            evidence_end=[None] * n,
        )
