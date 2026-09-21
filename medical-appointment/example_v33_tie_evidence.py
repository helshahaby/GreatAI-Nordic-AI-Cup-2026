import logging
import os
import re
import tempfile
from typing import Optional, Tuple

from faster_whisper import WhisperModel

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from utils import Span, decode_audio

logger = logging.getLogger(__name__)

MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium.en")

logger.info("Loading Whisper model %s", MODEL_SIZE)

_model = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    compute_type="float16",
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

    text = re.sub(r"[^a-z0-9.%]+", " ", text)
    return " ".join(text.split())


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "to", "of", "for", "and", "or", "in", "on", "at", "it",
    "this", "that", "there", "any", "do", "does", "did", "will",
    "would", "should", "could", "can", "have", "has", "had",
    "with", "from", "as", "about", "right", "yes", "no",
    "patient", "doctor",
}


def tokens(text: str):
    return [
        x for x in normalize(text).split()
        if len(x) > 1 and x not in STOPWORDS
    ]


def transcribe(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(audio_bytes)
        f.flush()

        segments, info = _model.transcribe(
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


def extract_numbers(text: str):
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", normalize(text)))


def candidate_windows(segments):
    windows = []

    n = len(segments)

    for size in (1, 2, 3):
        for i in range(n):
            j = min(n, i + size)

            if j - i != size:
                continue

            group = segments[i:j]

            text = " ".join(x["text"] for x in group)

            windows.append({
                "text": text,
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "segments": group,
            })

    return windows


def similarity(question: str, passage: str) -> float:
    q = set(tokens(question))
    p = set(tokens(passage))

    if not q:
        return 0.0

    overlap = len(q & p) / len(q)

    qnums = extract_numbers(question)
    pnums = extract_numbers(passage)

    if qnums:
        if qnums <= pnums:
            overlap += 0.35
        elif qnums & pnums:
            overlap += 0.10
        else:
            overlap -= 0.40

    return overlap


def tight_span(window, question):
    qtokens = set(tokens(question))

    selected = []

    for seg in window["segments"]:
        for word in seg["words"]:
            nw = normalize(word["text"])

            if nw and nw in qtokens:
                selected.append(word)

    if not selected:
        return window["start"], window["end"]

    start = max(window["start"], selected[0]["start"] - 1.0)
    end = min(window["end"], selected[-1]["end"] + 1.5)

    if end <= start:
        return window["start"], window["end"]

    return start, end



OPPOSITE_GROUPS = [
    ({"normal"}, {"abnormal"}),
    ({"stable"}, {"unstable"}),
    ({"harmless", "benign"}, {"suspicious"}),
    ({"positive"}, {"negative"}),
    ({"raised", "elevated", "high"}, {"normal", "low"}),
    ({"increase", "increased", "higher"}, {"decrease", "decreased", "lower"}),
    ({"continue", "continued", "unchanged"}, {"stop", "stopped", "discontinue", "discontinued"}),
    ({"renew", "renewed"}, {"stop", "stopped", "discontinue", "discontinued"}),
    ({"start", "started"}, {"stop", "stopped", "discontinue", "discontinued"}),
    ({"present", "found"}, {"absent"}),
    ({"with"}, {"without"}),
]


NEGATION_WORDS = {
    "no", "not", "none", "nothing", "without",
    "deny", "denies", "denied", "free",
}


def normalized_token_set(text: str):
    return set(normalize(text).split())


def number_conflict(question: str, passage: str) -> bool:
    qnums = extract_numbers(question)

    if not qnums:
        return False

    pnums = extract_numbers(passage)

    return bool(pnums) and not qnums.issubset(pnums)


def explicit_opposite_conflict(question: str, passage: str) -> bool:
    q = normalized_token_set(question)
    p = normalized_token_set(passage)

    for left, right in OPPOSITE_GROUPS:
        if q & left and p & right:
            return True

        if q & right and p & left:
            return True

    return False


def negation_conflict(question: str, passage: str) -> bool:
    q = normalized_token_set(question)
    p = normalized_token_set(passage)

    qneg = bool(q & NEGATION_WORDS)
    pneg = bool(p & NEGATION_WORDS)

    # Only use negation as a veto when question and passage also
    # share meaningful content. This avoids treating unrelated
    # negations as contradictions.
    shared = set(tokens(question)) & set(tokens(passage))

    if len(shared) < 2:
        return False

    return qneg != pneg


def contradiction_veto(question: str, passage: str) -> bool:
    if number_conflict(question, passage):
        return True

    if explicit_opposite_conflict(question, passage):
        return True

    return False


def answer_from_transcript(segments, question):
    windows = candidate_windows(segments)

    if not windows:
        return False, None

    ranked = sorted(
        ((similarity(question, w["text"]), w) for w in windows),
        key=lambda x: x[0],
        reverse=True,
    )

    best_score, best = ranked[0]

    qnums = extract_numbers(question)
    pnums = extract_numbers(best["text"])

    # A number mentioned in the question but absent from the strongest
    # supporting passage is a strong hard-negative signal.
    if qnums and not qnums.issubset(pnums):
        return False, None

    # Very weak topical match -> off-topic.
    if best_score < 0.30:
        return False, None

    # V3 precision layer:
    # V1 found a plausible passage, but reject it when that passage
    # explicitly conflicts with the proposition in the question.
    if contradiction_veto(question, best["text"]):
        return False, None

    start, end = tight_span(best, question)

    return True, (float(start), float(end))


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    audio_bytes = decode_audio(request.audio_base64)

    try:
        segments = transcribe(audio_bytes)

        if os.getenv("DUMP_TRANSCRIPTS") == "1":
            os.makedirs("transcript_dumps", exist_ok=True)
            out = os.path.join(
                "transcript_dumps",
                request.audio_filename + ".txt"
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
                answer, span = answer_from_transcript(segments, question)
            except Exception:
                logger.exception("Question failure: %s", question)
                answer, span = False, None

            answers.append(bool(answer))

            if answer and span is not None:
                starts.append(span[0])
                ends.append(span[1])
            else:
                starts.append(None)
                ends.append(None)

        return ASRQuestionResponseDto(
            answers=answers,
            evidence_start=starts,
            evidence_end=ends,
        )

    except Exception:
        logger.exception("Conversation failure: %s", request.audio_filename)

        n = len(request.questions)

        return ASRQuestionResponseDto(
            answers=[False] * n,
            evidence_start=[None] * n,
            evidence_end=[None] * n,
        )
