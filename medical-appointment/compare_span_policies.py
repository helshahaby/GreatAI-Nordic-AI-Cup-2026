import csv
import re
from collections import defaultdict

from example import (
    candidate_windows,
    similarity,
    extract_numbers,
    contradiction_veto,
)

def tiou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

def load_segments(tid):
    path = f"transcript_dumps/conversation_{tid}.mp3.txt"
    out = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = re.match(r"^([0-9.]+)-([0-9.]+)\s{2}(.*)$", line)

            if not m:
                continue

            out.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "text": m.group(3),
                "words": [],
            })

    return out

def retrieve(segments, question):
    windows = candidate_windows(segments)

    ranked = sorted(
        ((similarity(question, w["text"]), w) for w in windows),
        key=lambda x: x[0],
        reverse=True,
    )

    score, best = ranked[0]

    qnums = extract_numbers(question)
    pnums = extract_numbers(best["text"])

    if qnums and not qnums.issubset(pnums):
        return None

    if score < 0.30:
        return None

    if contradiction_veto(question, best["text"]):
        return None

    return best

def segment_similarity(question, seg):
    return similarity(question, seg["text"])

with open("data/question_train.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if r["question_type"] == "positive"]

groups = defaultdict(list)

for r in rows:
    groups[r["transcript_id"]].append(r)

scores = {
    "whole_window": [],
    "first_segment": [],
    "best_segment": [],
}

answered = 0

for tid, group in groups.items():
    segments = load_segments(tid)

    for r in group:
        gs = float(r["evidence_start"])
        ge = float(r["evidence_end"])

        best = retrieve(segments, r["question"])

        if best is None:
            for key in scores:
                scores[key].append(0.0)
            continue

        answered += 1

        scores["whole_window"].append(
            tiou(best["start"], best["end"], gs, ge)
        )

        first = best["segments"][0]

        scores["first_segment"].append(
            tiou(first["start"], first["end"], gs, ge)
        )

        best_seg = max(
            best["segments"],
            key=lambda s: segment_similarity(r["question"], s),
        )

        scores["best_segment"].append(
            tiou(best_seg["start"], best_seg["end"], gs, ge)
        )

print("POSITIVES:", len(rows))
print("ANSWERED YES:", answered)

for name, vals in scores.items():
    print(
        name,
        "mean_tIoU=",
        round(sum(vals) / len(vals), 4),
        "nonzero=",
        sum(v > 0 for v in vals),
    )
