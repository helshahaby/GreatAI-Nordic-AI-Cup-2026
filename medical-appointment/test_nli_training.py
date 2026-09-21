import csv
import os
from collections import defaultdict

import numpy as np
from transformers import pipeline

from example import (
    _semantic_model,
    candidate_windows,
    transcribe,
)
from utils import load_sample_audio


nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1,
)

with open("data/question_train.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)

for r in rows:
    groups[r["transcript_id"]].append(r)


def nli_result(premise, hypothesis):
    r = nli({
        "text": premise,
        "text_pair": hypothesis,
    })

    return r["label"], float(r["score"])


def semantic_top(question, windows, k=5):
    texts = [question] + [w["text"] for w in windows]

    emb = _semantic_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    scores = emb[1:] @ emb[0]

    order = np.argsort(scores)[::-1][:k]

    return [
        (float(scores[i]), windows[i])
        for i in order
    ]


positive_counts = defaultdict(int)
negative_counts = defaultdict(int)

positive_examples = []
negative_examples = []

for gi, (tid, group) in enumerate(groups.items(), 1):
    filename = f"conversation_{tid}.mp3"

    segments = transcribe(load_sample_audio(filename))
    windows = candidate_windows(segments)

    print(f"{gi:02d}/{len(groups)} {tid}")

    for row in group:
        question = row["question"]
        qtype = row["question_type"]

        if qtype == "positive":
            gs = float(row["evidence_start"])
            ge = float(row["evidence_end"])

            overlapping = [
                s for s in segments
                if s["end"] >= gs and s["start"] <= ge
            ]

            if not overlapping:
                continue

            premise = " ".join(
                s["text"] for s in overlapping
            )

            label, score = nli_result(
                premise,
                question,
            )

            positive_counts[label] += 1

            if label != "entailment":
                positive_examples.append(
                    (
                        label,
                        score,
                        question,
                        premise,
                    )
                )

        elif qtype == "hard_negative":
            candidates = semantic_top(
                question,
                windows,
                k=5,
            )

            best = None

            for sem, window in candidates:
                label, score = nli_result(
                    window["text"],
                    question,
                )

                item = (
                    label,
                    score,
                    sem,
                    window["text"],
                )

                if best is None:
                    best = item

                if label == "entailment":
                    if (
                        best[0] != "entailment"
                        or score > best[1]
                    ):
                        best = item

                elif (
                    best[0] != "entailment"
                    and score > best[1]
                ):
                    best = item

            label, score, sem, premise = best

            negative_counts[label] += 1

            if label == "entailment":
                negative_examples.append(
                    (
                        label,
                        score,
                        sem,
                        question,
                        premise,
                    )
                )


print("\nPOSITIVE GOLD-SPAN NLI")
for k in sorted(positive_counts):
    print(k, positive_counts[k])

print("\nHARD-NEGATIVE TOP5 NLI")
for k in sorted(negative_counts):
    print(k, negative_counts[k])

print("\nPOSITIVE NOT ENTAILED")
for label, score, q, p in positive_examples:
    print()
    print(label, round(score, 4))
    print("Q:", q)
    print("P:", p)

print("\nHARD NEGATIVE FALSE ENTAILMENTS")
for label, score, sem, q, p in negative_examples:
    print()
    print(
        label,
        "nli=",
        round(score, 4),
        "semantic=",
        round(sem, 4),
    )
    print("Q:", q)
    print("P:", p)
