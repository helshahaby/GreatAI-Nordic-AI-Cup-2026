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
            m = re.match(
                r"^([0-9.]+)-([0-9.]+)\s{2}(.*)$",
                line.rstrip()
            )

            if not m:
                continue

            out.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "text": m.group(3),
                "words": [],
            })

    return out


def classify_best(question, windows):
    ranked = sorted(
        (
            (similarity(question, w["text"]), i, w)
            for i, w in enumerate(windows)
        ),
        key=lambda x: (-x[0], x[1]),
    )

    score, _, best = ranked[0]

    qnums = extract_numbers(question)
    pnums = extract_numbers(best["text"])

    if qnums and not qnums.issubset(pnums):
        return False, score, best, ranked

    if score < 0.30:
        return False, score, best, ranked

    if contradiction_veto(question, best["text"]):
        return False, score, best, ranked

    return True, score, best, ranked


def tie_evidence(top_score, ranked):
    tied = [
        w for score, _, w in ranked
        if abs(score - top_score) <= 1e-9
    ]

    return min(
        tied,
        key=lambda w: (
            w["end"] - w["start"],
            w["start"],
        ),
    )


with open("data/question_train.csv", newline="", encoding="utf-8") as f:
    rows = [
        r for r in csv.DictReader(f)
        if r["question_type"] == "positive"
    ]


groups = defaultdict(list)

for r in rows:
    groups[r["transcript_id"]].append(r)


current_scores = []
v33_scores = []

current_nonzero = 0
v33_nonzero = 0

changed = 0
better = 0
worse = 0
same = 0

examples = []


for tid, group in groups.items():
    segments = load_segments(tid)
    windows = candidate_windows(segments)

    for r in group:
        q = r["question"]
        gs = float(r["evidence_start"])
        ge = float(r["evidence_end"])

        answer, top_score, best, ranked = classify_best(q, windows)

        if not answer:
            current_scores.append(0.0)
            v33_scores.append(0.0)
            same += 1
            continue

        current = tiou(
            best["start"],
            best["end"],
            gs,
            ge,
        )

        evidence = tie_evidence(top_score, ranked)

        proposed = tiou(
            evidence["start"],
            evidence["end"],
            gs,
            ge,
        )

        current_scores.append(current)
        v33_scores.append(proposed)

        current_nonzero += current > 0
        v33_nonzero += proposed > 0

        if (
            evidence["start"] != best["start"]
            or evidence["end"] != best["end"]
        ):
            changed += 1

        if proposed > current:
            better += 1
        elif proposed < current:
            worse += 1
        else:
            same += 1

        if proposed != current and len(examples) < 20:
            examples.append(
                (
                    tid,
                    q,
                    current,
                    proposed,
                    best,
                    evidence,
                    top_score,
                )
            )


print("POSITIVES:", len(rows))
print(
    "CURRENT mean tIoU:",
    round(sum(current_scores) / len(rows), 4),
    "nonzero=",
    current_nonzero,
)
print(
    "V3.3 mean tIoU:",
    round(sum(v33_scores) / len(rows), 4),
    "nonzero=",
    v33_nonzero,
)
print("EVIDENCE WINDOWS CHANGED:", changed)
print("BETTER:", better)
print("WORSE:", worse)
print("SAME:", same)

print()
print("CHANGED EXAMPLES")

for tid, q, old, new, b, e, score in examples:
    print()
    print(
        tid,
        "score=",
        round(score, 3),
        "old=",
        round(old, 3),
        "new=",
        round(new, 3),
    )
    print("Q:", q)
    print(
        "OLD:",
        round(b["start"], 2),
        "-",
        round(b["end"], 2),
        b["text"],
    )
    print(
        "NEW:",
        round(e["start"], 2),
        "-",
        round(e["end"], 2),
        e["text"],
    )
