import csv
import re
from collections import defaultdict

from example import candidate_windows, similarity

def tiou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

def load_segments(tid):
    path = f"transcript_dumps/conversation_{tid}.mp3.txt"
    segments = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(
                r"^([0-9.]+)-([0-9.]+)\s{2}(.*)$",
                line.rstrip()
            )

            if not m:
                continue

            segments.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "text": m.group(3),
                "words": [],
            })

    return segments

with open("data/question_train.csv", newline="", encoding="utf-8") as f:
    rows = [
        r for r in csv.DictReader(f)
        if r["question_type"] == "positive"
    ]

groups = defaultdict(list)

for r in rows:
    groups[r["transcript_id"]].append(r)

oracle = []
top1 = []
oracle_exact_segment = []
failures = []

for tid, group in groups.items():
    segments = load_segments(tid)
    windows = candidate_windows(segments)

    for r in group:
        q = r["question"]
        gs = float(r["evidence_start"])
        ge = float(r["evidence_end"])

        ranked = sorted(
            (
                (similarity(q, w["text"]), w)
                for w in windows
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        best_score, selected = ranked[0]

        selected_iou = tiou(
            selected["start"],
            selected["end"],
            gs,
            ge,
        )

        best_oracle = max(
            tiou(
                w["start"],
                w["end"],
                gs,
                ge,
            )
            for w in windows
        )

        best_segment = max(
            (
                tiou(
                    s["start"],
                    s["end"],
                    gs,
                    ge,
                )
                for s in segments
            ),
            default=0.0,
        )

        top1.append(selected_iou)
        oracle.append(best_oracle)
        oracle_exact_segment.append(best_segment)

        if selected_iou == 0 and best_oracle > 0:
            failures.append((
                tid,
                q,
                selected_iou,
                best_oracle,
                best_score,
            ))

print("POSITIVES:", len(rows))

print(
    "CURRENT TOP1 WINDOW:",
    round(sum(top1) / len(top1), 4),
    "nonzero=",
    sum(x > 0 for x in top1),
)

print(
    "ORACLE WINDOW:",
    round(sum(oracle) / len(oracle), 4),
    "nonzero=",
    sum(x > 0 for x in oracle),
)

print(
    "ORACLE SINGLE SEGMENT:",
    round(
        sum(oracle_exact_segment) /
        len(oracle_exact_segment),
        4,
    ),
    "nonzero=",
    sum(x > 0 for x in oracle_exact_segment),
)

print(
    "RANKING-RECOVERABLE MISSES:",
    len(failures),
)

print()
print("FIRST 20 RECOVERABLE MISSES")

for tid, q, current, best, score in failures[:20]:
    print(
        tid,
        "oracle=",
        round(best, 3),
        "score=",
        round(score, 3),
    )
    print("Q:", q)
    print()
