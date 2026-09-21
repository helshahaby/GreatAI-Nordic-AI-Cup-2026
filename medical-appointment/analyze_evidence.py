import csv
from collections import defaultdict

from example import (
    candidate_windows,
    similarity,
    extract_numbers,
    contradiction_veto,
)

with open("data/question_train.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if r["question_type"] == "positive"]

groups = defaultdict(list)
for r in rows:
    groups[r["transcript_id"]].append(r)

def tiou(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

total = 0
selected_overlap = 0
selected_zero = 0
vetoed = 0

records = []

for tid, group in groups.items():
    path = f"transcript_dumps/conversation_{tid}.mp3.txt"

    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        path = f"transcript_dumps/{tid}.txt"
        try:
            lines = open(path, encoding="utf-8").read().splitlines()
        except FileNotFoundError:
            print("MISSING", tid)
            continue

    segments = []

    for line in lines:
        if not line.strip():
            continue

        head, text = line.split("  ", 1)
        a, b = head.split("-")

        segments.append({
            "start": float(a),
            "end": float(b),
            "text": text.strip(),
            "words": [],
        })

    windows = candidate_windows(segments)

    for r in group:
        total += 1

        q = r["question"]
        gs = float(r["evidence_start"])
        ge = float(r["evidence_end"])

        ranked = sorted(
            ((similarity(q, w["text"]), w) for w in windows),
            key=lambda x: x[0],
            reverse=True,
        )

        score, best = ranked[0]

        qnums = extract_numbers(q)
        pnums = extract_numbers(best["text"])

        rejected = False

        if qnums and not qnums.issubset(pnums):
            rejected = True

        if score < 0.30:
            rejected = True

        if contradiction_veto(q, best["text"]):
            rejected = True

        overlap = tiou(
            best["start"],
            best["end"],
            gs,
            ge,
        )

        if rejected:
            vetoed += 1

        if overlap > 0:
            selected_overlap += 1
        else:
            selected_zero += 1

        records.append({
            "overlap": overlap,
            "rejected": rejected,
            "tid": tid,
            "score": score,
            "q": q,
            "gold": (gs, ge),
            "pred": (best["start"], best["end"]),
            "text": best["text"],
        })

print("TOTAL POSITIVES", total)
print("SELECTED WINDOW OVERLAPS GOLD", selected_overlap)
print("SELECTED WINDOW ZERO OVERLAP", selected_zero)
print("REJECTED POSITIVES", vetoed)

overlaps = [r["overlap"] for r in records]

if overlaps:
    print(
        "MEAN TIOU USING WHOLE SELECTED WINDOW",
        round(sum(overlaps) / len(overlaps), 4),
    )

print("\nZERO-OVERLAP POSITIVES")

for r in records:
    if r["overlap"] != 0:
        continue

    print()
    print(
        r["tid"],
        "score=",
        round(r["score"], 3),
        "rejected=",
        r["rejected"],
    )
    print(
        "GOLD",
        f"{r['gold'][0]:.2f}-{r['gold'][1]:.2f}",
        "SELECTED",
        f"{r['pred'][0]:.2f}-{r['pred'][1]:.2f}",
    )
    print("Q:", r["q"])
    print("P:", r["text"])
