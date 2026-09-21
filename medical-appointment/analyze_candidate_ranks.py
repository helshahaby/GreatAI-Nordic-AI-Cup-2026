import csv
import re
from collections import Counter, defaultdict

from example import candidate_windows, similarity


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


with open("data/question_train.csv", newline="", encoding="utf-8") as f:
    rows = [
        r for r in csv.DictReader(f)
        if r["question_type"] == "positive"
    ]


groups = defaultdict(list)

for r in rows:
    groups[r["transcript_id"]].append(r)


best_ranks = []
score_gaps = []
details = []


for tid, group in groups.items():
    segments = load_segments(tid)
    windows = candidate_windows(segments)

    for r in group:
        q = r["question"]
        gs = float(r["evidence_start"])
        ge = float(r["evidence_end"])

        ranked = sorted(
            [
                (similarity(q, w["text"]), w)
                for w in windows
            ],
            key=lambda x: x[0],
            reverse=True,
        )

        oracle_idx = max(
            range(len(ranked)),
            key=lambda i: tiou(
                ranked[i][1]["start"],
                ranked[i][1]["end"],
                gs,
                ge,
            ),
        )

        oracle_iou = tiou(
            ranked[oracle_idx][1]["start"],
            ranked[oracle_idx][1]["end"],
            gs,
            ge,
        )

        rank = oracle_idx + 1
        best_ranks.append(rank)

        top_score = ranked[0][0]
        oracle_score = ranked[oracle_idx][0]
        gap = top_score - oracle_score
        score_gaps.append(gap)

        details.append(
            (
                rank,
                gap,
                oracle_iou,
                tid,
                q,
                ranked[0][1]["start"],
                ranked[oracle_idx][1]["start"],
            )
        )


print("POSITIVES:", len(best_ranks))

for k in [1, 2, 3, 5, 10, 20]:
    n = sum(r <= k for r in best_ranks)
    print(
        f"ORACLE IN TOP {k}:",
        n,
        f"({n / len(best_ranks):.1%})",
    )


print()
print("ORACLE RANK DISTRIBUTION")
counts = Counter(best_ranks)

for r in sorted(counts)[:20]:
    print(r, counts[r])


sorted_gaps = sorted(score_gaps)

print()
print("TOP-vs-ORACLE SCORE GAP")

for pct in [25, 50, 75, 90, 95]:
    i = round((pct / 100) * (len(sorted_gaps) - 1))
    print(
        f"P{pct}:",
        round(sorted_gaps[i], 4),
    )


print()
print("FIRST 25 CASES WHERE ORACLE IS TOP-5 BUT NOT TOP-1")

shown = 0

for rank, gap, oiou, tid, q, top_start, oracle_start in details:
    if 1 < rank <= 5:
        print(
            tid,
            "rank=",
            rank,
            "gap=",
            round(gap, 3),
            "oracle_tIoU=",
            round(oiou, 3),
            "top_start=",
            round(top_start, 2),
            "oracle_start=",
            round(oracle_start, 2),
        )
        print("Q:", q)
        print()

        shown += 1

        if shown >= 25:
            break
