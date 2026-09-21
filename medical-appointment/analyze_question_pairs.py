import csv
from collections import defaultdict

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
)

with open("data/question_train.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)

for r in rows:
    groups[r["transcript_id"]].append(r)

records = []

for tid, group in groups.items():
    positives = [
        r for r in group
        if r["question_type"] == "positive"
    ]

    negatives = [
        r for r in group
        if r["question_type"] == "hard_negative"
    ]

    if not positives or not negatives:
        continue

    texts = (
        [r["question"] for r in negatives]
        + [r["question"] for r in positives]
    )

    emb = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    neg_emb = emb[:len(negatives)]
    pos_emb = emb[len(negatives):]

    sims = neg_emb @ pos_emb.T

    for i, neg in enumerate(negatives):
        order = sims[i].argsort()[::-1]

        best = positives[order[0]]
        score = float(sims[i][order[0]])

        records.append((
            score,
            tid,
            neg["question"],
            best["question"],
        ))

records.sort(reverse=True)

print("TOTAL HARD NEGATIVES:", len(records))

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    n = sum(score >= threshold for score, *_ in records)
    print(
        f">={threshold:.2f}:",
        n,
        f"({100*n/len(records):.1f}%)",
    )

print("\nTOP PAIRS")

for score, tid, neg, pos in records:
    print()
    print(tid, round(score, 4))
    print("NEG:", neg)
    print("POS:", pos)
