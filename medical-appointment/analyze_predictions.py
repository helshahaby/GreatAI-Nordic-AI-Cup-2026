import csv
import requests
from collections import Counter, defaultdict

from utils import encode_audio, load_sample_audio

with open("data/question_train.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)

for row in rows:
    groups[row["transcript_id"]].append(row)

errors = []
stats = Counter()

for idx, (tid, group) in enumerate(groups.items(), 1):
    filename = f"conversation_{tid}.mp3"

    r = requests.post(
        "http://127.0.0.1:9054/predict",
        json={
            "audio_base64": encode_audio(load_sample_audio(filename)),
            "audio_filename": filename,
            "questions": [x["question"] for x in group],
        },
        timeout=60,
    )

    r.raise_for_status()
    result = r.json()

    for row, pred, start, end in zip(
        group,
        result["answers"],
        result["evidence_start"],
        result["evidence_end"],
    ):
        gold = str(row["label"]).strip().lower() in {
            "1", "true", "yes"
        }

        qtype = row["question_type"]

        if pred == gold:
            stats[f"{qtype}_correct"] += 1
        else:
            stats[f"{qtype}_wrong"] += 1

            errors.append({
                "transcript": tid,
                "id": row["question_id"],
                "type": qtype,
                "gold": gold,
                "pred": pred,
                "question": row["question"],
                "pred_start": start,
                "pred_end": end,
                "gold_start": row["evidence_start"],
                "gold_end": row["evidence_end"],
            })

    print(f"{idx:02d}/{len(groups)} {tid}")

print("\nSUMMARY")
for k in sorted(stats):
    print(k, stats[k])

print("\nFALSE POSITIVES")
for e in errors:
    if not e["gold"] and e["pred"]:
        print(
            f'{e["transcript"]}\t'
            f'{e["type"]}\t'
            f'{e["pred_start"]}-{e["pred_end"]}\t'
            f'{e["question"]}'
        )

print("\nFALSE NEGATIVES")
for e in errors:
    if e["gold"] and not e["pred"]:
        print(
            f'{e["transcript"]}\t'
            f'{e["type"]}\t'
            f'GOLD={e["gold_start"]}-{e["gold_end"]}\t'
            f'{e["question"]}'
        )
