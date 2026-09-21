import json
import shutil
from pathlib import Path

ROOT = Path("src/helsinki")
OUT = Path("datasets/nordic")

CLASSES = [
    "hangar",
    "helicopter",
    "jet_plane",
    "large_launcher",
    "large_tower",
    "medium_launcher",
    "medium_plane",
    "mine_roller",
    "small_launcher",
    "small_plane",
    "small_tower",
    "ta-ta",
    "tank",
    "condor",
    "jammer",
    "spacecraft",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

W = 3840
H = 2160

# Temporal split instead of random adjacent-frame leakage
TRAIN_FRAMES = set(range(0, 20))
VAL_FRAMES = set(range(20, 25))

for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

for ann_path in sorted((ROOT / "annotations").glob("frame_*.json")):

    data = json.loads(ann_path.read_text())
    frame = int(data["frame"])

    if frame in TRAIN_FRAMES:
        split = "train"
    elif frame in VAL_FRAMES:
        split = "val"
    else:
        continue

    stem = ann_path.stem
    src_image = ROOT / "images" / f"{stem}.png"

    shutil.copy2(
        src_image,
        OUT / "images" / split / src_image.name
    )

    lines = []

    for a in data["annotations"]:

        cls = a["object_id"]

        if cls not in CLASS_TO_ID:
            print("WARNING unknown class:", cls)
            continue

        x1, y1, x2, y2 = a["bbox"]

        # Clip to image
        x1 = max(0, min(W, x1))
        y1 = max(0, min(H, y1))
        x2 = max(0, min(W, x2))
        y2 = max(0, min(H, y2))

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            continue

        xc = (x1 + x2) / 2 / W
        yc = (y1 + y2) / 2 / H
        nw = bw / W
        nh = bh / H

        lines.append(
            f"{CLASS_TO_ID[cls]} "
            f"{xc:.8f} {yc:.8f} "
            f"{nw:.8f} {nh:.8f}"
        )

    label_path = OUT / "labels" / split / f"{stem}.txt"
    label_path.write_text("\n".join(lines))

print("Dataset created:", OUT.resolve())
print("Classes:", len(CLASSES))
