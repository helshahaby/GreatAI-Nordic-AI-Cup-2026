import json
import random
from pathlib import Path

import cv2

random.seed(42)

ROOT = Path("src/helsinki")
OUT = Path("datasets/nordic_camera")

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

SOURCE_W = 3840
SOURCE_H = 2160

# What the challenge model actually receives.
OUTPUT_W = 960
OUTPUT_H = 540

# Source-pixel dimensions represented by each camera level.
LEVELS = {
    0: (3840, 2160),
    1: (1920, 1080),
    2: (960, 540),
}

# Keep the temporal split.
TRAIN_FRAMES = set(range(0, 20))
VAL_FRAMES = set(range(20, 25))


def intersection(box, crop):
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop

    ix1 = max(x1, cx1)
    iy1 = max(y1, cy1)
    ix2 = min(x2, cx2)
    iy2 = min(y2, cy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    return ix1, iy1, ix2, iy2


def visibility(box, clipped):
    x1, y1, x2, y2 = box
    ix1, iy1, ix2, iy2 = clipped

    original_area = max(1, (x2 - x1) * (y2 - y1))
    visible_area = (ix2 - ix1) * (iy2 - iy1)

    return visible_area / original_area


def make_crop(center_x, center_y, crop_w, crop_h):
    x1 = int(round(center_x - crop_w / 2))
    y1 = int(round(center_y - crop_h / 2))

    x1 = max(0, min(SOURCE_W - crop_w, x1))
    y1 = max(0, min(SOURCE_H - crop_h, y1))

    return x1, y1, x1 + crop_w, y1 + crop_h


def save_sample(image, annotations, crop, split, name):
    cx1, cy1, cx2, cy2 = crop

    cropped = image[cy1:cy2, cx1:cx2]

    if cropped.size == 0:
        return 0

    resized = cv2.resize(
        cropped,
        (OUTPUT_W, OUTPUT_H),
        interpolation=cv2.INTER_AREA
    )

    labels = []

    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    for a in annotations:
        cls = a["object_id"]

        if cls not in CLASS_TO_ID:
            continue

        box = a["bbox"]
        clipped = intersection(box, crop)

        if clipped is None:
            continue

        # Avoid training on tiny fragments at crop borders.
        if visibility(box, clipped) < 0.50:
            continue

        x1, y1, x2, y2 = clipped

        # Coordinates relative to source crop.
        x1 -= cx1
        x2 -= cx1
        y1 -= cy1
        y2 -= cy1

        # Scale into 960x540 transmitted image.
        sx = OUTPUT_W / crop_w
        sy = OUTPUT_H / crop_h

        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        bw = x2 - x1
        bh = y2 - y1

        if bw < 1 or bh < 1:
            continue

        xc = ((x1 + x2) / 2) / OUTPUT_W
        yc = ((y1 + y2) / 2) / OUTPUT_H
        nw = bw / OUTPUT_W
        nh = bh / OUTPUT_H

        labels.append(
            f"{CLASS_TO_ID[cls]} "
            f"{xc:.8f} {yc:.8f} "
            f"{nw:.8f} {nh:.8f}"
        )

    # Keep positive samples for this first model.
    if not labels:
        return 0

    image_path = OUT / "images" / split / f"{name}.jpg"
    label_path = OUT / "labels" / split / f"{name}.txt"

    cv2.imwrite(
        str(image_path),
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    label_path.write_text("\n".join(labels))

    return len(labels)


for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

sample_count = {"train": 0, "val": 0}
label_count = {"train": 0, "val": 0}

for ann_path in sorted((ROOT / "annotations").glob("frame_*.json")):

    data = json.loads(ann_path.read_text())
    frame = int(data["frame"])

    if frame in TRAIN_FRAMES:
        split = "train"
    elif frame in VAL_FRAMES:
        split = "val"
    else:
        continue

    image_path = ROOT / "images" / f"{ann_path.stem}.png"
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load:", image_path)
        continue

    annotations = data["annotations"]

    # -----------------------------------------------------
    # L0: whole scene -> 960x540
    # -----------------------------------------------------

    crop = (0, 0, SOURCE_W, SOURCE_H)

    n = save_sample(
        image,
        annotations,
        crop,
        split,
        f"{ann_path.stem}_L0"
    )

    if n:
        sample_count[split] += 1
        label_count[split] += n

    # -----------------------------------------------------
    # L1/L2: target-centered crops with jitter.
    # -----------------------------------------------------

    for object_index, a in enumerate(annotations):

        x1, y1, x2, y2 = a["bbox"]

        object_cx = (x1 + x2) / 2
        object_cy = (y1 + y2) / 2

        for level in [1, 2]:

            crop_w, crop_h = LEVELS[level]

            # More variants during training, deterministic single
            # target-centered observation for validation.
            variants = 3 if split == "train" else 1

            for variant in range(variants):

                if split == "train":
                    # Jitter camera position while keeping target nearby.
                    jitter_x = random.uniform(-0.20, 0.20) * crop_w
                    jitter_y = random.uniform(-0.20, 0.20) * crop_h
                else:
                    jitter_x = 0
                    jitter_y = 0

                center_x = object_cx + jitter_x
                center_y = object_cy + jitter_y

                crop = make_crop(
                    center_x,
                    center_y,
                    crop_w,
                    crop_h
                )

                name = (
                    f"{ann_path.stem}"
                    f"_obj{object_index:02d}"
                    f"_L{level}"
                    f"_v{variant}"
                )

                n = save_sample(
                    image,
                    annotations,
                    crop,
                    split,
                    name
                )

                if n:
                    sample_count[split] += 1
                    label_count[split] += n


print()
print("Camera-aware dataset created")
print("Location:", OUT.resolve())
print()

for split in ["train", "val"]:
    print(
        f"{split:5s}: "
        f"{sample_count[split]} images, "
        f"{label_count[split]} labels"
    )
