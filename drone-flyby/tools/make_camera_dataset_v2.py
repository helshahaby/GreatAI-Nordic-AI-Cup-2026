import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2


random.seed(42)

ROOT = Path("src/helsinki")
OUT = Path("datasets/nordic_camera_v2")

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

OUTPUT_W = 960
OUTPUT_H = 540

LEVELS = {
    0: (3840, 2160),
    1: (1920, 1080),
    2: (960, 540),
}

# ---------------------------------------------------------
# V2 strategy
# ---------------------------------------------------------
#
# Competition validation/evaluation is on separate data.
#
# We already used the 20/5 split to diagnose the detector.
# Now use ALL 25 labelled Helsinki frames for training.
#
# A small diagnostic val set is generated from all frames too.
# It is NOT an independent estimate of competition performance.
# local_evaluator.py remains our real A/B benchmark.
#

ALL_FRAMES = set(range(25))

# Extra target-centered augmentation for data-starved classes.
#
# medium_plane: 0 previous training examples
# hangar:       only 7 previous training examples
# medium_launcher: weaker AP
# mine_roller:  rare
#
RARE_MULTIPLIER = {
    "hangar": 10,
    "medium_plane": 12,
    "medium_launcher": 5,
    "mine_roller": 5,
}

NORMAL_VARIANTS = 3

# More conservative jitter for rare targets so the object
# remains useful inside the crop.
NORMAL_JITTER = 0.20
RARE_JITTER = 0.12


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

    original_area = max(
        1,
        (x2 - x1) * (y2 - y1)
    )

    visible_area = (
        (ix2 - ix1) *
        (iy2 - iy1)
    )

    return visible_area / original_area


def make_crop(
    center_x,
    center_y,
    crop_w,
    crop_h,
):
    x1 = int(
        round(center_x - crop_w / 2)
    )

    y1 = int(
        round(center_y - crop_h / 2)
    )

    x1 = max(
        0,
        min(SOURCE_W - crop_w, x1)
    )

    y1 = max(
        0,
        min(SOURCE_H - crop_h, y1)
    )

    return (
        x1,
        y1,
        x1 + crop_w,
        y1 + crop_h,
    )


def save_sample(
    image,
    annotations,
    crop,
    split,
    name,
):
    cx1, cy1, cx2, cy2 = crop

    cropped = image[
        cy1:cy2,
        cx1:cx2
    ]

    if cropped.size == 0:
        return 0

    resized = cv2.resize(
        cropped,
        (OUTPUT_W, OUTPUT_H),
        interpolation=cv2.INTER_AREA,
    )

    labels = []

    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    for a in annotations:

        cls = a["object_id"]

        if cls not in CLASS_TO_ID:
            continue

        box = a["bbox"]

        clipped = intersection(
            box,
            crop,
        )

        if clipped is None:
            continue

        # Retain the same rule as V1.
        if visibility(box, clipped) < 0.50:
            continue

        x1, y1, x2, y2 = clipped

        x1 -= cx1
        x2 -= cx1
        y1 -= cy1
        y2 -= cy1

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

        xc = (
            (x1 + x2) / 2
        ) / OUTPUT_W

        yc = (
            (y1 + y2) / 2
        ) / OUTPUT_H

        nw = bw / OUTPUT_W
        nh = bh / OUTPUT_H

        labels.append(
            f"{CLASS_TO_ID[cls]} "
            f"{xc:.8f} "
            f"{yc:.8f} "
            f"{nw:.8f} "
            f"{nh:.8f}"
        )

    # Positive samples only, same as V1.
    if not labels:
        return 0

    image_path = (
        OUT /
        "images" /
        split /
        f"{name}.jpg"
    )

    label_path = (
        OUT /
        "labels" /
        split /
        f"{name}.txt"
    )

    cv2.imwrite(
        str(image_path),
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    label_path.write_text(
        "\n".join(labels)
    )

    return len(labels)


# ---------------------------------------------------------
# Start clean
# ---------------------------------------------------------

if OUT.exists():
    shutil.rmtree(OUT)

for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True,
    )


sample_count = {
    "train": 0,
    "val": 0,
}

label_count = {
    "train": 0,
    "val": 0,
}


# ---------------------------------------------------------
# Process all labelled frames
# ---------------------------------------------------------

for ann_path in sorted(
    (ROOT / "annotations").glob(
        "frame_*.json"
    )
):

    data = json.loads(
        ann_path.read_text()
    )

    frame = int(data["frame"])

    if frame not in ALL_FRAMES:
        continue

    image_path = (
        ROOT /
        "images" /
        f"{ann_path.stem}.png"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print(
            "Could not load:",
            image_path
        )
        continue

    annotations = data[
        "annotations"
    ]


    # =====================================================
    # TRAINING
    # =====================================================

    split = "train"


    # -----------------------------------------------------
    # L0 full-frame sample
    # -----------------------------------------------------

    crop = (
        0,
        0,
        SOURCE_W,
        SOURCE_H,
    )

    n = save_sample(
        image,
        annotations,
        crop,
        split,
        f"{ann_path.stem}_L0",
    )

    if n:
        sample_count[split] += 1
        label_count[split] += n


    # -----------------------------------------------------
    # L1 / L2 target-centered training samples
    # -----------------------------------------------------

    for object_index, a in enumerate(
        annotations
    ):

        cls = a["object_id"]

        x1, y1, x2, y2 = a["bbox"]

        object_cx = (
            x1 + x2
        ) / 2

        object_cy = (
            y1 + y2
        ) / 2


        if cls in RARE_MULTIPLIER:

            variants = (
                NORMAL_VARIANTS *
                RARE_MULTIPLIER[cls]
            )

            jitter_fraction = (
                RARE_JITTER
            )

        else:

            variants = (
                NORMAL_VARIANTS
            )

            jitter_fraction = (
                NORMAL_JITTER
            )


        for level in [1, 2]:

            crop_w, crop_h = (
                LEVELS[level]
            )

            for variant in range(
                variants
            ):

                # Variant zero is always centered exactly.
                if variant == 0:

                    jitter_x = 0
                    jitter_y = 0

                else:

                    jitter_x = (
                        random.uniform(
                            -jitter_fraction,
                            jitter_fraction,
                        )
                        * crop_w
                    )

                    jitter_y = (
                        random.uniform(
                            -jitter_fraction,
                            jitter_fraction,
                        )
                        * crop_h
                    )


                center_x = (
                    object_cx +
                    jitter_x
                )

                center_y = (
                    object_cy +
                    jitter_y
                )

                crop = make_crop(
                    center_x,
                    center_y,
                    crop_w,
                    crop_h,
                )

                name = (
                    f"{ann_path.stem}"
                    f"_obj{object_index:02d}"
                    f"_{cls}"
                    f"_L{level}"
                    f"_v{variant:03d}"
                )

                n = save_sample(
                    image,
                    annotations,
                    crop,
                    split,
                    name,
                )

                if n:
                    sample_count[
                        split
                    ] += 1

                    label_count[
                        split
                    ] += n


    # =====================================================
    # DIAGNOSTIC VALIDATION
    # =====================================================
    #
    # One deterministic L1 and one deterministic L2
    # observation for each target.
    #
    # These come from training source frames, therefore
    # this is only a sanity-check set.
    #

    split = "val"

    for object_index, a in enumerate(
        annotations
    ):

        x1, y1, x2, y2 = a["bbox"]

        object_cx = (
            x1 + x2
        ) / 2

        object_cy = (
            y1 + y2
        ) / 2

        for level in [1, 2]:

            crop_w, crop_h = (
                LEVELS[level]
            )

            crop = make_crop(
                object_cx,
                object_cy,
                crop_w,
                crop_h,
            )

            name = (
                f"{ann_path.stem}"
                f"_obj{object_index:02d}"
                f"_L{level}"
                f"_diag"
            )

            n = save_sample(
                image,
                annotations,
                crop,
                split,
                name,
            )

            if n:
                sample_count[
                    split
                ] += 1

                label_count[
                    split
                ] += n


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

print()
print(
    "Nordic camera V2 dataset created"
)

print(
    "Location:",
    OUT.resolve(),
)

print()

for split in [
    "train",
    "val",
]:

    print(
        f"{split:5s}: "
        f"{sample_count[split]} images, "
        f"{label_count[split]} labels"
    )


# ---------------------------------------------------------
# Class distribution
# ---------------------------------------------------------

for split in [
    "train",
    "val",
]:

    counts = Counter()
    image_counts = Counter()

    label_dir = (
        OUT /
        "labels" /
        split
    )

    for label_file in (
        label_dir.glob("*.txt")
    ):

        seen = set()

        for line in (
            label_file
            .read_text()
            .splitlines()
        ):

            if not line.strip():
                continue

            cls_id = int(
                line.split()[0]
            )

            counts[cls_id] += 1
            seen.add(cls_id)

        for cls_id in seen:
            image_counts[
                cls_id
            ] += 1


    print()
    print(
        "=" * 72
    )

    print(
        split.upper()
    )

    print(
        "=" * 72
    )

    for cls_id, cls_name in enumerate(
        CLASSES
    ):

        print(
            f"{cls_id:2d} "
            f"{cls_name:18s} "
            f"boxes={counts[cls_id]:5d} "
            f"images={image_counts[cls_id]:5d}"
        )
