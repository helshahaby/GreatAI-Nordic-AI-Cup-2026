# GreatAI — Nordic AI Cup 2026

Solutions developed by **GreatAI** for all three Nordic AI Cup 2026 challenges.

**Developer:** Hossam Elshahaby

## Challenges

| Challenge | Solution | API Port |
|---|---|---:|
| Survival Simulator | EvoHive V2-KITE multi-agent controller | 9052 |
| Drone Flyby | FlySight YOLO real-time detector | 9053 |
| Medical Appointment | Faster-Whisper + clinical QA + temporal evidence | 9054 |

Each challenge directory keeps the organizer's original `README.md` and adds a separate `SOLUTION_README.md` documenting the GreatAI solution.

## Final Components

- Survival: `survival-simulator/src/utils/controllers/evohive_final.py`
- Drone: `drone-flyby/flysight_v2_best.pt`
- Medical: `medical-appointment/example_v31_champion.py`

## Local Development Results

Survival V2-KITE reached a mean local score of **583.01** over 40 development seeds.

Drone FlySight processed **25/25** local real-time requests with no skips, timeouts, or errors and about **25 ms** observed RTT. Its in-sample development mAP50 was approximately **0.982**.

Medical V3.1 reached **0.772 accuracy**, **0.372 mean tIoU**, and **0.532 combined score** on the local development evaluator.

These are development measurements, not official competition leaderboard scores.

## Hardware

Ubuntu with NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM).

## Security

Do not commit API keys, `.env` files, passwords, credentials, or temporary tunnel secrets.
