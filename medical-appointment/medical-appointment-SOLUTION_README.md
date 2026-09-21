# GreatAI Solution — Medical Appointment

## Solution

The system receives one medical consultation recording and ten yes/no questions. Positive answers also require localized temporal evidence.

The challenge score is:

`Score = 0.4 × Accuracy + 0.6 × mean tIoU`

## Architecture

Base64 audio → audio decoding → Faster-Whisper → timestamped transcript → shared processing for ten questions → candidate evidence windows → lexical/numeric matching → contradiction checks → yes/no answer + temporal evidence.

The consultation is transcribed **once per request** and reused for all ten questions.

## Final V3.1

Active runtime: `example.py`  
Frozen champion: `example_v31_champion.py`  
Submission snapshot: `example_SUBMISSION_FINAL.py`

Verified SHA-256:

`2a2d950b482ddccd394b320fa25b2ffc72567640343847599a56ef3a3614a6fc`

## Local Development Results

- Questions: 390
- Correct: 301
- Positive: 154/195
- Hard negative: 95/142
- Off-topic: 52/53
- Accuracy: 0.772
- Mean tIoU: 0.372
- Combined score: 0.532
- Mean conversation runtime: ~5.3 s
- Worst conversation runtime: ~10.3 s

These are local development results, not official leaderboard scores.

Offline oracle evidence diagnostics indicated additional ranking headroom, with approximately 0.743 oracle-window mean tIoU. This is a diagnostic upper-bound experiment, not a production score.

## Run

```bash
conda activate nordic-medical
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cublas/lib:$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
unset DUMP_TRANSCRIPTS
python api.py
```

Server: `0.0.0.0:9054`

```bash
curl -i http://127.0.0.1:9054/
```

Prediction endpoint: `POST /predict`

Temporary public endpoint:

```bash
cloudflared tunnel --url http://127.0.0.1:9054
```

## Restore Frozen Version

```bash
cp example_v31_champion.py example.py
python -m py_compile example.py
```

No external cloud inference API is required in the prediction path.
