# GreatAI Solution — Drone Flyby

## FlySight

GreatAI's **FlySight** solution uses YOLO for low-latency drone-image object detection.

Final portable model:

`flysight_v2_best.pt`

Active runtime: `example.py`  
API: `api.py`

The publication runtime should load:

```python
MODEL_PATH = Path(__file__).resolve().parent / "flysight_v2_best.pt"
```

## Pipeline

Incoming image → decoding → YOLO inference → bounding-box extraction → coordinate conversion → competition response.

## Local Development Results

- Requests accepted: 25/25
- Skipped: 0
- Timeouts: 0
- Errors: 0
- Observed RTT: approximately 25 ms
- Development mAP50: approximately 0.982

The mAP50 result was measured on available development imagery used during development, so it is an in-sample measurement rather than hidden-set competition performance.

## Run

```bash
conda activate nordic-drone
python api.py
```

Server: `0.0.0.0:9053`

```bash
curl -i http://127.0.0.1:9053/
```

Prediction endpoint: `POST /predict`

Temporary public endpoint:

```bash
cloudflared tunnel --url http://127.0.0.1:9053
```

## Reproducibility

Large source imagery and generated training runs are excluded from the publication repository. The trained final model is included.
