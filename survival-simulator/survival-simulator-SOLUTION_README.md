# GreatAI Solution — Survival Simulator

## Solution

GreatAI developed **EvoHive V2-KITE**, a multi-agent survival controller balancing resource acquisition with distance-aware predator avoidance.

Final frozen controller:

`src/utils/controllers/evohive_final.py`

Reference implementation:

`src/utils/controllers/evohive_v2_kite.py`

Both were verified with SHA-256:

`8ef9a8128cfbce9f2349be70fab55f37765dfedbf4bb3487d92a68117e43ab2b`

## Local Development Benchmark

- Mean score: 583.01
- Mean survival: 555.1 s
- Median survival: 553.1 s
- 25th percentile: 432.2 s
- Minimum: 92.6 s
- Maximum: 910.9 s

These are local simulator measurements, not official leaderboard results.

## Run

```bash
conda activate nordic-survival
python agent_server.py
```

Server: `0.0.0.0:9052`

Health check:

```bash
curl -i http://127.0.0.1:9052/
```

Prediction endpoint: `POST /predict`

For temporary external access:

```bash
cloudflared tunnel --url http://127.0.0.1:9052
```

Use `https://<generated-host>.trycloudflare.com/predict`.

## Development

Multiple predator, emergency, reproduction, foraging, adaptive movement, optimization, and stress-test variants were evaluated. `evohive_final.py` is the frozen selected controller.
