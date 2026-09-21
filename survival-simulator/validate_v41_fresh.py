import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2
from src.utils.controllers.evohive_v41_adaptive import action_decision as v41

SEEDS = [
    167, 389, 617, 911, 1283,
    1579, 1877, 2137, 2473, 2837,
    3163, 3499, 3821, 4153, 4481,
    4813, 5171, 5503, 5849, 6197,
    6521, 6869, 7211, 7547, 7901,
    8237, 8597, 8951, 9283, 9643,
]


def run(seed, policy):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    while True:
        state = sim.step(actions)

        if state["num_agents"] == 0 or state["sim_time"] >= 3000:
            return float(state["sim_time"]), float(sim.env.score)

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            action = policy(obs, rng)
            actions.append((agent.agent_id, action))


def summarize(values):
    survivals = [x[0] for x in values]
    scores = [x[1] for x in values]
    q = statistics.quantiles(survivals, n=4)

    return {
        "mean": statistics.mean(survivals),
        "median": statistics.median(survivals),
        "q25": q[0],
        "min": min(survivals),
        "max": max(survivals),
        "score": statistics.mean(scores),
        "caps": sum(t >= 3000 for t in survivals),
        "under200": sum(t < 200 for t in survivals),
    }


v2_results = []
v41_results = []

wins = losses = ties = 0
deltas = []

for seed in SEEDS:
    a = run(seed, v2)
    b = run(seed, v41)

    v2_results.append(a)
    v41_results.append(b)

    delta = b[0] - a[0]
    deltas.append(delta)

    if delta > 0.05:
        wins += 1
    elif delta < -0.05:
        losses += 1
    else:
        ties += 1

    print(
        f"{seed:5d}  "
        f"V2={a[0]:7.1f}  "
        f"V41={b[0]:7.1f}  "
        f"delta={delta:+7.1f}"
    )

print("\nSUMMARY")

for name, results in [
    ("V2 ", v2_results),
    ("V41", v41_results),
]:
    s = summarize(results)

    print(
        f"{name} "
        f"mean={s['mean']:7.1f} "
        f"median={s['median']:7.1f} "
        f"q25={s['q25']:7.1f} "
        f"min={s['min']:7.1f} "
        f"max={s['max']:7.1f} "
        f"meanScore={s['score']:8.2f} "
        f"under200={s['under200']}/30 "
        f"caps={s['caps']}/30"
    )

print(
    "\nV41 vs V2: "
    f"wins={wins} losses={losses} ties={ties} "
    f"mean_delta={statistics.mean(deltas):+.1f} "
    f"median_delta={statistics.median(deltas):+.1f}"
)
