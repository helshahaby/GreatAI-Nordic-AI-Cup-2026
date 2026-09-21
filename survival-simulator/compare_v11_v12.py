import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v11_frozen import action_decision as v1
from src.utils.controllers.evohive_v12_policy import action_decision as v11


SEEDS = [
    1, 7, 13, 21, 42,
    77, 101, 123, 256, 512,
    777, 999, 1337, 1600, 2026,
    3001, 4096, 5555, 7777, 9999
]


def run(seed, policy):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    while True:
        state = sim.step(actions)

        if state["num_agents"] == 0 or sim.env.time >= 3000:
            return float(state["score"]), float(sim.env.time)

        actions = []

        for agent, agent_state in zip(
            sim.env.agents,
            state["observations"]
        ):
            if agent_state is None:
                continue

            action = policy(agent_state, rng)
            actions.append((agent.agent_id, action))


def benchmark(name, policy):
    results = []

    print(f"\\n===== {name} =====")

    for seed in SEEDS:
        score, survival = run(seed, policy)
        results.append((seed, score, survival))

        print(
            f"seed={seed:<5} "
            f"score={score:10.3f} "
            f"survival={survival:8.1f}"
        )

    scores = [x[1] for x in results]
    times = [x[2] for x in results]

    print(
        f"{name} | "
        f"mean_score={statistics.mean(scores):.3f} | "
        f"median_score={statistics.median(scores):.3f} | "
        f"min_score={min(scores):.3f} | "
        f"mean_survival={statistics.mean(times):.1f} | "
        f"median_survival={statistics.median(times):.1f} | "
        f"min_survival={min(times):.1f} | "
        f"3000s={sum(t >= 3000 for t in times)}/{len(times)}"
    )

    return results


r1 = benchmark("EVOHIVE_V12", v1)
r11 = benchmark("EVOHIVE_V121", v11)

print("\\n===== PER-SEED CHANGE V1.2 - V1.1 =====")

wins = 0
losses = 0

for (seed, s1, t1), (_, s11, t11) in zip(r1, r11):
    delta = s11 - s1

    if delta > 0:
        wins += 1
    elif delta < 0:
        losses += 1

    print(
        f"seed={seed:<5} "
        f"score_delta={delta:+10.3f} "
        f"survival_delta={t11-t1:+8.1f}"
    )

print(f"wins={wins} losses={losses} ties={len(SEEDS)-wins-losses}")
