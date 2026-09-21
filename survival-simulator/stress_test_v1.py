import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_policy import action_decision


SEEDS = [
    1, 7, 13, 21, 42,
    77, 101, 123, 256, 512,
    777, 999, 1337, 1600, 2026,
    3001, 4096, 5555, 7777, 9999
]


def run(seed):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    while True:
        state = sim.step(actions)

        if state["num_agents"] == 0 or sim.env.time > 3000:
            return float(state["score"]), float(sim.env.time)

        actions = []

        for agent, agent_state in zip(
            sim.env.agents,
            state["observations"]
        ):
            if agent_state is None:
                continue

            action = action_decision(agent_state, rng)
            actions.append((agent.agent_id, action))


results = []

for seed in SEEDS:
    score, survival = run(seed)
    results.append((seed, score, survival))

    print(
        f"seed={seed:<5} "
        f"score={score:10.3f} "
        f"survival={survival:8.1f}"
    )


scores = [r[1] for r in results]
times = [r[2] for r in results]

print("\n===== EVOHIVE V1 STRESS TEST =====")
print(f"runs             = {len(results)}")
print(f"mean score       = {statistics.mean(scores):.3f}")
print(f"median score     = {statistics.median(scores):.3f}")
print(f"min score        = {min(scores):.3f}")
print(f"max score        = {max(scores):.3f}")
print(f"mean survival    = {statistics.mean(times):.1f}")
print(f"median survival  = {statistics.median(times):.1f}")
print(f"min survival     = {min(times):.1f}")
print(f"max survival     = {max(times):.1f}")

full = sum(t >= 3000 for t in times)
print(f"3000s survivors  = {full}/{len(times)}")

print("\nWorst five:")
for seed, score, survival in sorted(results, key=lambda x: x[1])[:5]:
    print(
        f"seed={seed:<5} "
        f"score={score:10.3f} "
        f"survival={survival:8.1f}"
    )
