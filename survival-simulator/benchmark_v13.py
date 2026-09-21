import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v11_frozen import action_decision as champion
from src.utils.controllers.evohive_v13_policy import action_decision as candidate

SEEDS = [
    1, 7, 13, 21, 42,
    77, 101, 123, 256, 512,
    777, 999, 1337, 1600, 2026,
    3001, 4096, 5555, 7777, 9999,
]


def run(seed, policy):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    while True:
        state = sim.step(actions)

        if state["num_agents"] == 0 or state["sim_time"] >= 3000:
            return state["score"], state["sim_time"]

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
    results = {}

    print()
    print(name)

    for seed in SEEDS:
        score, survival = run(seed, policy)
        results[seed] = (score, survival)

        print(
            f"seed={seed:<5} "
            f"score={score:10.3f} "
            f"survival={survival:8.1f}"
        )

    scores = [x[0] for x in results.values()]
    survival = [x[1] for x in results.values()]

    print(
        f"{name} SUMMARY | "
        f"mean_score={statistics.mean(scores):.3f} | "
        f"median_score={statistics.median(scores):.3f} | "
        f"min_score={min(scores):.3f} | "
        f"mean_survival={statistics.mean(survival):.1f} | "
        f"median_survival={statistics.median(survival):.1f} | "
        f"min_survival={min(survival):.1f} | "
        f"3000s={sum(t >= 3000 for t in survival)}/{len(SEEDS)}"
    )

    return results


base = benchmark("V11_FROZEN", champion)
test = benchmark("V13", candidate)

wins = 0
losses = 0
ties = 0

print()
print("PER-SEED DELTAS")

for seed in SEEDS:
    bs, bt = base[seed]
    cs, ct = test[seed]

    delta = ct - bt

    if delta > 0.05:
        wins += 1
    elif delta < -0.05:
        losses += 1
    else:
        ties += 1

    print(
        f"seed={seed:<5} "
        f"V11={bt:8.1f} "
        f"V13={ct:8.1f} "
        f"delta={delta:+8.1f}"
    )

print()
print(
    f"HEAD_TO_HEAD wins={wins} "
    f"losses={losses} ties={ties}"
)
