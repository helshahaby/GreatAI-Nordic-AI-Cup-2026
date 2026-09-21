import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v11_frozen import action_decision as v11
from src.utils.controllers.evohive_emergency_20 import action_decision as e20

SEEDS = [
    314, 628, 941, 1257, 1717,
    2345, 2718, 3333, 4242, 4871,
    6007, 6789, 7123, 8080, 8642,
    9001, 10101, 12345, 13579, 24680,
    27182, 31415, 42424, 54321, 65537,
    77771, 88883, 99991, 104729, 130363,
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


def evaluate(name, policy):
    results = []

    print("\n" + name)

    for seed in SEEDS:
        score, survival = run(seed, policy)
        results.append((seed, score, survival))
        print(
            f"{seed:<7} score={score:10.3f} "
            f"time={survival:8.1f}"
        )

    times = [x[2] for x in results]
    scores = [x[1] for x in results]
    q25 = statistics.quantiles(times, n=4)[0]

    print(
        f"{name} SUMMARY | "
        f"mean={statistics.mean(times):.1f} | "
        f"median={statistics.median(times):.1f} | "
        f"q25={q25:.1f} | "
        f"min={min(times):.1f} | "
        f"meanScore={statistics.mean(scores):.2f} | "
        f"caps={sum(t >= 3000 for t in times)}/{len(times)}"
    )

    return results


base = evaluate("V11", v11)
candidate = evaluate("E20", e20)

wins = losses = ties = 0

print("\nDELTAS")

for (seed, _, bt), (_, _, ct) in zip(base, candidate):
    delta = ct - bt

    if delta > 0.05:
        wins += 1
    elif delta < -0.05:
        losses += 1
    else:
        ties += 1

    print(
        f"seed={seed:<7} "
        f"V11={bt:8.1f} "
        f"E20={ct:8.1f} "
        f"delta={delta:+8.1f}"
    )

print(
    f"\nHEAD_TO_HEAD wins={wins} "
    f"losses={losses} ties={ties}"
)
