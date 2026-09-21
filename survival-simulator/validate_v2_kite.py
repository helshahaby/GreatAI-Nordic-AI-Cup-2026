import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v11_frozen import action_decision as v11
from src.utils.controllers.evohive_v2_kite import action_decision as v2

SEEDS = [
    271, 433, 587, 761, 887,
    1031, 1193, 1429, 1699, 1877,
    2081, 2269, 2539, 2791, 2953,
    3203, 3469, 3719, 4001, 4253,
    4513, 4783, 5023, 5297, 5501,
    5801, 6073, 6323, 6599, 6829,
    7103, 7331, 7583, 7879, 8101,
    8363, 8623, 8923, 9209, 9473,
    9733, 10037, 10301, 10613, 10909,
    11213, 11519, 11807, 12101, 12409,
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

        for agent, obs in zip(sim.env.agents, state["observations"]):
            if obs is None:
                continue

            action = policy(obs, rng)
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

    times = [r[2] for r in results]
    scores = [r[1] for r in results]
    q = statistics.quantiles(times, n=4)

    print(
        f"{name} SUMMARY | "
        f"mean={statistics.mean(times):.1f} | "
        f"median={statistics.median(times):.1f} | "
        f"q25={q[0]:.1f} | "
        f"q75={q[2]:.1f} | "
        f"min={min(times):.1f} | "
        f"max={max(times):.1f} | "
        f"meanScore={statistics.mean(scores):.2f} | "
        f"caps={sum(t >= 3000 for t in times)}/{len(times)}"
    )

    return results

base = evaluate("V11", v11)
candidate = evaluate("V2-KITE", v2)

wins = 0
losses = 0
ties = 0
deltas = []

print("\nDELTAS")

for (seed, _, bt), (_, _, ct) in zip(base, candidate):
    delta = ct - bt
    deltas.append(delta)

    if delta > 0.05:
        wins += 1
    elif delta < -0.05:
        losses += 1
    else:
        ties += 1

    print(
        f"seed={seed:<7} "
        f"V11={bt:8.1f} "
        f"V2={ct:8.1f} "
        f"delta={delta:+8.1f}"
    )

print(
    f"\nRESULT | "
    f"wins={wins} | "
    f"losses={losses} | "
    f"ties={ties} | "
    f"mean_delta={statistics.mean(deltas):+.1f} | "
    f"median_delta={statistics.median(deltas):+.1f}"
)
