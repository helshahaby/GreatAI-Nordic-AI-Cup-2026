import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2
from src.utils.controllers.evohive_v6_fruitlock import action_decision as v6
from src.utils.controllers.evohive_v62_emergency_fruit import action_decision as v62

SEEDS = [
    5521, 3253, 853, 8597, 10061,
    21283, 24509, 24821, 27077, 32321,
]


def run(seed, policy):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []
    max_pop = 0

    while True:
        state = sim.step(actions)
        max_pop = max(max_pop, int(state["num_agents"]))

        if state["num_agents"] == 0 or state["sim_time"] >= 3000:
            return (
                float(state["sim_time"]),
                float(sim.env.score),
                max_pop,
            )

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            action = policy(obs, rng)
            actions.append((agent.agent_id, action))


rows = []

for seed in SEEDS:
    a = run(seed, v2)
    b = run(seed, v6)
    c = run(seed, v62)

    rows.append((seed, a, b, c))

    print(
        f"seed={seed:5d} | "
        f"V2={a[0]:7.1f}/{a[1]:8.2f}/{a[2]:2d} | "
        f"V6={b[0]:7.1f}/{b[1]:8.2f}/{b[2]:2d} | "
        f"V62={c[0]:7.1f}/{c[1]:8.2f}/{c[2]:2d} | "
        f"d62-v2={c[0]-a[0]:+7.1f}"
    )


def report(label, index):
    values = [row[index][0] for row in rows]
    scores = [row[index][1] for row in rows]
    pops = [row[index][2] for row in rows]

    print(
        f"{label}: "
        f"meanT={statistics.mean(values):.1f} "
        f"medianT={statistics.median(values):.1f} "
        f"meanScore={statistics.mean(scores):.2f} "
        f"avgMaxPop={statistics.mean(pops):.1f}"
    )


print("\nALL 10")
report("V2 ", 1)
report("V6 ", 2)
report("V62", 3)

for title, subset in [
    ("RESCUE GROUP", rows[:5]),
    ("REGRESSION GROUP", rows[5:]),
]:
    print(f"\n{title}")

    for name, idx in [("V2", 1), ("V6", 2), ("V62", 3)]:
        times = [row[idx][0] for row in subset]
        scores = [row[idx][1] for row in subset]

        print(
            f"{name}: "
            f"meanT={statistics.mean(times):.1f} "
            f"medianT={statistics.median(times):.1f} "
            f"meanScore={statistics.mean(scores):.2f}"
        )


dt = [c[0] - a[0] for _, a, _, c in rows]
ds = [c[1] - a[1] for _, a, _, c in rows]

print("\nV62 vs V2")
print("wins       =", sum(x > 0 for x in dt))
print("losses     =", sum(x < 0 for x in dt))
print("ties       =", sum(x == 0 for x in dt))
print(f"mean dT    = {statistics.mean(dt):+.1f}")
print(f"median dT  = {statistics.median(dt):+.1f}")
print(f"mean dScore= {statistics.mean(ds):+.2f}")
