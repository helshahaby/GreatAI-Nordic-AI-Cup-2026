import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2
from src.utils.controllers.evohive_v6_fruitlock import action_decision as v6

SEEDS = [5521, 3253, 853, 8597, 10061]


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


results = []

for seed in SEEDS:
    a = run(seed, v2)
    b = run(seed, v6)

    results.append((seed, a, b))

    print(
        f"seed={seed:5d} "
        f"V2 time={a[0]:7.1f} score={a[1]:8.2f} maxPop={a[2]:2d} | "
        f"V6 time={b[0]:7.1f} score={b[1]:8.2f} maxPop={b[2]:2d} | "
        f"dT={b[0]-a[0]:+7.1f} "
        f"dScore={b[1]-a[1]:+8.2f}"
    )

dts = [b[0] - a[0] for _, a, b in results]
dscores = [b[1] - a[1] for _, a, b in results]

print("\nSUMMARY")
print(f"wins       = {sum(x > 0 for x in dts)}")
print(f"losses     = {sum(x < 0 for x in dts)}")
print(f"ties       = {sum(x == 0 for x in dts)}")
print(f"mean dT    = {statistics.mean(dts):+.1f}")
print(f"median dT  = {statistics.median(dts):+.1f}")
print(f"mean dScore= {statistics.mean(dscores):+.2f}")
