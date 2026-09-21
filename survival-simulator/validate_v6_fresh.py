import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2
from src.utils.controllers.evohive_v6_fruitlock import action_decision as v6

SEEDS = [
    20011, 20327, 20639, 20963, 21283,
    21611, 21929, 22247, 22571, 22877,
    23203, 23531, 23857, 24181, 24509,
    24821, 25147, 25469, 25793, 26119,
    26437, 26759, 27077, 27397, 27733,
    28051, 28387, 28703, 29027, 29363,
    29683, 30011, 30347, 30671, 31013,
    31337, 31667, 31991, 32321, 32647,
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
            return float(state["sim_time"]), float(sim.env.score), max_pop

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            action = policy(obs, rng)
            actions.append((agent.agent_id, action))


v2r = []
v6r = []

for i, seed in enumerate(SEEDS, 1):
    a = run(seed, v2)
    b = run(seed, v6)

    v2r.append(a)
    v6r.append(b)

    print(
        f"{i:02d}/40 seed={seed} "
        f"V2={a[0]:7.1f}/{a[1]:8.2f}/{a[2]:2d} "
        f"V6={b[0]:7.1f}/{b[1]:8.2f}/{b[2]:2d} "
        f"dT={b[0]-a[0]:+7.1f}"
    )


def summary(name, rows):
    times = [x[0] for x in rows]
    scores = [x[1] for x in rows]
    pops = [x[2] for x in rows]

    q = statistics.quantiles(times, n=4)

    print(
        f"{name}: "
        f"mean={statistics.mean(times):.1f} "
        f"median={statistics.median(times):.1f} "
        f"q25={q[0]:.1f} "
        f"min={min(times):.1f} "
        f"max={max(times):.1f} "
        f"meanScore={statistics.mean(scores):.2f} "
        f"avgMaxPop={statistics.mean(pops):.1f} "
        f"<100={sum(t < 100 for t in times)}/40 "
        f"<200={sum(t < 200 for t in times)}/40 "
        f">=750={sum(t >= 750 for t in times)}/40 "
        f">=1000={sum(t >= 1000 for t in times)}/40 "
        f"caps={sum(t >= 3000 for t in times)}/40"
    )


print("\nSUMMARY")
summary("V2", v2r)
summary("V6", v6r)

dt = [b[0] - a[0] for a, b in zip(v2r, v6r)]
ds = [b[1] - a[1] for a, b in zip(v2r, v6r)]

print("\nPAIRED")
print("wins       =", sum(x > 0 for x in dt))
print("losses     =", sum(x < 0 for x in dt))
print("ties       =", sum(x == 0 for x in dt))
print(f"mean dT    = {statistics.mean(dt):+.1f}")
print(f"median dT  = {statistics.median(dt):+.1f}")
print(f"mean dScore= {statistics.mean(ds):+.2f}")
