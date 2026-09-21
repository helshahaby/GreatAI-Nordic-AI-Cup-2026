import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v11
from src.utils.controllers.evohive_v4_diverse import action_decision as v2
from src.utils.controllers.evohive_opt_predator import action_decision as opt


SEEDS = [
    151, 457, 853, 1217, 1741,
    2267, 2741, 3253, 3761, 4271,
    4787, 5303, 5801, 6311, 6823,
    7333, 7841, 8353, 8861, 9371,
]


OPT24 = {
    "sprint_distance": 115.88017318499261,
    "partial_distance": 140.5951985780688,
    "partial_fraction": 0.19417187305192546,
    "face_distance": 86.29180972751521,
    "close_turn": 0.3033552639554437,
}


def run(seed, policy, params=None):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    while True:
        state = sim.step(actions)

        if state["num_agents"] == 0 or state["sim_time"] >= 3000:
            return float(state["sim_time"]), float(state["score"])

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            if obs is None:
                continue

            if params is None:
                action = policy(obs, rng)
            else:
                action = policy(obs, rng, params)

            actions.append((agent.agent_id, action))


def q25(xs):
    return statistics.quantiles(xs, n=4, method="inclusive")[0]


results = {
    "V11": [],
    "V2": [],
    "OPT24": [],
}

for seed in SEEDS:
    a, _ = run(seed, v11)
    b, _ = run(seed, v2)
    c, _ = run(seed, opt, OPT24)

    results["V11"].append(a)
    results["V2"].append(b)
    results["OPT24"].append(c)

    print(
        f"{seed:5d}  "
        f"V11={a:7.1f}  "
        f"V2={b:7.1f}  "
        f"OPT24={c:7.1f}  "
        f"dV2={c-b:+7.1f}"
    )


print("\nSUMMARY")

for name, xs in results.items():
    print(
        f"{name:5s} "
        f"mean={statistics.mean(xs):7.1f} "
        f"median={statistics.median(xs):7.1f} "
        f"q25={q25(xs):7.1f} "
        f"min={min(xs):7.1f} "
        f"max={max(xs):7.1f} "
        f"caps={sum(x >= 2999.9 for x in xs)}/{len(xs)}"
    )


v2xs = results["V2"]
oxs = results["OPT24"]

wins = sum(o > v for o, v in zip(oxs, v2xs))
losses = sum(o < v for o, v in zip(oxs, v2xs))
ties = len(SEEDS) - wins - losses

deltas = [o-v for o, v in zip(oxs, v2xs)]

print(
    "\nOPT24 vs V2:"
    f" wins={wins}"
    f" losses={losses}"
    f" ties={ties}"
    f" mean_delta={statistics.mean(deltas):+.1f}"
    f" median_delta={statistics.median(deltas):+.1f}"
)
