import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2
from src.utils.controllers.evohive_v5_scan import action_decision as v5

SEEDS = [
    193, 431, 673, 977, 1231,
    1489, 1783, 2063, 2381, 2689,
    3011, 3319, 3631, 3943, 4241,
    4561, 4877, 5189, 5521, 5851,
    6151, 6481, 6791, 7121, 7481,
    7793, 8111, 8443, 8761, 9091,
    9421, 9743, 10061, 10391, 10711,
    11027, 11351, 11677, 12011, 12343,
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


def run_pair(seed):
    a = run(seed, v2)
    b = run(seed, v5)
    return seed, a, b


def summary(results):
    survival = [x[0] for x in results]
    scores = [x[1] for x in results]
    q = statistics.quantiles(survival, n=4)

    return {
        "mean": statistics.mean(survival),
        "median": statistics.median(survival),
        "q25": q[0],
        "min": min(survival),
        "max": max(survival),
        "score": statistics.mean(scores),
        "under100": sum(x < 100 for x in survival),
        "under200": sum(x < 200 for x in survival),
        "over750": sum(x >= 750 for x in survival),
        "over1000": sum(x >= 1000 for x in survival),
        "caps": sum(x >= 3000 for x in survival),
    }


if __name__ == "__main__":
    rows = []

    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = {
            pool.submit(run_pair, seed): seed
            for seed in SEEDS
        }

        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda x: x[0])

    v2_results = []
    v5_results = []
    deltas = []

    for seed, a, b in rows:
        v2_results.append(a)
        v5_results.append(b)
        delta = b[0] - a[0]
        deltas.append(delta)

        print(
            f"{seed:5d} "
            f"V2={a[0]:7.1f} "
            f"V5={b[0]:7.1f} "
            f"delta={delta:+7.1f}"
        )

    print("\nSUMMARY")

    for name, results in (("V2", v2_results), ("V5", v5_results)):
        s = summary(results)

        print(
            f"{name} "
            f"mean={s['mean']:7.1f} "
            f"median={s['median']:7.1f} "
            f"q25={s['q25']:7.1f} "
            f"min={s['min']:7.1f} "
            f"max={s['max']:7.1f} "
            f"meanScore={s['score']:8.2f} "
            f"<100={s['under100']}/40 "
            f"<200={s['under200']}/40 "
            f">=750={s['over750']}/40 "
            f">=1000={s['over1000']}/40 "
            f"caps={s['caps']}/40"
        )

    wins = sum(d > 0.05 for d in deltas)
    losses = sum(d < -0.05 for d in deltas)
    ties = len(deltas) - wins - losses

    print(
        "\nV5 vs V2: "
        f"wins={wins} losses={losses} ties={ties} "
        f"mean_delta={statistics.mean(deltas):+.1f} "
        f"median_delta={statistics.median(deltas):+.1f}"
    )
