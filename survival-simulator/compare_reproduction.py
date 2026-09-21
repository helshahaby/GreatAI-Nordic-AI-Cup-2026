import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2
from src.utils.controllers.evohive_r65 import action_decision as r65
from src.utils.controllers.evohive_r75 import action_decision as r75
from src.utils.controllers.evohive_r65slow import action_decision as r65slow

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

    max_agents = 5

    while True:
        state = sim.step(actions)
        max_agents = max(max_agents, int(state["num_agents"]))

        if state["num_agents"] == 0 or state["sim_time"] >= 3000:
            return (
                float(state["sim_time"]),
                float(sim.env.score),
                max_agents,
            )

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            action = policy(obs, rng)
            actions.append((agent.agent_id, action))


def run_seed(seed):
    return (
        seed,
        run(seed, v2),
        run(seed, r65),
        run(seed, r75),
        run(seed, r65slow),
    )


def summarize(results):
    times = [x[0] for x in results]
    scores = [x[1] for x in results]
    max_agents = [x[2] for x in results]
    q = statistics.quantiles(times, n=4)

    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "q25": q[0],
        "min": min(times),
        "max": max(times),
        "score": statistics.mean(scores),
        "maxA": statistics.mean(max_agents),
        "u100": sum(t < 100 for t in times),
        "u200": sum(t < 200 for t in times),
        "o750": sum(t >= 750 for t in times),
        "o1000": sum(t >= 1000 for t in times),
        "caps": sum(t >= 3000 for t in times),
    }


if __name__ == "__main__":
    rows = []

    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(run_seed, seed) for seed in SEEDS]

        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda x: x[0])

    names = ["V2", "R65", "R75", "R65S"]
    all_results = {name: [] for name in names}

    for seed, *results in rows:
        for name, result in zip(names, results):
            all_results[name].append(result)

        values = " ".join(
            f"{name}={result[0]:7.1f}"
            for name, result in zip(names, results)
        )

        print(f"{seed:5d} {values}")

    print("\nSUMMARY")

    for name in names:
        s = summarize(all_results[name])

        print(
            f"{name:4s} "
            f"mean={s['mean']:7.1f} "
            f"median={s['median']:7.1f} "
            f"q25={s['q25']:7.1f} "
            f"min={s['min']:7.1f} "
            f"max={s['max']:7.1f} "
            f"score={s['score']:8.2f} "
            f"avgMaxA={s['maxA']:5.1f} "
            f"<100={s['u100']:2d} "
            f"<200={s['u200']:2d} "
            f">=750={s['o750']:2d} "
            f">=1000={s['o1000']:2d} "
            f"caps={s['caps']:2d}"
        )

    baseline = all_results["V2"]

    print("\nPAIRED VS V2")

    for name in names[1:]:
        candidate = all_results[name]

        deltas = [
            b[0] - a[0]
            for a, b in zip(baseline, candidate)
        ]

        score_deltas = [
            b[1] - a[1]
            for a, b in zip(baseline, candidate)
        ]

        wins = sum(d > 0.05 for d in deltas)
        losses = sum(d < -0.05 for d in deltas)
        ties = len(deltas) - wins - losses

        print(
            f"{name:4s} "
            f"wins={wins:2d} "
            f"losses={losses:2d} "
            f"ties={ties:2d} "
            f"meanDT={statistics.mean(deltas):+7.1f} "
            f"medianDT={statistics.median(deltas):+7.1f} "
            f"meanDScore={statistics.mean(score_deltas):+8.2f}"
        )
