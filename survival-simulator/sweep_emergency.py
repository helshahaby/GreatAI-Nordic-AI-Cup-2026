import importlib
import random
import statistics

from src.core import SimulationCore

SEEDS = [
    1, 7, 13, 21, 42,
    77, 101, 123, 256, 512,
    777, 999, 1337, 1600, 2026,
    3001, 4096, 5555, 7777, 9999,
]

POLICIES = [
    ("V11", "src.utils.controllers.evohive_v11_frozen"),
    ("E08", "src.utils.controllers.evohive_emergency_08"),
    ("E10", "src.utils.controllers.evohive_emergency_10"),
    ("E12", "src.utils.controllers.evohive_emergency_12"),
    ("E14", "src.utils.controllers.evohive_emergency_14"),
    ("E16", "src.utils.controllers.evohive_emergency_16"),
    ("E18", "src.utils.controllers.evohive_emergency_18"),
    ("E20", "src.utils.controllers.evohive_emergency_20"),
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


all_results = {}

for name, module_name in POLICIES:
    module = importlib.import_module(module_name)
    policy = module.action_decision

    results = []

    print()
    print(name)

    for seed in SEEDS:
        score, survival = run(seed, policy)
        results.append((seed, score, survival))

        print(
            f"{seed:<5} "
            f"score={score:9.3f} "
            f"time={survival:7.1f}"
        )

    all_results[name] = results

print()
print("SUMMARY")

for name, _ in POLICIES:
    results = all_results[name]

    scores = [x[1] for x in results]
    times = [x[2] for x in results]

    q25 = statistics.quantiles(times, n=4)[0]

    print(
        f"{name:<4} "
        f"mean={statistics.mean(times):7.1f} "
        f"median={statistics.median(times):7.1f} "
        f"q25={q25:7.1f} "
        f"min={min(times):7.1f} "
        f"meanScore={statistics.mean(scores):8.2f} "
        f"caps={sum(x >= 3000 for x in times):2d}"
    )

print()
print("WINS VS V11")

baseline = {
    seed: survival
    for seed, _, survival in all_results["V11"]
}

for name, _ in POLICIES[1:]:
    wins = 0
    losses = 0

    for seed, _, survival in all_results[name]:
        if survival > baseline[seed] + 0.05:
            wins += 1
        elif survival < baseline[seed] - 0.05:
            losses += 1

    print(
        f"{name:<4} wins={wins:2d} "
        f"losses={losses:2d}"
    )
