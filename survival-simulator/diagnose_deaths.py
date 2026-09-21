import random
from collections import deque

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2

SEEDS = [5521, 8597, 853, 3253, 10061, 193, 2689, 8761]


def run(seed):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    history = {}
    previous_ids = set()

    print(f"\n===== SEED {seed} =====")

    while True:
        state = sim.step(actions)
        t = float(state["sim_time"])

        agents = list(sim.env.agents)
        obs_list = state["observations"]

        current_ids = {a.agent_id for a in agents}

        for dead_id in sorted(previous_ids - current_ids):
            print(f"\nDEATH id={dead_id} t={t:.1f}")

            for row in history.get(dead_id, []):
                print(
                    f"  t={row['t']:6.1f} "
                    f"E={row['energy']:7.1f} "
                    f"age={row['age']:6.1f} "
                    f"fruit={row['fruit']:2d} "
                    f"nearF={row['near_fruit']:7.1f} "
                    f"trees={row['trees']:2d} "
                    f"nearT={row['near_tree']:7.1f} "
                    f"pred={row['pred']:2d} "
                    f"nearP={row['near_pred']:7.1f}"
                )

        if int(t * 10 + 0.5) % 50 == 0:
            for agent, obs in zip(agents, obs_list):
                objects = obs["observations"]

                fruits = [
                    float(o["distance"])
                    for o in objects
                    if o.get("type") == "Fruit"
                ]

                trees = [
                    float(o["distance"])
                    for o in objects
                    if o.get("type") == "Tree"
                ]

                predators = [
                    float(o["distance"])
                    for o in objects
                    if o.get("type") == "Predator"
                ]

                row = {
                    "t": t,
                    "energy": float(agent.energy),
                    "age": float(agent.age),
                    "fruit": len(fruits),
                    "near_fruit": min(fruits) if fruits else -1.0,
                    "trees": len(trees),
                    "near_tree": min(trees) if trees else -1.0,
                    "pred": len(predators),
                    "near_pred": min(predators) if predators else -1.0,
                }

                history.setdefault(
                    agent.agent_id,
                    deque(maxlen=5)
                ).append(row)

        previous_ids = current_ids

        if state["num_agents"] == 0 or t >= 150.0:
            print(
                f"\nEND seed={seed} "
                f"t={t:.1f} "
                f"pop={state['num_agents']} "
                f"score={sim.env.score:.3f}"
            )
            return

        actions = []

        for agent, obs in zip(agents, obs_list):
            action = v2(obs, rng)
            actions.append((agent.agent_id, action))


for seed in SEEDS:
    run(seed)
