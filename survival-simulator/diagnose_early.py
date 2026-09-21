import random
from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2
from src.utils.controllers.evohive_v41_adaptive import action_decision as v41

SEEDS = [8597, 5849, 5171, 2473]


def run(seed, name, policy):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    next_report = 0.0

    print(f"\n===== {name} seed={seed} =====")

    while True:
        state = sim.step(actions)
        t = float(state["sim_time"])

        if t + 1e-6 >= next_report:
            agents = list(sim.env.agents)

            energies = [float(a.energy) for a in agents]
            ages = [float(a.age) for a in agents]

            fruit_visible = []
            tree_visible = []
            nearest_fruit = []
            nearest_tree = []

            for obs in state["observations"]:
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

                fruit_visible.append(len(fruits))
                tree_visible.append(len(trees))

                if fruits:
                    nearest_fruit.append(min(fruits))

                if trees:
                    nearest_tree.append(min(trees))

            avg_e = sum(energies) / len(energies) if energies else 0.0
            min_e = min(energies) if energies else 0.0

            avg_fv = (
                sum(fruit_visible) / len(fruit_visible)
                if fruit_visible else 0.0
            )

            avg_tv = (
                sum(tree_visible) / len(tree_visible)
                if tree_visible else 0.0
            )

            nf = (
                sum(nearest_fruit) / len(nearest_fruit)
                if nearest_fruit else -1.0
            )

            nt = (
                sum(nearest_tree) / len(nearest_tree)
                if nearest_tree else -1.0
            )

            print(
                f"t={t:6.1f} "
                f"pop={len(agents):2d} "
                f"avgE={avg_e:7.1f} "
                f"minE={min_e:7.1f} "
                f"fruitSeen={avg_fv:5.1f} "
                f"treeSeen={avg_tv:5.1f} "
                f"nearFruit={nf:7.1f} "
                f"nearTree={nt:7.1f} "
                f"worldFruit={len(sim.env.fruits):3d} "
                f"pred={len(sim.env.predators):2d}"
            )

            next_report += 10.0

        if state["num_agents"] == 0 or t >= 120.0:
            print(
                f"END t={t:.1f} "
                f"score={sim.env.score:.3f} "
                f"agents={state['num_agents']}"
            )
            return

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            action = policy(obs, rng)
            actions.append((agent.agent_id, action))


for seed in SEEDS:
    run(seed, "V2", v2)
    run(seed, "V41", v41)
