import random

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2
from src.utils.controllers.evohive_v5_scan import action_decision as v5

SEEDS = [853, 3253, 8861, 8597]


def run(seed, policy):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []
    max_agents = 5

    while True:
        state = sim.step(actions)
        t = float(state["sim_time"])
        max_agents = max(max_agents, int(state["num_agents"]))

        if state["num_agents"] == 0 or t >= 150.0:
            energies = [float(a.energy) for a in sim.env.agents]

            return (
                t,
                int(state["num_agents"]),
                max_agents,
                sum(energies),
                float(sim.env.score),
            )

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            action = policy(obs, rng)
            actions.append((agent.agent_id, action))


for seed in SEEDS:
    a = run(seed, v2)
    b = run(seed, v5)

    print(
        f"seed={seed:5d} "
        f"V2: t={a[0]:6.1f} pop={a[1]:2d} maxA={a[2]:2d} "
        f"E={a[3]:8.1f} score={a[4]:8.2f} | "
        f"V5: t={b[0]:6.1f} pop={b[1]:2d} maxA={b[2]:2d} "
        f"E={b[3]:8.1f} score={b[4]:8.2f}"
    )
