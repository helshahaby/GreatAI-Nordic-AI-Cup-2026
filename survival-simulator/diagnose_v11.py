import random
import sys

from src.core import SimulationCore
from src.utils.controllers.evohive_v11_policy import action_decision


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1

sim = SimulationCore(seed=seed)
rng = random.Random(seed)
actions = []

last_agents = None
last_predators = None
max_agents = 0
last_report = -10.0

print(f"===== EVOHIVE V1 DIAGNOSTIC seed={seed} =====")

while True:
    state = sim.step(actions)

    t = float(sim.env.time)
    n_agents = len(sim.env.agents)
    n_predators = len(sim.env.predators)
    n_fruits = len(sim.env.fruits)

    max_agents = max(max_agents, n_agents)

    energies = [a.energy for a in sim.env.agents]

    if energies:
        avg_energy = sum(energies) / len(energies)
        min_energy = min(energies)
        max_energy = max(energies)
    else:
        avg_energy = min_energy = max_energy = 0.0

    changed = (
        last_agents is None
        or n_agents != last_agents
        or n_predators != last_predators
    )

    periodic = t - last_report >= 10.0

    if changed or periodic:
        print(
            f"t={t:7.1f} "
            f"score={state['score']:9.2f} "
            f"agents={n_agents:3d} "
            f"pred={n_predators:3d} "
            f"fruit={n_fruits:4d} "
            f"Eavg={avg_energy:7.1f} "
            f"Emin={min_energy:7.1f} "
            f"Emax={max_energy:7.1f}"
        )

        last_report = t

    last_agents = n_agents
    last_predators = n_predators

    if n_agents == 0 or t >= 3000:
        print()
        print("===== FINAL =====")
        print(f"seed={seed}")
        print(f"time={t:.1f}")
        print(f"score={state['score']:.3f}")
        print(f"max_agents={max_agents}")
        print(f"predators={n_predators}")
        print(f"fruits={n_fruits}")
        break

    actions = []

    for agent, agent_state in zip(
        sim.env.agents,
        state["observations"]
    ):
        if agent_state is None:
            continue

        action = action_decision(agent_state, rng)
        actions.append((agent.agent_id, action))
