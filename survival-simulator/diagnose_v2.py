import random
from collections import defaultdict

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision

SEEDS = [887, 1699, 2081, 5297, 8101, 10613, 7331, 11213]

def run(seed):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    stats = defaultdict(int)
    last_ids = set()
    max_population = 0
    snapshots = []
    last_snapshot = -100

    while True:
        state = sim.step(actions)

        current_agents = list(sim.env.agents)
        current_ids = {a.agent_id for a in current_agents}

        born = current_ids - last_ids if last_ids else set()
        if last_ids:
            stats["births"] += len(born)

        disappeared = last_ids - current_ids
        stats["disappeared"] += len(disappeared)

        last_ids = current_ids
        max_population = max(max_population, len(current_agents))

        t = state["sim_time"]

        if t - last_snapshot >= 100:
            last_snapshot = t

            if current_agents:
                avg_energy = sum(a.energy for a in current_agents) / len(current_agents)
                avg_age = sum(a.age for a in current_agents) / len(current_agents)
                avg_speed = sum(a.speed for a in current_agents) / len(current_agents)
                avg_sprint = sum(a.sprint_speed for a in current_agents) / len(current_agents)
                avg_maxe = sum(a.max_energy for a in current_agents) / len(current_agents)
            else:
                avg_energy = avg_age = avg_speed = avg_sprint = avg_maxe = 0

            snapshots.append((
                t,
                len(current_agents),
                len(sim.env.predators),
                len(sim.env.fruits),
                avg_energy,
                avg_age,
                avg_speed,
                avg_sprint,
                avg_maxe,
            ))

        if state["num_agents"] == 0 or t >= 3000:
            print("\n" + "=" * 90)
            print(
                f"SEED {seed} FINAL "
                f"time={t:.1f} score={state['score']:.3f} "
                f"max_population={max_population} "
                f"births={stats['births']} "
                f"disappeared={stats['disappeared']}"
            )

            print(
                "time     pop pred fruit "
                "avgEnergy avgAge avgSpeed avgSprint avgMaxEnergy"
            )

            for x in snapshots:
                print(
                    f"{x[0]:7.1f} "
                    f"{x[1]:3d} "
                    f"{x[2]:4d} "
                    f"{x[3]:5d} "
                    f"{x[4]:9.1f} "
                    f"{x[5]:6.1f} "
                    f"{x[6]:8.2f} "
                    f"{x[7]:9.2f} "
                    f"{x[8]:12.1f}"
                )

            return

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            if obs is None:
                continue

            action = action_decision(obs, rng)
            actions.append((agent.agent_id, action))

for seed in SEEDS:
    run(seed)
