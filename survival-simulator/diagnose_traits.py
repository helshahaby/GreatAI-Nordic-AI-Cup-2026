import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.evohive_v2_kite import action_decision as v2

SEEDS = [193, 5521, 10061, 2689, 3761, 8101]


def mean(values):
    return statistics.mean(values) if values else 0.0


def run(seed):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []
    next_report = 0.0
    max_pop = 5

    print(f"\n===== seed={seed} =====")

    while True:
        state = sim.step(actions)
        t = float(state["sim_time"])

        agents = list(sim.env.agents)
        max_pop = max(max_pop, len(agents))

        if agents and t + 1e-6 >= next_report:
            print(
                f"t={t:6.1f} "
                f"pop={len(agents):2d} "
                f"E={mean([float(a.energy) for a in agents]):7.1f} "
                f"speed={mean([float(a.speed) for a in agents]):6.2f} "
                f"sprint={mean([float(a.sprint_speed) for a in agents]):6.2f} "
                f"hear={mean([float(a.hearing_radius) for a in agents]):7.1f} "
                f"vision={mean([float(a.vision_radius) for a in agents]):7.1f} "
                f"cone={mean([float(a.cone_angle) for a in agents]):6.3f} "
                f"maxE={mean([float(a.max_energy) for a in agents]):7.1f} "
                f"pred={len(sim.env.predators):2d}"
            )

            next_report += 100.0

        if state["num_agents"] == 0 or t >= 1200.0:
            print(
                f"END seed={seed} "
                f"time={t:.1f} "
                f"score={sim.env.score:.3f} "
                f"maxPop={max_pop}"
            )
            return

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            action = v2(obs, rng)
            actions.append((agent.agent_id, action))


for seed in SEEDS:
    run(seed)
