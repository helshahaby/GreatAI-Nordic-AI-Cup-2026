import inspect
import random

from src.core import SimulationCore
from src.utils.controllers.evohive_v41_adaptive import action_decision as v2
from src.utils.controllers.evohive_opt_predator import action_decision as opt


SEEDS = [853, 3253, 4787, 9371]

OPT24 = {
    "sprint_distance": 115.88017318499261,
    "partial_distance": 140.5951985780688,
    "partial_fraction": 0.19417187305192546,
    "face_distance": 86.29180972751521,
    "close_turn": 0.3033552639554437,
}


def run(seed, name, policy, params=None):
    sim = SimulationCore(seed=seed)
    env = sim.env
    rng = random.Random(seed)

    stats = {
        "births": 0,
        "fruit_eaten": 0,
        "fruit_rotted": 0,
        "energy_deaths": 0,
        "predator_deaths": 0,
        "predator_seen_ticks": 0,
        "close_predator_ticks": 0,
        "multi_predator_ticks": 0,
        "low_energy_threat_ticks": 0,
    }

    death_ages = []
    death_energies = []
    max_agents = len(env.agents)

    original_spawn = env.spawn_agent
    original_remove_fruit = env.remove_fruit
    original_kill = env.kill_agent

    def tracked_spawn(*args, **kwargs):
        parent = kwargs.get("parent")
        result = original_spawn(*args, **kwargs)

        if parent is not None:
            stats["births"] += 1

        return result

    def tracked_remove_fruit(fruit):
        eaten = False

        for agent in list(env.agents):
            dx = agent.x - fruit.x
            dy = agent.y - fruit.y

            if (dx * dx + dy * dy) ** 0.5 < agent.size + fruit.radius:
                eaten = True
                break

        if eaten:
            stats["fruit_eaten"] += 1
        else:
            stats["fruit_rotted"] += 1

        return original_remove_fruit(fruit)

    def tracked_kill(agent):
        caller = inspect.currentframe().f_back
        line = caller.f_lineno

        death_ages.append(float(agent.age))
        death_energies.append(float(agent.energy))

        if line < 700:
            stats["energy_deaths"] += 1
        else:
            stats["predator_deaths"] += 1

        return original_kill(agent)

    env.spawn_agent = tracked_spawn
    env.remove_fruit = tracked_remove_fruit
    env.kill_agent = tracked_kill

    actions = []

    while True:
        state = sim.step(actions)

        max_agents = max(max_agents, len(env.agents))

        if state["num_agents"] == 0 or env.time >= 3000:
            break

        actions = []

        for agent, agent_state in zip(
            env.agents,
            state["observations"]
        ):
            if agent_state is None:
                continue

            predators = [
                o for o in agent_state["observations"]
                if o.get("type") == "Predator"
            ]

            if predators:
                stats["predator_seen_ticks"] += 1

                nearest = min(
                    float(p["distance"])
                    for p in predators
                )

                if nearest < 120.0:
                    stats["close_predator_ticks"] += 1

                if len(predators) >= 2:
                    stats["multi_predator_ticks"] += 1

                ratio = (
                    float(agent_state["energy"])
                    / max(float(agent_state["max_energy"]), 1.0)
                )

                if ratio < 0.20:
                    stats["low_energy_threat_ticks"] += 1

            if params is None:
                action = policy(agent_state, rng)
            else:
                action = policy(agent_state, rng, params)

            actions.append((agent.agent_id, action))

    avg_death_age = (
        sum(death_ages) / len(death_ages)
        if death_ages else 0.0
    )

    avg_death_energy = (
        sum(death_energies) / len(death_energies)
        if death_energies else 0.0
    )

    print(
        f"{name:<5} "
        f"seed={seed:<5} "
        f"time={env.time:7.1f} "
        f"score={env.score:9.3f} "
        f"maxA={max_agents:<3} "
        f"births={stats['births']:<3} "
        f"fruit={stats['fruit_eaten']:<4} "
        f"energyD={stats['energy_deaths']:<3} "
        f"predD={stats['predator_deaths']:<3} "
        f"deathAge={avg_death_age:6.1f} "
        f"deathE={avg_death_energy:7.1f} "
        f"seen={stats['predator_seen_ticks']:<6} "
        f"close={stats['close_predator_ticks']:<6} "
        f"multi={stats['multi_predator_ticks']:<6} "
        f"lowThreat={stats['low_energy_threat_ticks']:<6} "
        f"predFinal={len(env.predators):<2} "
        f"fruitFinal={len(env.fruits):<4}"
    )


print("V2-KITE vs OPT24 predator diagnostic\n")

for seed in SEEDS:
    run(seed, "V2", v2)
    run(seed, "OPT24", opt, OPT24)
