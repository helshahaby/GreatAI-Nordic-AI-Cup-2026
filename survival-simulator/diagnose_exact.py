import inspect
import random

from src.core import SimulationCore
from src.utils.controllers.evohive_v11_frozen import action_decision as v11
from src.utils.controllers.evohive_v12_policy import action_decision as v12


SEEDS = [21, 42, 5555, 9999]


def run(seed, name, policy):
    sim = SimulationCore(seed=seed)
    env = sim.env
    rng = random.Random(seed)

    stats = {
        "births": 0,
        "fruit_eaten": 0,
        "fruit_rotted": 0,
        "energy_deaths": 0,
        "predator_deaths": 0,
        "spawn_requests": 0,
        "successful_spawns": 0,
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

        # In Environment.non_agent_step:
        # first kill_agent call = energy/natural death
        # second kill_agent call = predator collision.
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
        for agent_id, action in actions:
            if action.spawn_agent:
                stats["spawn_requests"] += 1

                agent = env.agents_dict.get(agent_id)

                if agent is not None and agent.energy > 100:
                    stats["successful_spawns"] += 1

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

            action = policy(agent_state, rng)
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
        f"{name:<3} "
        f"seed={seed:<5} "
        f"time={env.time:7.1f} "
        f"score={env.score:9.3f} "
        f"maxA={max_agents:<3} "
        f"births={stats['births']:<3} "
        f"spawnReq={stats['spawn_requests']:<3} "
        f"fruitEat={stats['fruit_eaten']:<4} "
        f"energyD={stats['energy_deaths']:<3} "
        f"predD={stats['predator_deaths']:<3} "
        f"deathAge={avg_death_age:6.1f} "
        f"deathE={avg_death_energy:7.1f} "
        f"predFinal={len(env.predators):<2} "
        f"fruitFinal={len(env.fruits):<4}"
    )


print(
    "Exact V1.1 vs V1.2 death/foraging diagnostic"
)

for seed in SEEDS:
    run(seed, "V11", v11)
    run(seed, "V12", v12)
