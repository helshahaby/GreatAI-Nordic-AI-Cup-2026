import random
import statistics

from src.core import SimulationCore
from src.utils.controllers.dummy_agent_policy import action_decision as dummy_policy
from src.utils.controllers.evohive_policy import action_decision as evohive_policy
from src.utils.controllers.evohive_nospawn_policy import action_decision as evohive_nospawn_policy
from src.utils.controllers.evohive_v2_policy import action_decision as evohive_v2_policy


SEEDS = [1, 42, 123, 1337, 2026]


def run_policy(policy, seed):
    sim = SimulationCore(seed=seed)
    action_rng = random.Random(seed)

    actions = []
    final_score = 0.0

    while True:
        state = sim.step(actions)
        final_score = float(state["score"])

        if state["num_agents"] == 0 or sim.env.time > 3000:
            break

        actions = []

        for agent, agent_state in zip(
            sim.env.agents,
            state["observations"]
        ):
            if agent_state is None:
                continue

            action = policy(agent_state, action_rng)
            actions.append((agent.agent_id, action))

    return final_score, float(sim.env.time)


def benchmark(name, policy):
    print(f"\n===== {name} =====")

    scores = []
    times = []

    for seed in SEEDS:
        score, survival_time = run_policy(policy, seed)

        scores.append(score)
        times.append(survival_time)

        print(
            f"seed={seed:<5} "
            f"score={score:10.3f} "
            f"survival={survival_time:8.1f}"
        )

    print(
        f"{name} MEAN | "
        f"score={statistics.mean(scores):.3f} | "
        f"survival={statistics.mean(times):.1f}"
    )


if __name__ == "__main__":
    benchmark("DUMMY", dummy_policy)
    benchmark("EVOHIVE_V1", evohive_policy)
    benchmark("EVOHIVE_NOSPAWN", evohive_nospawn_policy)
    benchmark("EVOHIVE_V2", evohive_v2_policy)
