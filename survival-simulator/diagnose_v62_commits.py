import random
import contextlib
import io

from src.core import SimulationCore
from src.utils.controllers.evohive_v62_debug import action_decision

SEEDS = [5521, 10061, 21283, 24509, 24821, 27077, 32321]

for seed in SEEDS:
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []
    events = []

    while True:
        state = sim.step(actions)

        if state["num_agents"] == 0 or state["sim_time"] >= 3000:
            break

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            buf = io.StringIO()

            with contextlib.redirect_stdout(buf):
                action = action_decision(obs, rng)

            for line in buf.getvalue().splitlines():
                if line.startswith("FRUITCOMMIT"):
                    events.append((float(state["sim_time"]), line))

            actions.append((agent.agent_id, action))

    print()
    print(
        f"=== SEED {seed} "
        f"END={state['sim_time']:.1f} "
        f"SCORE={sim.env.score:.2f} "
        f"COMMITS={len(events)} ==="
    )

    for t, event in events[:100]:
        print(f"t={t:.1f} {event}")

    if len(events) > 100:
        print(f"... {len(events)-100} additional commits")
