import math
import random

from src.core import SimulationCore
from src.utils.controllers.evohive_v6_fruitlock import action_decision as v2

SEED = 5521
TARGET = 3

sim = SimulationCore(seed=SEED)
rng = random.Random(SEED)
actions = []

print(
    "time,id,energy,x,y,heading,"
    "fruit_dist,fruit_angle,"
    "move_dist,move_dir,turn,"
    "actual_move,actual_world_angle"
)

while True:
    before = {}

    for a in sim.env.agents:
        before[a.agent_id] = (
            float(a.x),
            float(a.y),
            float(a.direction),
        )

    state = sim.step(actions)
    t = float(state["sim_time"])

    agents = list(sim.env.agents)
    obs_list = state["observations"]

    if TARGET not in {a.agent_id for a in agents}:
        print(f"DEAD target={TARGET} t={t:.1f}")
        break

    actions = []

    for agent, obs in zip(agents, obs_list):
        action = v2(obs, rng)
        actions.append((agent.agent_id, action))

        if agent.agent_id != TARGET:
            continue

        fruits = [
            o for o in obs["observations"]
            if o.get("type") == "Fruit"
        ]

        fruit = min(
            fruits,
            key=lambda o: float(o["distance"]),
            default=None,
        )

        old = before.get(TARGET)

        if old is None:
            actual_move = 0.0
            actual_angle = 0.0
        else:
            dx = float(agent.x) - old[0]
            dy = float(agent.y) - old[1]
            actual_move = math.hypot(dx, dy)

            if actual_move > 1e-9:
                actual_angle = math.atan2(dy, dx)
            else:
                actual_angle = float("nan")

        if fruit is None:
            fd = -1.0
            fa = float("nan")
        else:
            fd = float(fruit["distance"])
            fa = float(fruit["angle"])

        print(
            f"{t:.1f},{agent.agent_id},"
            f"{float(agent.energy):.2f},"
            f"{float(agent.x):.2f},"
            f"{float(agent.y):.2f},"
            f"{float(agent.direction):.4f},"
            f"{fd:.2f},{fa:.4f},"
            f"{float(action.move_distance):.2f},"
            f"{float(action.move_direction):.4f},"
            f"{float(action.turn_angle):.4f},"
            f"{actual_move:.2f},"
            f"{actual_angle:.4f}"
        )

    if state["num_agents"] == 0 or t >= 60.0:
        break
