import math
import random

from src.utils.DTOs import ActionRequest


DEFAULT_PARAMS = {
    "sprint_distance": 100.0,
    "partial_distance": 140.0,
    "partial_fraction": 0.55,
    "face_distance": 85.0,
    "close_turn": 0.30,
}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def nearest(obs, kind):
    xs = [x for x in obs if x.get("type") == kind]
    return min(xs, key=lambda x: x.get("distance", float("inf"))) if xs else None


def action_decision(state, rng, params=None):
    p = DEFAULT_PARAMS if params is None else params

    agent_id = int(state["agent_id"])
    observations = state["observations"]

    energy = float(state["energy"])
    age = float(state["age"])
    speed = float(state["speed"])
    sprint_speed = float(state["sprint_speed"])
    max_energy = float(state["max_energy"])

    energy_ratio = energy / max(max_energy, 1.0)

    predator = nearest(observations, "Predator")
    fruit = nearest(observations, "Fruit")
    tree = nearest(observations, "Tree")

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    if predator is not None:
        d = float(predator["distance"])
        a = float(predator["angle"])

        move_direction = wrap(a + math.pi)

        if d < p["sprint_distance"]:
            move_distance = sprint_speed
        elif d < p["partial_distance"]:
            move_distance = speed + p["partial_fraction"] * (
                sprint_speed - speed
            )
        else:
            move_distance = speed

        if d >= p["face_distance"]:
            turn_angle = a
        else:
            turn_angle = clamp(
                a,
                -p["close_turn"],
                p["close_turn"],
            )

        return ActionRequest(
            agent_id=agent_id,
            move_distance=float(move_distance),
            move_direction=float(move_direction),
            turn_angle=float(wrap(turn_angle)),
            spawn_agent=False,
        )

    if fruit is not None:
        d = float(fruit["distance"])
        a = float(fruit["angle"])

        move_direction = a

        if energy_ratio < 0.12:
            move_distance = min(sprint_speed, d)
        else:
            move_distance = min(speed, d)

        turn_angle = clamp(a * 0.10, -0.12, 0.12)

    elif tree is not None:
        d = float(tree["distance"])
        a = float(tree["angle"])

        if d > 45.0:
            move_direction = a
            move_distance = min(speed * 0.90, max(0.0, d - 30.0))
            turn_angle = clamp(a * 0.10, -0.16, 0.16)
        else:
            side = 1.0 if agent_id % 2 == 0 else -1.0
            move_direction = wrap(a + side * math.pi / 2.0)
            move_distance = speed * 0.70
            turn_angle = side * 0.10

    else:
        phase = agent_id * 2.399963229728653
        sector = int(age / 18.0)

        headings = (
            -1.2,
            0.0,
            1.2,
            2.4,
            -2.4,
            0.6,
            -0.6,
            math.pi,
        )

        move_direction = wrap(
            headings[(sector + agent_id) % len(headings)]
            + 0.10 * math.sin(age * 0.07 + phase)
        )

        move_distance = speed * 0.90

        turn_angle = (
            0.11
            if (sector + agent_id) % 2 == 0
            else -0.11
        )

    edge = None
    edge_distance = float("inf")
    edge_angle = None

    for o in observations:
        if o.get("type") != "Edge":
            continue

        coords = o.get("coords")
        if not coords:
            continue

        for x, y in coords:
            d = math.hypot(x, y)

            if d < edge_distance:
                edge_distance = d
                edge_angle = math.atan2(y, x)
                edge = o

    if edge is not None and edge_distance < 25.0:
        move_direction = wrap(edge_angle + math.pi)
        move_distance = speed
        turn_angle = clamp(move_direction * 0.10, -0.15, 0.15)

    threshold = max(180.0, 0.55 * max_energy)

    if (
        energy > threshold
        and age > 20.0
        and predator is None
        and (int(age) + agent_id * 7) % 60 == 0
    ):
        spawn_agent = True

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(clamp(move_distance, 0.0, sprint_speed)),
        move_direction=float(wrap(move_direction)),
        turn_angle=float(wrap(turn_angle)),
        spawn_agent=spawn_agent,
    )
