import math
import random

from src.utils.DTOs import ActionRequest


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def objects_of_type(observations, kind):
    return [o for o in observations if o.get("type") == kind]


def nearest(observations, kind):
    items = objects_of_type(observations, kind)
    return min(items, key=lambda x: x.get("distance", float("inf"))) if items else None


def trait_quality(state):
    """
    Approximate evolutionary quality relative to the original agent traits.

    Baselines:
      speed        10
      sprint       20
      hearing      50
      vision       200
      vision angle pi/3
      max energy   500

    Survival-related mobility and perception receive higher weights.
    """
    speed = float(state["speed"])
    sprint = float(state["sprint_speed"])
    hearing = float(state["hearing_radius"])
    vision = float(state["vision_range"])
    angle = float(state["vision_angle"])
    max_energy = float(state["max_energy"])

    return (
        0.25 * (speed / 10.0) +
        0.30 * (sprint / 20.0) +
        0.15 * (hearing / 50.0) +
        0.15 * (vision / 200.0) +
        0.05 * (angle / (math.pi / 3.0)) +
        0.10 * (max_energy / 500.0)
    )


def closest_edge(observations):
    best_distance = float("inf")
    best_angle = None

    for obs in observations:
        if obs.get("type") != "Edge":
            continue

        coords = obs.get("coords")
        if not coords:
            continue

        (x1, y1), (x2, y2) = coords
        dx = x2 - x1
        dy = y2 - y1

        denom = dx * dx + dy * dy

        if denom > 1e-9:
            t = -(x1 * dx + y1 * dy) / denom
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy
        else:
            px, py = x1, y1

        d = math.hypot(px, py)

        if d < best_distance:
            best_distance = d
            best_angle = math.atan2(py, px)

    return best_distance, best_angle


def action_decision(state: dict, rng: random.Random):
    agent_id = int(state["agent_id"])
    obs = state["observations"]

    energy = float(state["energy"])
    max_energy = float(state["max_energy"])
    age = float(state["age"])

    speed = float(state["speed"])
    sprint = float(state["sprint_speed"])

    energy_ratio = energy / max(max_energy, 1.0)

    predators = sorted(
        objects_of_type(obs, "Predator"),
        key=lambda x: x["distance"]
    )

    fruit = nearest(obs, "Fruit")
    tree = nearest(obs, "Tree")

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn = False

    # =====================================================
    # 1. SURVIVAL
    # =====================================================
    if predators:
        predator = predators[0]

        d = float(predator["distance"])
        a = float(predator["angle"])

        # Direct radial escape.
        escape = wrap(a + math.pi)

        # If several predators are sensed, combine their repulsion.
        if len(predators) > 1:
            vx = 0.0
            vy = 0.0

            for p in predators:
                pd = max(float(p["distance"]), 1.0)
                pa = float(p["angle"])

                # Direction away from predator.
                away = pa + math.pi

                # Nearby predators have much greater influence.
                weight = 1.0 / (pd * pd)

                vx += math.cos(away) * weight
                vy += math.sin(away) * weight

            if abs(vx) + abs(vy) > 1e-12:
                escape = math.atan2(vy, vx)

        move_direction = wrap(escape)

        # Predator normal/sprint speeds are 11/15.
        #
        # Sprint aggressively while we still have enough energy.
        # At very low energy the simulator itself prevents sprinting.
        if d < 45:
            move_distance = sprint
        elif d < 90:
            move_distance = sprint
        elif d < 150:
            move_distance = min(
                sprint,
                speed + 0.60 * max(0.0, sprint - speed)
            )
        else:
            move_distance = speed

        # Turning costs energy and movement does not require facing
        # the movement direction. Only make a very small orientation
        # adjustment.
        turn_angle = max(-0.08, min(0.08, escape * 0.04))

        return ActionRequest(
            agent_id=agent_id,
            move_distance=float(move_distance),
            move_direction=float(move_direction),
            turn_angle=float(turn_angle),
            spawn_agent=False,
        )

    # =====================================================
    # 2. FORAGE
    # =====================================================
    if fruit is not None:
        d = float(fruit["distance"])
        a = float(fruit["angle"])

        move_direction = wrap(a)

        # Normally walk because walking is dramatically cheaper.
        move_distance = min(speed, d)

        # Emergency food acquisition.
        if energy_ratio < 0.20 and d > speed:
            move_distance = min(
                sprint,
                speed + 0.45 * max(0.0, sprint - speed)
            )

        # No need to face the fruit to reach it.
        turn_angle = 0.0

    # =====================================================
    # 3. FRUIT-PRODUCING TREE AREA
    # =====================================================
    elif tree is not None:
        d = float(tree["distance"])
        a = float(tree["angle"])

        # Don't collide with the tree itself.
        if d > 30:
            move_distance = min(speed * 0.80, max(0.0, d - 25))
            move_direction = wrap(a)
        else:
            # Orbit around tree while waiting/searching for fruit.
            move_distance = speed * 0.45
            move_direction = wrap(
                a + (math.pi / 2 if agent_id % 2 == 0 else -math.pi / 2)
            )

        turn_angle = 0.0

    # =====================================================
    # 4. EXPLORATION
    # =====================================================
    else:
        phase = (agent_id * 1.61803398875) % (2.0 * math.pi)

        move_distance = speed * 0.70

        # Different descendants disperse in different patterns.
        move_direction = (
            0.50 * math.sin(age * 0.035 + phase)
            + 0.20 * math.sin(age * 0.011 + phase * 2)
        )

        # Slowly scan the environment.
        turn_angle = 0.018 * math.sin(age * 0.025 + phase)

    # =====================================================
    # 5. OBSTACLE / BOUNDARY AVOIDANCE
    # =====================================================
    edge_distance, edge_angle = closest_edge(obs)

    if edge_angle is not None:
        # Only override when genuinely close.
        if edge_distance < 18:
            move_direction = wrap(edge_angle + math.pi)
            move_distance = speed * 0.75
            turn_angle = 0.0

        elif edge_distance < 35 and abs(edge_angle) < math.pi / 3:
            # Steer sideways instead of reversing.
            side = -1.0 if edge_angle > 0 else 1.0
            move_direction = wrap(
                move_direction + side * math.pi / 3
            )

    # =====================================================
    # 6. EVOLUTIONARY REPRODUCTION
    # =====================================================
    quality = trait_quality(state)

    # Children start with limited energy while spawning costs the
    # parent 100, so maintain a substantial reserve.
    #
    # Strong mutated individuals are permitted to reproduce earlier.
    if quality >= 1.12:
        threshold = max(165.0, 0.42 * max_energy)
        interval = 42
    elif quality >= 1.03:
        threshold = max(185.0, 0.48 * max_energy)
        interval = 52
    else:
        threshold = max(210.0, 0.55 * max_energy)
        interval = 68

    if (
        age > 18.0
        and energy > threshold
        and energy - 100.0 > 55.0
    ):
        slot = int(age)

        # Deterministic spreading of breeding events between agents.
        if (slot + agent_id * 11) % interval == 0:
            spawn = True

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, min(move_distance, sprint))),
        move_direction=float(wrap(move_direction)),
        turn_angle=float(wrap(turn_angle)),
        spawn_agent=spawn,
    )
