import math
import random
import numpy as np

from src.utils.DTOs import ActionRequest


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def closest(observations, object_type):
    objects = [o for o in observations if o.get("type") == object_type]
    if not objects:
        return None
    return min(objects, key=lambda o: o.get("distance", float("inf")))


def edge_danger(observations):
    """
    Estimate whether a visible obstacle edge is dangerously close.
    Edge coordinates are already agent-relative.
    """
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

        if denom <= 1e-9:
            px, py = x1, y1
        else:
            t = -(x1 * dx + y1 * dy) / denom
            t = max(0.0, min(1.0, t))
            px = x1 + t * dx
            py = y1 + t * dy

        distance = math.hypot(px, py)

        if distance < best_distance:
            best_distance = distance
            best_angle = math.atan2(py, px)

    return best_distance, best_angle


def action_decision(observation_response: dict, rng: random.Random):
    agent_id = observation_response["agent_id"]
    observations = observation_response["observations"]

    energy = float(observation_response["energy"])
    age = float(observation_response["age"])

    speed = float(observation_response["speed"])
    sprint_speed = float(observation_response["sprint_speed"])
    max_energy = float(observation_response["max_energy"])

    energy_ratio = energy / max(max_energy, 1.0)

    predator = closest(observations, "Predator")
    fruit = closest(observations, "Fruit")
    tree = closest(observations, "Tree")

    move_distance = 0.0
    move_direction = 0.0
    turn_angle = 0.0
    spawn_agent = False

    # ---------------------------------------------------------
    # ROLE 1: SURVIVOR
    # Predator avoidance always has highest priority.
    # ---------------------------------------------------------
    if predator is not None:
        predator_distance = float(predator["distance"])
        predator_angle = float(predator["angle"])

        escape_angle = wrap_angle(predator_angle + math.pi)

        if predator_distance < 105.0:
            move_distance = sprint_speed
        elif predator_distance < 135.0:
            move_distance = min(
                sprint_speed,
                speed + 0.65 * (sprint_speed - speed)
            )
        else:
            move_distance = speed

        move_direction = escape_angle

        # Retreat while facing the predator.
        turn_angle = wrap_angle(predator_angle)

        return ActionRequest(
            agent_id=agent_id,
            move_distance=float(max(0.0, min(move_distance, sprint_speed))),
            move_direction=float(wrap_angle(move_direction)),
            turn_angle=float(wrap_angle(turn_angle)),
            spawn_agent=False,
        )

    # ---------------------------------------------------------
    # ROLE 2: FORAGER
    # Food is the next priority.
    # ---------------------------------------------------------
    if fruit is not None:
        fruit_distance = float(fruit["distance"])
        fruit_angle = float(fruit["angle"])

        move_direction = fruit_angle

        # Don't waste sprint energy reaching food unless energy
        # is becoming dangerous.
        if energy_ratio < 0.12 and fruit_distance > speed:
            move_distance = min(sprint_speed, fruit_distance)
        else:
            move_distance = min(speed, fruit_distance)

        # Small turn only; movement itself can be directional.
        turn_angle = max(
            -0.12,
            min(0.12, fruit_angle * 0.10)
        )

    # ---------------------------------------------------------
    # ROLE 3: RESOURCE SCOUT
    #
    # If no fruit is currently visible, use trees as persistent
    # resource landmarks. Search a wider ring around each tree.
    # ---------------------------------------------------------
    elif tree is not None:
        tree_angle = float(tree["angle"])
        tree_distance = float(tree["distance"])

        orbit_side = 1.0 if agent_id % 2 == 0 else -1.0

        if tree_distance > 70.0:
            # Reach the resource region efficiently.
            move_direction = tree_angle
            move_distance = speed

            turn_angle = max(
                -0.12,
                min(0.12, tree_angle * 0.12)
            )

        elif tree_distance > 28.0:
            # Sweep diagonally around the tree rather than
            # repeatedly converging on its centre.
            move_direction = wrap_angle(
                tree_angle + orbit_side * 0.85
            )
            move_distance = speed * 0.90

            turn_angle = orbit_side * 0.08

        else:
            # Wider tangential sweep through the fruit-producing
            # region.
            move_direction = wrap_angle(
                tree_angle + orbit_side * math.pi / 2.0
            )
            move_distance = speed * 0.85

            turn_angle = orbit_side * 0.07

    # ---------------------------------------------------------
    # ROLE 4: DISPERSED EXPLORER
    #
    # No fruit and no tree visible. Spread agents through the
    # map with stable deterministic trajectories. Avoid frequent
    # large heading changes that can create local wandering.
    # ---------------------------------------------------------
    else:
        golden_angle = 2.399963229728653

        # Different deterministic search direction for each
        # lineage/agent.
        base_heading = wrap_angle(agent_id * golden_angle)

        # Change search lane slowly rather than every few seconds.
        lane = int(age / 45.0)

        lane_offset = (
            -0.55,
             0.00,
             0.55,
             1.10,
            -1.10,
        )[lane % 5]

        move_direction = wrap_angle(
            base_heading
            + lane_offset
            + 0.08 * math.sin(age * 0.05 + agent_id)
        )

        move_distance = speed

        # Slow sensor sweep. Turning is intentionally small to
        # reduce energy use and trajectory oscillation.
        scan_side = 1.0 if (agent_id + lane) % 2 == 0 else -1.0
        turn_angle = scan_side * 0.055

    # ---------------------------------------------------------
    # EDGE AVOIDANCE
    # ---------------------------------------------------------
    edge_distance, edge_angle = edge_danger(observations)

    if edge_angle is not None and edge_distance < 25.0:
        # Move away from the closest obstacle.
        move_direction = wrap_angle(edge_angle + math.pi)
        move_distance = min(speed, max(speed * 0.40, edge_distance))

        turn_angle = max(
            -0.15,
            min(0.15, move_direction * 0.10)
        )

    # ---------------------------------------------------------
    # ROLE 5: BREEDER
    #
    # Spawning costs 100 energy. Unlike the dummy controller,
    # never request reproduction continuously.
    #
    # V1 is intentionally conservative.
    # ---------------------------------------------------------
    reproduction_threshold = max(180.0, 0.55 * max_energy)

    if (
        energy > reproduction_threshold
        and age > 20.0
        and predator is None
    ):
        # Sparse deterministic reproduction windows.
        slot = int(age)

        if (slot + agent_id * 7) % 60 == 0:
            spawn_agent = True

    return ActionRequest(
        agent_id=agent_id,
        move_distance=float(max(0.0, min(move_distance, sprint_speed))),
        move_direction=float(wrap_angle(move_direction)),
        turn_angle=float(wrap_angle(turn_angle)),
        spawn_agent=spawn_agent,
    )
