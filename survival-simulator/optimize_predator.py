import json
import math
import random
import statistics
import time

import optuna

from src.core import SimulationCore
from src.utils.controllers.evohive_opt_predator import action_decision


DEV_SEEDS = [
    101, 307, 503, 709,
    907, 1103, 1301, 1601,
    1901, 2203, 2503, 2801,
]


def run(seed, params):
    sim = SimulationCore(seed=seed)
    rng = random.Random(seed)
    actions = []

    while True:
        state = sim.step(actions)

        t = float(state["sim_time"])

        if state["num_agents"] == 0 or t >= 3000.0:
            return t, float(state["score"])

        actions = []

        for agent, obs in zip(sim.env.agents, state["observations"]):
            if obs is None:
                continue

            action = action_decision(obs, rng, params)

            actions.append(
                (agent.agent_id, action)
            )


def q25(values):
    xs = sorted(values)

    if len(xs) == 1:
        return xs[0]

    pos = 0.25 * (len(xs) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return xs[lo]

    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def objective(trial):
    sprint_distance = trial.suggest_float(
        "sprint_distance", 70.0, 125.0
    )

    partial_distance = trial.suggest_float(
        "partial_distance",
        sprint_distance + 5.0,
        180.0,
    )

    params = {
        "sprint_distance": sprint_distance,
        "partial_distance": partial_distance,
        "partial_fraction": trial.suggest_float(
            "partial_fraction", 0.15, 0.95
        ),
        "face_distance": trial.suggest_float(
            "face_distance", 60.0, 130.0
        ),
        "close_turn": trial.suggest_float(
            "close_turn", 0.05, 0.60
        ),
    }

    survivals = []
    scores = []

    for i, seed in enumerate(DEV_SEEDS):
        survival, score = run(seed, params)

        survivals.append(survival)
        scores.append(score)

        # Early rejection of clearly catastrophic candidates.
        if i >= 3 and min(survivals) < 45.0:
            raise optuna.TrialPruned()

    mean_survival = statistics.mean(survivals)
    median_survival = statistics.median(survivals)
    lower_quartile = q25(survivals)
    minimum = min(survivals)
    mean_score = statistics.mean(scores)

    caps = sum(t >= 2999.9 for t in survivals)

    # Robustness-oriented objective.
    value = (
        0.25 * mean_survival
        + 0.30 * median_survival
        + 0.30 * lower_quartile
        + 0.15 * minimum
        + 0.02 * mean_score
        + caps * 1000.0
    )

    trial.set_user_attr("mean_survival", mean_survival)
    trial.set_user_attr("median_survival", median_survival)
    trial.set_user_attr("q25", lower_quartile)
    trial.set_user_attr("minimum", minimum)
    trial.set_user_attr("mean_score", mean_score)
    trial.set_user_attr("caps", caps)

    print(
        f"trial={trial.number:03d} "
        f"objective={value:8.2f} "
        f"mean={mean_survival:7.1f} "
        f"median={median_survival:7.1f} "
        f"q25={lower_quartile:7.1f} "
        f"min={minimum:7.1f} "
        f"caps={caps}"
    )

    return value


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(
        seed=20260921,
        n_startup_trials=10,
        multivariate=True,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="evohive_predator",
        storage="sqlite:///evohive_predator.db",
        load_if_exists=True,
    )

    start = time.time()

    study.optimize(
        objective,
        n_trials=30,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    best = study.best_trial

    print("\n" + "=" * 72)
    print("BEST TRIAL")
    print("=" * 72)
    print("trial:", best.number)
    print("objective:", best.value)

    print("\nPARAMETERS")
    print(json.dumps(best.params, indent=2))

    print("\nMETRICS")
    print(json.dumps(best.user_attrs, indent=2))

    print(
        f"\nelapsed={(time.time() - start) / 60.0:.1f} minutes"
    )

    with open("best_predator_params.json", "w") as f:
        json.dump(best.params, f, indent=2)
