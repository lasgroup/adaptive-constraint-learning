import time
import pickle
import datetime

import matplotlib.pyplot as plt
import numpy as np

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from constrained_bai.driver import get_driver
from constrained_bai.driver_bandit import DriverBandit


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(RunObserver):
    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        custom_id = "{}".format(timestamp)
        return custom_id  # started_event returns the _run._id


ex = Experiment("driver_solve_with_restarts")
ex.observers = [SetID(), FileStorageObserver("results")]


@ex.config
def cfg():
    problem = {"cars_trajectory": "blocked", "goal": "target_velocity"}
    penalty_lambda = 1
    n_restarts = 10
    constraint = None
    experiment_label = None


@ex.automain
def run(_run, seed, problem, penalty_lambda, n_restarts, constraint, experiment_label):
    env = get_driver(
        problem["cars_trajectory"],
        problem["goal"],
        penalty_lambda=penalty_lambda,
        constraint=constraint,
    )
    env_eval = get_driver(
        problem["cars_trajectory"],
        problem["goal"],
        penalty_lambda=penalty_lambda,
        constraint="default",
    )

    max_rew = -float("inf")
    max_features = None

    for _ in range(n_restarts):
        policy, policy_features, _ = env.get_optimal_policy()
        reward = np.dot(policy_features, env.reward_w)
        if env.constraint_w is not None:
            constraint = np.dot(policy_features, env.constraint_w)
        else:
            constraint = None
        print("reward", reward, "constraint", constraint)
        if (constraint is None or constraint < env.threshold) and reward > max_rew:
            print("updating solution")
            max_rew = reward
            max_features = policy_features

    reward = np.dot(max_features, env_eval.reward_w)
    constraint = np.dot(max_features, env_eval.constraint_w)

    print("Found solution with")
    print("reward", reward)
    print("constraint", constraint)

    return {"reward": reward, "constraint": constraint}
