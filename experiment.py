import pickle
import datetime

import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.observers import FileStorageObserver, RunObserver

from constrained_bai.bandit import (
    get_toy_instance_1,
    get_toy_instance_2,
    get_toy_instance_3,
    get_toy_instance_4,
    get_toy_instance_5,
    get_random_instance_1,
    get_unit_sphere_1,
)
from constrained_bai.algorithms.oracle_static import CLBOracleStatic
from constrained_bai.algorithms.g_allocation_static import GAllocationStatic
from constrained_bai.algorithms.adaptive import CLBAdaptiveAlgorithm
from constrained_bai.algorithms.adaptive_greedy import CLBAdaptiveGreedyAlgorithm
from constrained_bai.algorithms.uniform_static import UniformSamplingStatic

from constrained_bai.driver import get_driver
from constrained_bai.driver_bandit import DriverBandit
from constrained_bai.noise_models import get_normal, linear_binary


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


ex = Experiment("constained_bai_experiment")
ex.observers = [SetID(), FileStorageObserver("results")]


@ex.config
def cfg():
    # method = "adaptive_greedy_reward_feasible_tuned"
    # method = "adaptive_greedy_reward_uncertain_tuned"
    method = "adaptive_tuned"
    sqbeta_tuned = 0.5
    # experiment = {"instance": "toy_1", "epsilon": 0.1}
    # experiment = {"instance": "toy_3", "epsilon": 0.05, "dimension":10}
    # experiment = {"instance": "toy_3", "epsilon": 0.05, "dimension":3}
    # experiment = {"instance": "toy_4", "epsilon": 0.05, "dimension":5, "n_arms": 10}
    experiment = {"instance": "toy_5"}
    # experiment = {"instance": "unit_sphere", "n_arms": 30, "dimension": 30}
    # experiment = {
    #     "instance": "driver",
    #     "cars_trajectory": "changing_lane",
    #     "goal": "target_velocity",
    #     "policy_file": "driver_candidate_policies_1000.pkl",
    #     "noise_sigma": None,
    #     "binary_feedback": True,
    # }
    debug = False
    experiment_label = None


@ex.automain
def run(_run, seed, method, experiment, sqbeta_tuned, debug, experiment_label):
    # epsilons = (0.1, 0.2)
    experiment_instance = experiment["instance"]

    print(f"Experiment: {experiment}")

    if experiment_instance == "toy_1":
        bandit = get_toy_instance_1(experiment["epsilon"])
    elif experiment_instance == "toy_2":
        bandit = get_toy_instance_2(experiment["epsilon"])
    elif experiment_instance == "toy_3":
        bandit = get_toy_instance_3(experiment["epsilon"], d=experiment["dimension"])
    elif experiment_instance == "toy_4":
        bandit = get_toy_instance_4(
            experiment["epsilon"],
            d=experiment["dimension"],
            n_arms=experiment["n_arms"],
        )
    elif experiment_instance == "toy_5":
        bandit = get_toy_instance_5()
    elif experiment_instance == "unit_sphere":
        bandit = get_unit_sphere_1(experiment["dimension"], experiment["n_arms"])
    elif experiment_instance == "driver":
        env = get_driver(
            experiment["cars_trajectory"],
            experiment["goal"],
            penalty_lambda=0,
            constraint="default",
        )
        with open(experiment["policy_file"], "rb") as f:
            policies = pickle.load(f)
        if experiment["binary_feedback"]:
            noise_model = linear_binary
        else:
            noise_model = get_normal(experiment["noise_sigma"])
        bandit = DriverBandit(
            env,
            policies,
            noise_model,
            normalize_constraints=experiment["binary_feedback"],
        )
    else:
        raise NotImplementedError(f"unkown experiment {experiment_instance}")

    if method == "oracle":
        algorithm = CLBOracleStatic(delta=0.05, debug=debug)
    elif method == "g-allocation":
        algorithm = GAllocationStatic(delta=0.05, debug=debug)
    elif method == "uniform":
        algorithm = UniformSamplingStatic(delta=0.05, debug=debug)
    elif method == "adaptive_static":
        algorithm = CLBAdaptiveAlgorithm(
            delta=0.05, epsilon=0.01, phase_base=2, debug=debug
        )
    elif method == "adaptive_maxvar_all":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05, tuned_ci=False, selection="maxvar_all", debug=debug
        )
    elif method == "adaptive_uniform_all":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05, tuned_ci=False, selection="uniform_all", debug=debug
        )
    elif method == "adaptive":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05, tuned_ci=False, selection="adaptive", debug=debug
        )
    elif method == "adaptive_uniform":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05, tuned_ci=False, selection="uniform", debug=debug
        )
    elif method == "adaptive_maxvar_all_tuned":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05,
            tuned_ci=True,
            selection="maxvar_all",
            sqbeta_tuned=sqbeta_tuned,
            debug=debug,
        )
    elif method == "adaptive_uniform_all_tuned":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05,
            tuned_ci=True,
            selection="uniform_all",
            sqbeta_tuned=sqbeta_tuned,
            debug=debug,
        )
    elif method == "adaptive_tuned":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05,
            tuned_ci=True,
            selection="adaptive",
            sqbeta_tuned=sqbeta_tuned,
            debug=debug,
        )
    elif method == "adaptive_uniform_tuned":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05,
            tuned_ci=True,
            selection="uniform",
            sqbeta_tuned=sqbeta_tuned,
            debug=debug,
        )
    elif method == "adaptive_greedy_reward_uncertain":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05,
            tuned_ci=False,
            selection="greedy_reward_uncertain",
            debug=debug,
        )
    elif method == "adaptive_greedy_reward_uncertain_tuned":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05,
            tuned_ci=True,
            selection="greedy_reward_uncertain",
            sqbeta_tuned=sqbeta_tuned,
            debug=debug,
        )
    elif method == "adaptive_greedy_reward_feasible":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05,
            tuned_ci=False,
            selection="greedy_reward_feasible",
            debug=debug,
        )
    elif method == "adaptive_greedy_reward_feasible_tuned":
        algorithm = CLBAdaptiveGreedyAlgorithm(
            delta=0.05,
            tuned_ci=True,
            selection="greedy_reward_feasible",
            sqbeta_tuned=sqbeta_tuned,
            debug=debug,
        )
    else:
        raise NotImplementedError(f"unkown method {method}")

    n_iter, best_arm_i = algorithm.run(bandit)
    constraint_violation = bandit.get_constraint(best_arm_i) > bandit.threshold
    correct = best_arm_i == bandit.best_arm_i
    print(
        f"Finished after {n_iter} iterations,  constraint violation: {constraint_violation},  returned correct arm: {correct}"
    )
    return {
        "iterations": n_iter,
        "constraint_violation": constraint_violation,
        "correct": correct,
    }
