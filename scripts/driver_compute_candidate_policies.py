import time
import pickle
import sys

from multiprocessing import Pool

import numpy as np

from constrained_bai.driver import (
    get_driver_target_velocity_only_reward,
    get_driver_target_location_only_reward,
    get_driver_constraint_target_velocity,
    get_driver_constraint_target_location,
    get_driver_target_velocity,
    get_driver_target_location,
)


def get_policy(arg):
    i, envs = arg
    idx = np.random.randint(0, len(envs))
    print("Policy", i, "idx", idx)
    env = envs[idx]

    theta, phi, threshold = env.reward_w, env.constraint_w, env.threshold
    theta += np.random.normal(size=theta.shape) / 10
    if phi is not None:
        phi += np.random.normal(size=phi.shape) / 10
        threshold += np.random.normal() / 10
    policy, r_features, c_features = env.get_optimal_policy(
        phi=phi, threshold=threshold, theta=theta, verbose=True
    )

    return policy


def main():
    """Simple script to precompute a set of candidate policies.

    Randomly samples reward functions and optimizes for them to
    obtrain the policies. The script takes two command line arguments:
        - Number of policies to compute
        - Number of parallel processes to use
    The resulting policies are always saved in "driver_candidate_policies.pkl".
    """
    n_policies = int(sys.argv[1])
    n_processes = int(sys.argv[2])

    my_envs = [
        get_driver_constraint_target_velocity(),
        get_driver_constraint_target_location(),
        get_driver_constraint_target_velocity(blocking_cars=True),
        get_driver_target_velocity(),
        get_driver_target_location(),
        get_driver_target_velocity(blocking_cars=True),
    ]

    if n_processes > 1:
        pool = Pool(n_processes)
        args = [(i, my_envs) for i in range(n_policies)]
        policies = pool.map(get_policy, args)
    else:
        policies = []
        for i in range(n_policies):
            policies.append(get_policy((i, my_envs)))

    filename = "driver_candidate_policies.pkl"
    with open(filename, "wb") as f:
        pickle.dump(policies, f)


if __name__ == "__main__":
    main()
