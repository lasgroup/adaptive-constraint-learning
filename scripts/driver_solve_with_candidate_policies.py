import time
import pickle

import matplotlib.pyplot as plt
import numpy as np

from constrained_bai.driver import (
    get_driver_target_velocity,
    get_driver_target_velocity_only_reward,
    get_driver_target_location_only_reward,
    get_driver_target_location,
    get_driver_constraint_target_velocity,
    get_driver_constraint_target_location,
)
from constrained_bai.driver_bandit import DriverBandit


def main():
    filename = "driver_candidate_policies.pkl"

    with open(filename, "rb") as f:
        policies = pickle.load(f)

    print(f"Loaded {len(policies)} policies...")

    # env = get_driver_target_velocity_only_reward()
    # env = get_driver_target_velocity()
    # env = get_driver_constraint_target_velocity()
    # env = get_driver_target_location_only_reward()
    # env = get_driver_target_location()
    # env = get_driver_constraint_target_location()
    # env = get_driver_target_velocity_only_reward(blocking_cars=True)
    env = get_driver_target_velocity(blocking_cars=True)
    # env = get_driver_constraint_target_velocity(blocking_cars=True)

    do_ce = False

    bandit = DriverBandit(env, policies, None)
    policy_bandit = bandit.policies[bandit.best_arm_i]

    if do_ce:
        policy_ce, _, _ = env.get_optimal_policy()

    s = env.reset()
    done = False
    r = 0
    while not done:
        a = policy_bandit[int(s[-1])]
        s, reward, done, info = env.step(a)
        # print("action", a)
        # print("state", s)
        # print("features", env.get_features())
        r += reward
        # env.render("human")
        # time.sleep(0.5)

    print("Policy bandit reward:", r)
    env.plot_history()
    plt.show()

    if do_ce:
        s = env.reset()
        done = False
        r = 0
        while not done:
            a = policy_ce[int(s[-1])]
            s, reward, done, info = env.step(a)
            # print("action", a)
            # print("state", s)
            # print("features", env.get_features())
            r += reward
            # env.render("human")
            # time.sleep(0.5)

        print("Policy CE reward:", r)
        env.plot_history()
        plt.show()


if __name__ == "__main__":
    main()
