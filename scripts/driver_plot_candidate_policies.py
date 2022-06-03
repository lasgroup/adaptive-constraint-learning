import sys
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
    filename = sys.argv[1]

    with open(filename, "rb") as f:
        policies = pickle.load(f)

    print(f"Loaded {len(policies)} policies...")

    env = get_driver_constraint_target_location()
    bandit = DriverBandit(env, policies, None)

    for i, policy in enumerate(policies):
        print("Policy", i)
        s = env.reset()
        done = False
        r = 0
        while not done:
            a = policy[int(s[-1])]
            s, reward, done, info = env.step(a)
            r += reward

        print("Policy reward:", r)
        print("Bandit reward:", bandit.get_reward(i))
        print("Bandit constraint:", bandit.get_constraint(i))
        print("Bandit feasible:", bandit.is_feasible(i))

        env.plot_history()
        plt.show()


if __name__ == "__main__":
    main()
