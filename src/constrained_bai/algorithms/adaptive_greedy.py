from abc import abstractmethod

import numpy as np

from constrained_bai.util.helpers import compute_xnorm
from .base import CLBAlgorithm


class CLBAdaptiveGreedyAlgorithm(CLBAlgorithm):
    def __init__(
        self,
        delta=0.01,
        tuned_ci=False,
        selection="adaptive",
        sqbeta_tuned=0.5,
        debug=False,
    ):
        self.delta = delta
        self.debug = debug
        self.tuned_ci = tuned_ci
        assert selection in (
            "adaptive",
            "maxvar_all",
            "uniform",
            "uniform_all",
            "greedy_reward_uncertain",
            "greedy_reward_feasible",
        )
        self.selection = selection
        self.sqbeta_tuned = sqbeta_tuned

    def run(self, bandit, logging_callback=None):
        """
        Run the adaptive algorithm.
        """
        N = 0

        lam = 0.1
        A = lam * np.identity(bandit.d)
        A_inv = np.linalg.inv(A)
        K = np.zeros((bandit.d, 1))

        all_arms = list(range(bandit.n_arms))
        uncertain_arms = all_arms
        best_feasible_arm = None
        lowest_max_constraint_arm = None
        lowest_max_constraint = float("inf")
        best_feasible_val = -float("inf")

        arm_count = np.ones(bandit.n_arms)

        stop = False
        while not stop:
            N += 1

            if self.selection == "adaptive":
                arm_i, A_inv = self.get_next_arm(bandit, A_inv, uncertain_arms)
            elif self.selection == "maxvar_all":
                arm_i, A_inv = self.get_next_arm(bandit, A_inv, all_arms)
            elif self.selection == "uniform":
                arm_i, A_inv = self.get_next_arm_random(bandit, A_inv, uncertain_arms)
            elif self.selection == "uniform_all":
                arm_i, A_inv = self.get_next_arm_random(bandit, A_inv, all_arms)
            elif self.selection == "greedy_reward_uncertain":
                arm_i, A_inv = self.get_next_arm_greedy_reward(
                    bandit, A_inv, uncertain_arms
                )
            elif self.selection == "greedy_reward_feasible":
                if best_feasible_arm is not None:
                    arm_i, A_inv = self.get_next_arm_greedy_reward(
                        bandit, A_inv, [best_feasible_arm]
                    )
                else:
                    arm_i, A_inv = self.get_next_arm_random(bandit, A_inv, all_arms)
            else:
                raise Exception("Unknown selection criterion:", self.selection)

            arm_count[arm_i] += 1

            feat = bandit.constraint_features[arm_i]
            obs = bandit.observe_constraint(arm_i)
            K += np.reshape(feat, (bandit.d, 1)) * obs

            if self.debug:
                print("N", N)
                print("query", feat)
                print("observe", obs)

            Xnorm = compute_xnorm(bandit.arms_X_constraint, A_inv)
            self.phihat = A_inv @ K

            new_uncertain_arms = []

            R = 1  # R-subgaussian
            L = 1  # arm norm
            S = 1  # constraint param norm
            if self.tuned_ci:
                sqbeta = self.sqbeta_tuned
            else:
                sqbeta = (
                    R * np.sqrt(bandit.d * np.log((1 + N * L**2 / lam) / self.delta))
                    + np.sqrt(lam) * S
                )

            if self.debug:
                print("N", N, ";   sqbeta", sqbeta)

            for arm_i in uncertain_arms:
                c = np.dot(bandit.constraint_features[arm_i], self.phihat)
                s = Xnorm[arm_i]

                min_constraint = c - sqbeta * s
                max_constraint = c + sqbeta * s

                if max_constraint < lowest_max_constraint:
                    lowest_max_constraint = max_constraint
                    lowest_max_constraint_arm = arm_i

                if self.debug:
                    # print("arm", bandit.constraint_features[arm_i])
                    print("\tmin_constraint", min_constraint)

                if min_constraint < bandit.threshold:
                    if self.debug:
                        print("\tmax_constraint", max_constraint)
                    if max_constraint < bandit.threshold:
                        val = bandit.get_reward(arm_i)
                        if val > best_feasible_val:
                            best_feasible_val = val
                            best_feasible_arm = arm_i
                    else:
                        new_uncertain_arms.append(arm_i)

            uncertain_arms = []
            for arm_i in new_uncertain_arms:
                val = bandit.get_reward(arm_i)
                if val >= best_feasible_val:
                    uncertain_arms.append(arm_i)

            if len(uncertain_arms) == 0:
                stop = True

            if logging_callback is not None:
                rec_arm = (
                    best_feasible_arm
                    if best_feasible_arm is not None
                    else lowest_max_constraint_arm
                )
                logging_callback(bandit, N, self.phihat, rec_arm)

            if self.debug:
                yhat = bandit.arms_X_constraint @ self.phihat
                print("yhat", yhat)
                print("Xnorm", Xnorm)
                print("phihat", np.round(self.phihat, 3))

            if self.debug:
                print("uncertain_arms", uncertain_arms)

        print(f"Best arms after {N} iterations: {best_feasible_arm}")
        return N, best_feasible_arm

    def _get_new_A_inv(self, A_inv, x):
        new_A_inv = np.copy(A_inv)
        new_A_inv -= (new_A_inv @ x @ x.T @ new_A_inv) / (1 + x.T @ new_A_inv @ x)
        return new_A_inv

    def get_next_arm(self, bandit, A_inv, select_from):
        rho = compute_xnorm(bandit.arms_X_constraint[select_from], A_inv)
        max_i = np.argmax(rho)
        max_x = np.reshape(bandit.arms_X_constraint[select_from][max_i], (bandit.d, 1))

        value = float("inf")
        min_i = None
        min_A_inv = None

        for arm_i in range(bandit.n_arms):
            x = np.reshape(bandit.constraint_features[arm_i], (bandit.d, 1))
            new_A_inv = self._get_new_A_inv(A_inv, x)
            new_xnorm = max_x.T @ new_A_inv @ max_x

            if new_xnorm < value:
                min_i = arm_i
                value = new_xnorm
                min_A_inv = new_A_inv

        return min_i, min_A_inv

    def get_next_arm_random(self, bandit, A_inv, select_from):
        arm_i = np.random.choice(select_from)
        x = np.reshape(bandit.constraint_features[arm_i], (bandit.d, 1))
        return arm_i, self._get_new_A_inv(A_inv, x)

    def get_next_arm_greedy_reward(self, bandit, A_inv, select_from):
        best_arm_i = None
        best_reward = -float("inf")
        for arm_i in select_from:
            reward = bandit.get_reward(arm_i)
            if reward > best_reward:
                best_arm_i, best_reward = arm_i, reward
        x = np.reshape(bandit.constraint_features[best_arm_i], (bandit.d, 1))
        return best_arm_i, self._get_new_A_inv(A_inv, x)
