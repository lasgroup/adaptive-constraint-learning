from abc import abstractmethod

import numpy as np

from constrained_bai.util.helpers import compute_xnorm
from .base import CLBAlgorithm


class CLBStaticAllocationAlgorithm(CLBAlgorithm):
    def __init__(self, delta=0.01, phase_base=2, debug=False):
        self.delta = delta
        self.debug = debug
        self.phase_base = phase_base
        self.fixed_allocation = True

    def run(self, bandit):
        """
        Run a static allocation algorithm.

        First determine the allocation, then pull arms (stochasically) following
        this allocation. After each observation update the model and check a
        stopping condition.

        The allocation and the stopping condition have to be implemented.
        """
        phase_index = 0
        N = 0

        uncertain_arms = list(range(bandit.n_arms))
        best_feasible_arm = None
        best_feasible_val = -float("inf")

        if self.fixed_allocation:
            design, design_val = self.get_allocation(bandit, uncertain_arms)

        stop = False
        while not stop:
            phase_index += 1
            delta_t = self.delta**2 / (phase_index**2 * bandit.n_arms)
            sqbeta = np.sqrt(2 * np.log(1 / delta_t))

            if not self.fixed_allocation:
                design, design_val = self.get_allocation(bandit, uncertain_arms)
            phase_length = self.get_phase_length(phase_index, delta_t, design_val)

            allocation = self.round(design, phase_length)
            # allocation = np.random.multinomial(phase_length, pvals=design)

            # print(f"Round {phase_index} allocation:", allocation)
            # print("Design:", np.round(design, 3))
            # print("Allocation:", np.round(allocation, 0))

            lam = 0.01
            A = lam * np.identity(bandit.d)
            K = np.zeros(bandit.d)

            for arm_i, n_pulls in enumerate(allocation):
                if n_pulls > 0:
                    n_pulls = int(n_pulls)
                    feat = bandit.constraint_features[arm_i]
                    feat_t = np.reshape(feat, (bandit.d, 1))
                    N += n_pulls
                    A += n_pulls * np.outer(feat, feat)
                    obs = bandit.observe_constraint(arm_i, n=n_pulls)
                    # if n_pulls > 1:
                    #     breakpoint()
                    K += np.repeat(feat_t, n_pulls, axis=1) @ obs

            A_inv = np.linalg.inv(A)
            Xnorm = compute_xnorm(bandit.arms_X_constraint, A_inv)
            self.phihat = A_inv @ K

            # check stopping condition
            new_uncertain_arms = []
            for arm_i in uncertain_arms:
                center_i = np.dot(bandit.arms_X_constraint[arm_i], self.phihat)
                xnorm_i = Xnorm[arm_i]
                min_constraint = center_i - sqbeta * xnorm_i
                if min_constraint < bandit.threshold:
                    max_constraint = center_i + sqbeta * xnorm_i
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

            if self.debug:
                yhat = bandit.arms_X_constraint @ self.phihat
                print("yhat", yhat)
                print("Xnorm", Xnorm)
                print("phihat", np.round(self.phihat, 3))

            if len(uncertain_arms) == 0:
                stop = True
                return_arm_i = best_feasible_arm

        if return_arm_i is None:
            print("Unable to find optimal arm")
            print("uncertain_arms:", uncertain_arms)

        print(f"Best arms after {N} iterations: {return_arm_i}")
        return N, return_arm_i

    def get_phase_length(self, phase_index, delta_t, design_val):
        return np.ceil((self.phase_base**phase_index) * np.log(1 / delta_t))

    def round(self, design, N):
        """Implements the efficient rounding procedure described in Chapter 12 of [1].

        [1] Friedrich Pukelsheim. Optimal design of experiments. SIAM, 2006.
        """
        support = design > 0
        where_support = np.where(support)[0]
        p = np.sum(support)  # cardinality of support
        allocation = np.ceil((N - 0.5 * p) * design)
        while np.sum(allocation) != N:
            if np.sum(allocation) > N:
                j = np.argmax(allocation[support] / design[support])
                allocation[where_support[j]] -= 1
            else:  # np.sum(allocation) < N
                j = np.argmin((allocation[support] - 1) / design[support])
                allocation[where_support[j]] += 1
        return allocation

    @abstractmethod
    def get_allocation(self, bandit, uncertain_arms):
        pass
