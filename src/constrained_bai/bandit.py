from typing import List, Callable

import numpy as np


class ConstrainedLinearBandit:
    def __init__(
        self,
        arms: List[np.ndarray],
        reward_param: np.ndarray,
        constraint_param: np.ndarray,
        threshold: float,
        noise_function: Callable,
        need_to_learn_threshold: bool = False,
        normalize_constraints: bool = False,
    ):
        self.n_arms = len(arms)
        assert self.n_arms >= 1

        self.max_constraint = -float("inf")
        self.min_constraint = float("inf")
        if constraint_param is not None:
            consts = []
            for arm in arms:
                const = np.dot(arm, constraint_param)
                consts.append(const)
                if const > self.max_constraint:
                    self.max_constraint = const
                if const < self.min_constraint:
                    self.min_constraint = const

        if need_to_learn_threshold:
            reward_param = np.array(list(reward_param) + [0])

            if constraint_param is not None:
                constraint_param = np.array(list(constraint_param) + [0])
                self.max_constraint -= threshold
                self.min_constraint -= threshold
                constraint_param[-1] -= threshold

            threshold = 0

            new_arms = []
            for arm in arms:
                new_arm = np.array(list(arm) + [1])
                new_arms.append(new_arm)
            arms = new_arms

        self.normalize_constraints = normalize_constraints
        self.d = len(arms[0])

        for arm in arms:
            assert arm.shape == (self.d,)
        assert reward_param.shape == (self.d,)
        assert constraint_param is None or constraint_param.shape == (self.d,)

        self.reward_param = reward_param
        self.constraint_param = constraint_param
        self.noise_function = noise_function
        self.threshold = threshold

        self.arms = arms
        self.reward_features = []
        self.constraint_features = []
        for arm_i in range(self.n_arms):
            self.reward_features.append(self._get_reward_features(arm_i))
            self.constraint_features.append(self._get_constraint_features(arm_i))

        self.arms_X_reward = np.stack(self.reward_features, axis=0)
        self.arms_X_constraint = np.stack(self.constraint_features, axis=0)

        self.n_feasible = 0
        self.best_arm_i = None
        self.best_reward = -float("inf")
        for i in range(len(arms)):
            reward_i = self.get_reward(i)
            if self.is_feasible(i):
                self.n_feasible += 1
                if reward_i > self.best_reward:
                    self.best_arm_i = i
                    self.best_reward = reward_i

        self.arms_geq_i = [
            arm_i
            for arm_i in range(self.n_arms)
            if self.get_reward(arm_i) >= self.best_reward
        ]
        self.arms_Xgeq_constraint = self.arms_X_constraint[self.arms_geq_i]

        # legacy interface
        self.arms_X = self.arms_X_constraint
        self.arms_Xgeq = self.arms_Xgeq_constraint

        print("min_constraint", self.min_constraint)
        print("max_constraint", self.max_constraint)
        print("threshold", self.threshold)

        print(f"Number of feasible arms: {self.n_feasible}")
        print(f"Best arm: {self.best_arm_i}  (value: {self.best_reward})")

    def _get_reward_features(self, arm_i):
        return self.arms[arm_i]

    def _get_constraint_features(self, arm_i):
        features = self.arms[arm_i]
        eps = 0.000001  # for numerical stability
        if self.normalize_constraints:
            const = np.dot(self.constraint_param, features)
            if const > self.threshold:
                features /= self.max_constraint - self.threshold
                features -= eps
            else:
                features /= self.threshold - self.min_constraint
                features += eps
        return features

    def get_constraint(self, arm_i):
        return np.dot(self.constraint_param, self.constraint_features[arm_i])

    def get_reward(self, arm_i):
        return np.dot(self.reward_param, self.reward_features[arm_i])

    def observe_constraint(self, arm_i, n=1):
        const = self.get_constraint(arm_i)
        return self.noise_function(const, n=n)

    def is_feasible(self, arm_i):
        if self.constraint_param is None:
            return True
        return self.get_constraint(arm_i) <= self.threshold


def get_toy_instance_1(eps):
    from constrained_bai.noise_models import get_normal

    threshold = 0
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    arms = [0.5 * e1 + 0.5 * e2, 0.5 * e1 - 0.5 * e2, e1 + eps * e2, e1 - eps * e2]
    reward_param = e1 + e2
    constraint_param = e2
    bandit = ConstrainedLinearBandit(
        arms, reward_param, constraint_param, threshold, get_normal(0.05)
    )
    return bandit


def get_toy_instance_2(eps):
    from constrained_bai.noise_models import get_normal

    threshold = 1
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    arms = [e2, (1 - eps) * e1, (1 + eps) * e1]
    reward_param = e1
    constraint_param = e1
    bandit = ConstrainedLinearBandit(
        arms, reward_param, constraint_param, threshold, get_normal(0.05)
    )
    return bandit


def get_toy_instance_3(eps, d):
    from constrained_bai.noise_models import get_normal

    threshold = 1

    arms = []
    for i in range(d - 1):
        x_i = np.zeros(d)
        x_i[i] = 1
        arms.append(x_i)

    e_d = np.zeros(d)
    e_d[-1] = 1

    y1 = (1 - eps) * e_d
    y2 = (1 + eps) * e_d
    arms += [y1, y2]

    reward_param = e_d
    constraint_param = e_d

    bandit = ConstrainedLinearBandit(
        arms, reward_param, constraint_param, threshold, get_normal(0.05)
    )
    return bandit


def get_toy_instance_4(eps, d, n_arms):
    from constrained_bai.noise_models import get_normal

    threshold = 0

    arms = []
    for _ in range(n_arms):
        arm = np.random.multivariate_normal(np.zeros(d), np.identity(d))
        arm /= np.linalg.norm(arm)
        print(arm)
        arms.append(arm)

    reward_param = np.random.multivariate_normal(np.zeros(d), np.identity(d))
    reward_param /= np.linalg.norm(reward_param)
    print("reward_param", reward_param)

    constraint_param = np.random.multivariate_normal(np.zeros(d), np.identity(d))
    constraint_param -= (
        (1 - eps) * np.dot(constraint_param, reward_param) * reward_param
    )
    constraint_param /= np.linalg.norm(constraint_param)
    print("constraint_param", constraint_param)

    bandit = ConstrainedLinearBandit(
        arms, reward_param, constraint_param, threshold, get_normal(0.05)
    )
    return bandit


def get_toy_instance_5():
    from constrained_bai.noise_models import get_normal

    threshold = 0.25

    e1 = np.array([1])
    arms = [
        0.1 * e1,
        0.2 * e1,
        0.3 * e1,
        0.4 * e1,
        0.5 * e1,
        0.6 * e1,
        0.7 * e1,
        0.8 * e1,
        0.9 * e1,
        e1,
    ]
    reward_param = e1
    constraint_param = e1
    bandit = ConstrainedLinearBandit(
        arms, reward_param, constraint_param, threshold, get_normal(0.05)
    )
    return bandit


def get_unit_sphere_1(d, n_arms):
    from constrained_bai.noise_models import get_normal

    threshold = 0

    arms = []
    for _ in range(n_arms):
        arm = np.random.multivariate_normal(np.zeros(d), np.identity(d))
        arm /= np.linalg.norm(arm)
        print(arm)
        arms.append(arm)

    reward_param = np.random.multivariate_normal(np.zeros(d), np.identity(d))
    reward_param /= np.linalg.norm(reward_param)
    print("reward_param", reward_param)

    min_dist = float("inf")
    min_i, min_j = None, None
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            dist = np.linalg.norm(arms[i] - arms[j])
            if dist < min_dist:
                min_i, min_j = i, j
                min_dist = dist

    constraint_param = arms[min_j] - arms[min_i]
    print("constraint_param", constraint_param)

    bandit = ConstrainedLinearBandit(
        arms, reward_param, constraint_param, threshold, get_normal(0.05)
    )
    return bandit


def get_random_instance_1(d, n_arms):
    from constrained_bai.noise_models import get_normal

    threshold = 0
    arms = []
    for _ in range(n_arms):
        arm = np.random.uniform(low=-1, high=1, size=d)
        arms.append(arm)

    reward_param = np.random.uniform(low=-1, high=1, size=d)

    valid = False
    while not valid:
        constraint_param = np.random.uniform(low=-1, high=1, size=d)
        for arm in arms:
            if np.dot(constraint_param, arm) < threshold:
                valid = True

    print("reward_param", reward_param)
    print("constraint_param", constraint_param)

    best_arm, best_value = None, -float("inf")
    for arm_i, arm in enumerate(arms):
        print(f"Arm {arm_i}: {arm}")
        if np.dot(constraint_param, arm) < threshold:
            value = np.dot(reward_param, arm)
            if value > best_value:
                best_arm = arm_i
                best_value = value

    print(f"True best arm: {best_arm}  (value: {best_value})")
    print()

    bandit = ConstrainedLinearBandit(
        arms, reward_param, constraint_param, threshold, get_normal(0.05)
    )
    return bandit
