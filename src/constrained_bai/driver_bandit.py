from constrained_bai.bandit import ConstrainedLinearBandit

import numpy as np


class DriverBandit(ConstrainedLinearBandit):
    def __init__(
        self,
        driver_env,
        candidate_policies,
        noise_function,
        normalize_constraints=False,
    ):
        self.env = driver_env
        self.policies = candidate_policies
        arms = []
        for policy in self.policies:
            r_features, c_features = self.env.get_features_from_policy(policy)
            arms.append(c_features)
        super().__init__(
            arms,
            self.env.reward_w,
            self.env.constraint_w,
            self.env.threshold,
            noise_function,
            need_to_learn_threshold=True,
            normalize_constraints=normalize_constraints,
        )
