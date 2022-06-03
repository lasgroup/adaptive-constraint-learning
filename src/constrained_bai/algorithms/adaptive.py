from abc import abstractmethod

import numpy as np

from constrained_bai.util.helpers import compute_xnorm, frank_wolfe_allocation
from .static_allocation_algorithm import CLBStaticAllocationAlgorithm


class CLBAdaptiveAlgorithm(CLBStaticAllocationAlgorithm):
    def __init__(self, delta=0.01, epsilon=0.2, phase_base=2, debug=False):
        super().__init__(delta=delta, phase_base=phase_base, debug=debug)
        self.epsilon = epsilon
        self.fixed_allocation = False

    def get_phase_length(self, phase_index, delta_t, design_val):
        return np.ceil(
            np.log(1 / delta_t)
            * (1 + self.epsilon)
            * design_val
            * self.phase_base ** (2 * phase_index + 3)
        )

    def get_allocation(self, bandit, uncertain_arms):
        return frank_wolfe_allocation(
            bandit, bandit.arms_X[uncertain_arms], oracle=False
        )
