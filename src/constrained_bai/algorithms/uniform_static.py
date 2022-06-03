import numpy as np

from constrained_bai.util.helpers import (
    compute_xnorm,
)

from .static_allocation_algorithm import CLBStaticAllocationAlgorithm


class UniformSamplingStatic(CLBStaticAllocationAlgorithm):
    def get_allocation(self, bandit, uncertain_arms):
        design = np.ones(bandit.n_arms)
        design /= design.sum()
        return design, 0
