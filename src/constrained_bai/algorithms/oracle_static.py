import numpy as np

from constrained_bai.util.helpers import compute_xnorm, frank_wolfe_allocation
from .static_allocation_algorithm import CLBStaticAllocationAlgorithm


class CLBOracleStatic(CLBStaticAllocationAlgorithm):
    def get_allocation(self, bandit, uncertain_arms):
        return frank_wolfe_allocation(bandit, bandit.arms_Xgeq, oracle=True)
