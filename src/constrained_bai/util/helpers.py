import numpy as np
import scipy.optimize


def get_key_or_default(dictionary, key, default):
    if key in dictionary:
        return dictionary[key]
    else:
        return default


def get_key_or_none(dictionary, key):
    return get_key_or_default(dictionary, key, None)


def compute_xnorm(arms_X, A_inv):
    """
    Computes || x ||_{A^{-1}} for each arm x in arms_X.
    """
    Xnorm = []
    for arm_x in arms_X:
        x = np.array(arm_x)
        xnorm = np.matmul(x, np.matmul(A_inv, x))
        xnorm = np.sqrt(xnorm)
        Xnorm.append(xnorm)
    return np.array(Xnorm)


# def compute_xnorm(arms_X, A_inv):
#     """
#     Computes || x ||_{A^{-1}} for each arm x in arms_X, efficiently.
#     """
#     U, D, V = np.linalg.svd(A_inv)
#     Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T
#     Xnorm = (arms_X @ Ainvhalf) ** 2
#     Xnorm = Xnorm @ np.ones((Xnorm.shape[1], 1))[:, 0]
#     return np.sqrt(Xnorm)

#
# def compute_xnorm(arms_X, A_inv):
#     """
#     Computes || x ||_{A^{-1}} for each arm x in arms_X, efficiently.
#     """
#     U, D, V = np.linalg.svd(A_inv)
#     Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T
#     Xnorm = (arms_X @ Ainvhalf) ** 2
#     Xnorm = Xnorm @ np.ones((Xnorm.shape[1], 1))[:, 0]
#     return Xnorm


def frank_wolfe_allocation(bandit, arms_to_optimize_over, oracle=False):
    """
    Run Frank-Wolfe algorithm to determine a design.
    """

    design = np.ones(bandit.n_arms)
    design /= design.sum()

    max_iter = 1000

    for iter in range(1, max_iter):
        A_inv = np.linalg.pinv(bandit.arms_X.T @ np.diag(design) @ bandit.arms_X)
        Xnorm = compute_xnorm(arms_to_optimize_over, A_inv)

        if oracle:
            # quantity to optimize ||x|| / |phi^T x|
            rho = Xnorm / np.abs(arms_to_optimize_over @ bandit.constraint_param)
        else:
            # quantity to optimize ||x||
            rho = Xnorm

        # compute gradient with respect to lambda and solve linear problem
        idx = np.argmax(rho)
        y = arms_to_optimize_over[idx, :]
        g = ((bandit.arms_X @ A_inv @ y) * (bandit.arms_X @ A_inv @ y)).flatten()
        g_idx = np.argmax(g)

        # perform frank-wolfe update with fixed stepsize
        gamma = 2 / (iter + 2)
        design_update = -gamma * design
        design_update[g_idx] += gamma

        relative = np.linalg.norm(design_update) / (np.linalg.norm(design))

        design += design_update

        if relative < 0.01:  # stop if change in last step is small
            break

    idx_fix = np.where(design < 1e-5)[0]
    drop_total = design[idx_fix].sum()
    design[idx_fix] = 0
    design[np.argmax(design)] += drop_total

    return design, np.max(rho)
