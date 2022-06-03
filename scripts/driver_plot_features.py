import sys
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np

from constrained_bai.driver import get_driver_constraint_target_velocity


def main():
    env = get_driver_constraint_target_velocity()
    env.reset()
    env.cars[0].state = [0, 0.8, np.pi / 2, 0]

    xmin, xmax = env.xlim
    ymin, ymax = env.ylim

    n_x = 100
    n_y = 500

    # f = 2  # stay on street
    # f = 3  # stay in lane
    f = 7  # distance to other car

    xrange = np.linspace(xmin, xmax, n_x)
    yrange = np.linspace(ymin, ymax, n_y)
    xx, yy = np.meshgrid(xrange, yrange)
    v = np.zeros_like(xx)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            x, y = xx[i, j], yy[i, j]
            state = [x, y, np.pi / 2, 0]
            feat = env.get_constraint_features(state=state)[f]
            # print(x, y, feat)
            v[i, j] = feat

    env.render(mode="human_static")
    plt.imshow(v, extent=[xmin, xmax, ymin, ymax], origin="lower", alpha=0.5)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
