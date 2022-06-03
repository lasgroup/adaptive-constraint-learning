"""
Driving environment based on:
- https://github.com/dsadigh/driving-preferences
- https://github.com/Stanford-ILIAD/easy-active-learning/
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib.image import AxesImage, BboxImage
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.ndimage import rotate, zoom

IMG_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "img", "driver")

GRASS = np.tile(plt.imread(os.path.join(IMG_FOLDER, "grass.png")), (5, 5, 1))

CAR = {
    color: zoom(
        np.array(
            plt.imread(os.path.join(IMG_FOLDER, "car-{}.png".format(color))) * 255.0,
            dtype=np.uint8,  # zoom requires uint8 format
        ),
        [0.3, 0.3, 1.0],
    )
    for color in ["gray", "orange", "purple", "red", "white", "yellow"]
}

COLOR_AGENT = "orange"
COLOR_ROBOT = "white"

CAR_AGENT = CAR[COLOR_AGENT]
CAR_ROBOT = CAR[COLOR_ROBOT]
CAR_SCALE = 0.15 / max(list(CAR.values())[0].shape[:2])

LANE_SCALE = 10.0
LANE_COLOR = (0.4, 0.4, 0.4)  # 'gray'
LANE_BCOLOR = "white"

STEPS = 100


def set_image(
    obj,
    data,
    scale=CAR_SCALE,
    x=[0.0, 0.0, 0.0, 0.0],
):
    ox = x[0]
    oy = x[1]
    angle = x[2]
    img = rotate(data, np.rad2deg(angle))
    h, w = img.shape[0], img.shape[1]
    obj.set_data(img)
    obj.set_extent(
        [
            ox - scale * w * 0.5,
            ox + scale * w * 0.5,
            oy - scale * h * 0.5,
            oy + scale * h * 0.5,
        ]
    )


class Car:
    def __init__(self, initial_state, actions):
        self.initial_state = initial_state
        self.state = self.initial_state
        self.actions = actions
        self.action_i = 0

    def reset(self):
        self.state = self.initial_state
        self.action_i = 0

    def update(self, update_fct) -> None:
        u1, u2 = self.actions[self.action_i % len(self.actions)]
        self.state = update_fct(self.state, u1, u2)
        self.action_i += 1

    def gaussian(self, x, height=0.07, width=0.03):
        car_pos = np.asarray([self.state[0], self.state[1]])
        car_theta = self.state[2]
        car_heading = (np.cos(car_theta), np.sin(car_theta))
        pos = np.asarray([x[0], x[1]])
        d = car_pos - pos
        dh = np.dot(d, car_heading)
        dw = np.cross(d, car_heading)
        return np.exp(-0.5 * ((dh / height) ** 2 + (dw / width) ** 2))


class Lane:
    def __init__(
        self,
        start_pos,
        end_pos,
        width,
    ):
        self.start_pos = np.asarray(start_pos)
        self.end_pos = np.asarray(end_pos)
        self.width = width
        d = self.end_pos - self.start_pos
        self.dir = d / np.linalg.norm(d)
        self.perp = np.asarray([-self.dir[1], self.dir[0]])

    def gaussian(self, state, sigma=0.5):
        pos = np.asarray([state[0], state[1]])
        dist_perp = np.dot(pos - self.start_pos, self.perp)
        return np.exp(-0.5 * (dist_perp / (sigma * self.width / 2.0)) ** 2)

    def direction(self, x):
        return np.cos(x[2]) * self.dir[0] + np.sin(x[2]) * self.dir[1]

    def shifted(self, m):
        return Lane(
            self.start_pos + self.perp * self.width * m,
            self.end_pos + self.perp * self.width * m,
            self.width,
        )


def get_lane_x(lane):
    if lane == "left":
        return -0.17
    elif lane == "right":
        return 0.17
    elif lane == "middle":
        return 0
    else:
        raise Exception("Unknown lane:", lane)


class Driver:
    def __init__(
        self,
        cars,
        reward_weights,
        constraint_weights=None,
        threshold=0,
        starting_lane="middle",
        starting_speed=0.41,
    ):
        initial_x = get_lane_x(starting_lane)
        self.initial_state = [initial_x, -0.1, np.pi / 2, starting_speed]
        self.state = self.initial_state

        self.episode_length = 20
        self.dt = 0.2

        self.friction = 1
        self.vmax = 1
        self.xlim = (-0.7, 0.7)
        # self.ylim = (-0.2, 0.8)
        self.ylim = (-0.2, 2)
        print("state0", self.state)

        lane = Lane([0.0, -1.0], [0.0, 1.0], 0.17)
        road = Lane([0.0, -1.0], [0.0, 1.0], 0.17 * 3)
        self.lanes = [lane.shifted(0), lane.shifted(-1), lane.shifted(1)]
        self.fences = [lane.shifted(2), lane.shifted(-2)]
        self.roads = [road]
        self.cars = cars

        n_features_reward = len(self.get_reward_features())
        assert reward_weights.shape == (n_features_reward,)
        self.reward_w = np.array(reward_weights)
        self.n_features_reward = n_features_reward

        n_features_constraint = len(self.get_constraint_features())
        if constraint_weights is not None:
            assert constraint_weights.shape == (n_features_constraint,)
            self.constraint_w = np.array(constraint_weights)
            self.threshold = threshold
        else:
            self.constraint_w = None
            self.threshold = 0
        self.n_features_constraint = n_features_constraint

        self.action_d = 2
        self.action_min = np.array([-1, -1])
        self.action_max = np.array([1, 1])

        self.time = 0
        self.history = []
        self._update_history()

    def _update_history(self):
        self.history.append((np.array(self.state), self._get_car_states()))

    def _get_car_states(self):
        return [np.array(car.state) for car in self.cars]

    def _update_state(self, state, u1, u2):
        x, y, theta, v = state
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * u1
        dv = u2 - self.friction * v
        new_v = max(min(v + dv * self.dt, self.vmax), -self.vmax)
        return [x + dx * self.dt, y + dy * self.dt, theta + dtheta * self.dt, new_v]

    def _get_reward_for_state(self, state=None):
        if state is None:
            state = self.state
        return np.dot(self.reward_w, self.get_reward_features(state))

    def step(self, action):
        action = np.array(action)
        u1, u2 = action

        self.state = self._update_state(self.state, u1, u2)
        for car in self.cars:
            car.update(self._update_state)

        self.time += 1
        done = bool(self.time >= self.episode_length)
        reward = self._get_reward_for_state()
        self._update_history()
        return np.array(self.state + [self.time]), reward, done, dict()

    def reset(self):
        self.state = self.initial_state
        self.time = 0
        for car in self.cars:
            car.reset()
        self.history = []
        self._update_history()
        return np.array(self.state + [self.time])

    def get_reward_features(self, state=None):
        return self._get_features(state=state)

    def get_constraint_features(self, state=None):
        return self._get_features(state=state)

    def _get_features(self, state=None):
        if state is None:
            state = self.state
        x, y, theta, v = state

        off_street = int(np.abs(x) > self.roads[0].width / 2)

        b = 10000
        a = 10
        d_to_lane = np.min([(x - 0.17) ** 2, x**2, (x + 0.17) ** 2])
        not_in_lane = 1 / (1 + np.exp(-b * d_to_lane + a))

        big_angle = np.abs(np.cos(theta))

        drive_backward = int(v < 0)
        too_fast = int(v > 0.6)

        distance_to_other_car = 0
        b = 30
        a = 0.01
        for car in self.cars:
            car_x, car_y, car_theta, car_v = car.state
            distance_to_other_car += np.exp(
                -b * (10 * (x - car_x) ** 2 + (y - car_y) ** 2) + b * a
            )

        keeping_speed = -np.square(v - 0.4)
        target_location = -np.square(x - 0.17)

        return np.array(
            [
                keeping_speed,
                target_location,
                off_street,
                not_in_lane,
                big_angle,
                drive_backward,
                too_fast,
                distance_to_other_car,
            ],
            dtype=float,
        )

    def _get_features_from_flat_policy(self, policy):
        a_dim = self.action_d
        n_policy_steps = len(policy) // a_dim
        n_repeat = self.episode_length // n_policy_steps

        self.reset()
        r_features = np.zeros_like(self.get_reward_features())
        c_features = np.zeros_like(self.get_constraint_features())
        for i in range(self.episode_length):
            if i % n_repeat == 0:
                action_i = a_dim * (i // n_repeat)
                action = (policy[action_i], policy[action_i + 1])
            s, _, done, _ = self.step(action)
            assert (i < self.episode_length - 1) or done
            r_features += self.get_reward_features()
            c_features += self.get_constraint_features()
        return r_features, c_features

    def get_features_from_policy(self, policy):
        a_dim = self.action_d
        n_policy_steps = len(policy)
        n_repeat = self.episode_length // n_policy_steps

        self.reset()
        r_features = np.zeros_like(self.get_reward_features())
        c_features = np.zeros_like(self.get_constraint_features())
        for i in range(self.episode_length):
            if i % n_repeat == 0:
                action = policy[i]
            s, _, done, _ = self.step(action)
            assert (i < self.episode_length - 1) or done
            r_features += self.get_reward_features()
            c_features += self.get_constraint_features()
        return r_features, c_features

    def get_optimal_policy(
        self,
        phi=None,
        threshold=0,
        theta=None,
        n_action_repeat=1,
        iterations=50,
        n_candidates=50,
        n_elite=5,
        verbose=False,
    ):
        """Implements a cross entropy method for (constrained) policy optimization.

        If phi is given to define a constraint function, the ordering defined by [1]
        is used to determine elite policies. Otherwise an ordering by rewards is used,
        i.e. the algorithm defaults to a vanilla CE method.

        [1] Wen, Min, and Ufuk Topcu. "Constrained cross-entropy method for safe
            reinforcement learning." IEEE Transactions on Automatic Control (2020).
        """
        a_dim = self.action_d
        n_policy_steps = self.episode_length // n_action_repeat

        a_low = np.array(list(self.action_min) * n_policy_steps)
        a_high = np.array(list(self.action_max) * n_policy_steps)

        if theta is None:
            theta = self.reward_w
        if phi is None:
            phi = self.constraint_w
            threshold = self.threshold

        if verbose:
            print("theta", theta)
            print("phi", phi)
            print("threshold", threshold)

        mu = np.zeros(n_policy_steps * a_dim)
        std = 5 * np.ones(n_policy_steps * a_dim)

        optimal_policy = mu
        optimal_rew = -np.inf

        for i in range(iterations):
            if verbose:
                print("iteration", i)

            policies = []
            rewards = []
            constraints = []

            for j in range(n_candidates):
                policy = mu + std * np.random.randn(n_policy_steps * a_dim)
                policy = np.clip(policy, a_low, a_high)
                policies.append(policy)
                r_features, c_features = self._get_features_from_flat_policy(policy)
                reward = np.dot(r_features, theta)
                rewards.append(reward)
                if phi is not None:
                    constraint = np.dot(c_features, phi)
                    constraints.append(constraint)
                if reward > optimal_rew and (phi is None or constraint <= threshold):
                    # this works because dynamics are deterministic
                    optimal_policy = policy
                    optimal_rew = reward

            if phi is None:
                # unconstrained optimization
                idx = np.argsort(rewards)[::-1]
                elite = [policies[k] for k in idx[:n_elite]]
            else:
                idx = np.argsort(constraints)
                if constraints[idx[n_elite - 1]] > threshold:
                    elite = [policies[k] for k in idx[:n_elite]]
                else:
                    feasible = [
                        k for k in range(n_candidates) if constraints[k] <= threshold
                    ]
                    feasible = sorted(feasible, key=lambda k: -rewards[k])
                    elite = [policies[k] for k in feasible[:n_elite]]

            mu = np.array(elite).mean(axis=0)
            std = np.array(elite).std(axis=0)

            if verbose:
                print("mu", mu)
                print("std", std)

        r_features, c_features = self._get_features_from_flat_policy(optimal_policy)

        if verbose:
            print()
            print("optimal_policy", optimal_policy)
            print("f_reward", r_features)
            print("rew", np.dot(r_features, theta))
            if phi is not None:
                print("f_constraint", c_features)
                print("cons", np.dot(c_features, phi))
            print()

        policy_repeat = []
        for i in range(n_policy_steps):
            policy_repeat.extend([optimal_policy[2 * i : 2 * i + 2]] * n_action_repeat)

        return np.array(policy_repeat), r_features, c_features

    def render(self, mode="human"):
        if mode not in ("human", "rgb_array", "human_static"):
            raise NotImplementedError("render mode {} not supported".format(mode))
        fig = plt.figure(figsize=(7, 7))

        ax = plt.gca()
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.set_aspect("equal")

        grass = BboxImage(ax.bbox, interpolation="bicubic", zorder=-1000)
        grass.set_data(GRASS)
        ax.add_artist(grass)

        for lane in self.lanes:
            path = Path(
                [
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    - lane.perp * lane.width * 0.5,
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    + lane.perp * lane.width * 0.5,
                    lane.end_pos + LANE_SCALE * lane.dir + lane.perp * lane.width * 0.5,
                    lane.end_pos + LANE_SCALE * lane.dir - lane.perp * lane.width * 0.5,
                    lane.start_pos
                    - LANE_SCALE * lane.dir
                    - lane.perp * lane.width * 0.5,
                ],
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
            )
            ax.add_artist(
                PathPatch(
                    path,
                    facecolor=LANE_COLOR,
                    lw=0.5,
                    edgecolor=LANE_BCOLOR,
                    zorder=-100,
                )
            )

        for car in self.cars:
            img = AxesImage(ax, interpolation="bicubic", zorder=20)
            set_image(img, CAR_ROBOT, x=car.state)
            ax.add_artist(img)

        human = AxesImage(ax, interpolation=None, zorder=100)
        set_image(human, CAR_AGENT, x=self.state)
        ax.add_artist(human)

        plt.axis("off")
        plt.tight_layout()
        if mode != "human_static":
            fig.canvas.draw()
            rgb = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
            rgb = rgb.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            del fig
            if mode == "rgb_array":
                return rgb
            elif mode == "human":
                plt.imshow(rgb, origin="upper")
                plt.axis("off")
                plt.tight_layout()
                plt.pause(0.05)
                plt.clf()
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def plot_history(self):
        x_player = []
        y_player = []
        N_cars = len(self.cars)
        x_cars = [[] for _ in range(N_cars)]
        y_cars = [[] for _ in range(N_cars)]
        for player_state, car_states in self.history:
            x_player.append(player_state[0])
            y_player.append(player_state[1])
            for i in range(N_cars):
                x_cars[i].append(car_states[i][0])
                y_cars[i].append(car_states[i][1])

        # print("x_player", x_player)
        # print("y_player", y_player)
        # print("x_cars", x_cars)
        # print("y_cars", y_cars)

        self.reset()
        self.render(mode="human_static")
        plt.axis("off")
        plt.tight_layout()
        for i in range(N_cars):
            plt.plot(
                x_cars[i],
                y_cars[i],
                zorder=10,
                linestyle="-",
                color=COLOR_ROBOT,
                linewidth=2.5,
                marker="o",
                markersize=8,
                markevery=1,
            )
            plt.plot(
                x_cars[i][self.episode_length // 2],
                y_cars[i][self.episode_length // 2],
                zorder=10,
                linestyle="-",
                color="red",
                linewidth=2.5,
                marker="o",
                markersize=8,
                markevery=1,
            )
        plt.plot(
            x_player,
            y_player,
            zorder=10,
            linestyle="-",
            color=COLOR_AGENT,
            linewidth=2.5,
            marker="o",
            markersize=8,
            markevery=1,
        )
        plt.plot(
            x_player[self.episode_length // 2],
            y_player[self.episode_length // 2],
            zorder=10,
            linestyle="-",
            color="red",
            linewidth=2.5,
            marker="o",
            markersize=8,
            markevery=1,
        )

    def sample_features_rewards(self, n_samples):
        min_val = -1
        max_val = 1
        samples = min_val + (max_val - min_val) * np.random.sample(
            (n_samples, self.Ndim_repr)
        )
        samples[:, -1] = 0
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        samples[:, -1] = 1
        return samples, np.matmul(samples, self.reward_w.T)


def get_cars(cars_trajectory):
    if cars_trajectory == "blocked":
        # three cars
        x1 = -0.17
        y1 = 0.6
        s1 = 0
        car1 = Car([x1, y1, np.pi / 2.0, s1], [(0, s1)] * 20)
        x2 = 0
        y2 = 0.65
        s2 = 0
        car2 = Car([x2, y2, np.pi / 2.0, s2], [(0, s2)] * 20)
        x3 = 0.17
        y3 = 0.7
        s3 = 0
        car3 = Car([x3, y3, np.pi / 2.0, s3], [(0, s3)] * 20)
        cars = [car1, car2, car3]
        # x1 = -0.17
        # y1 = 1.7
        # s1 = 0.4
        # car1 = Car([x1, y1, np.pi / 2.0, s1], [(0, s1)] * 20)
        # x2 = 0
        # y2 = 0.4
        # s2 = 0.15
        # car2 = Car([x2, y2, np.pi / 2.0, s2], [(0, s2)] * 20)
        # x3 = 0.17
        # y3 = 0.6
        # s3 = 0.1
        # car3 = Car([x3, y3, np.pi / 2.0, s3], [(0, s3)] * 20)
        # cars = [car1, car2, car3]
    elif cars_trajectory == "changing_lane":
        # car driving from right to middle lane
        car_x = get_lane_x("right")
        straight_speed = 0.328
        car = Car(
            [car_x, 0, np.pi / 2.0, 0.41],
            [(0, straight_speed)] * 5
            + [(1, straight_speed)] * 6
            + [(-1, straight_speed)] * 6
            + [(0, straight_speed)] * 3,
        )
        cars = [car]
    else:
        raise Exception("Unknown cars trajectory:", cars_trajectory)
    return cars


def get_reward_weights(goal, penalty_lambda):
    if goal == "target_velocity":
        goal_weights = np.array(
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=float,
        )
    elif goal == "target_location":
        goal_weights = np.array(
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=float,
        )
    else:
        raise Exception("Unknown goal:", goal)
    penalty_weights = np.array(
        [
            0,  # keep speed
            0,  # target location
            0.3,  # off street
            0.05,  # not in lane
            0.02,  # big angle
            0.5,  # drive backward
            0.3,  # too fast
            0.8,  # crash
        ],
        dtype=float,
    )
    return goal_weights - penalty_lambda * penalty_weights


def get_constraint_weigths_threshold(constraint):
    if constraint is None:
        return None, None
    elif constraint == "default":
        constraint_weights = np.array(
            [
                0,  # keep speed
                0,  # target location
                0.3,  # off street
                0.05,  # not in lane
                0.02,  # big angle
                0.5,  # drive backward
                0.3,  # too fast
                0.8,  # distance to other car
            ],
            dtype=float,
        )
        threshold = 1
        return constraint_weights, threshold


def get_driver(cars_trajectory, goal, penalty_lambda=0, constraint=None):
    cars = get_cars(cars_trajectory)
    reward_weights = get_reward_weights(goal, penalty_lambda)
    constraint_weights, threshold = get_constraint_weigths_threshold(constraint)

    if cars_trajectory == "blocked":
        starting_speed = 0.1
    else:
        starting_speed = 0.41

    return Driver(
        cars,
        reward_weights,
        constraint_weights=constraint_weights,
        threshold=threshold,
        starting_speed=starting_speed,
    )


def get_driver_target_velocity(blocking_cars=False):
    if blocking_cars:
        cars_trajectory = "blocked"
    else:
        cars_trajectory = "changing_lane"
    return get_driver(cars_trajectory, "target_velocity", penalty_lambda=1)


def get_driver_target_velocity_only_reward(blocking_cars=False):
    if blocking_cars:
        cars_trajectory = "blocked"
    else:
        cars_trajectory = "changing_lane"
    return get_driver(cars_trajectory, "target_velocity", penalty_lambda=0)


def get_driver_target_location():
    return get_driver("changing_lane", "target_location", penalty_lambda=0.5)


def get_driver_target_location_only_reward():
    return get_driver("changing_lane", "target_location", penalty_lambda=0)


def get_driver_constraint_target_velocity(blocking_cars=False):
    if blocking_cars:
        cars_trajectory = "blocked"
    else:
        cars_trajectory = "changing_lane"
    return get_driver(
        cars_trajectory, "target_velocity", penalty_lambda=0, constraint="default"
    )


def get_driver_constraint_target_location():
    return get_driver(
        "changing_lane", "target_location", penalty_lambda=0, constraint="default"
    )


if __name__ == "__main__":
    import time
    import pickle

    # env = get_driver_target_velocity_only_reward()
    env = get_driver_target_velocity()
    # env = get_driver_constraint_target_velocity()
    # env = get_driver_target_location_only_reward()
    # env = get_driver_target_location()
    # env = get_driver_constraint_target_location()
    # env = get_driver_target_velocity(blocking_cars=True)
    # env = get_driver_constraint_target_velocity(blocking_cars=True)

    policy, r_features, c_features = env.get_optimal_policy()

    s = env.reset()
    done = False
    r = 0
    while not done:
        a = policy[int(s[-1])]
        s, reward, done, info = env.step(a)
        # print("action", a)
        # print("state", s)
        # print("features", env.get_features())
        r += reward
        # env.render("human")
        # time.sleep(0.2)

    print("policy features", r_features)
    print("policy features constraint", c_features)
    # print("constraint", np.dot(env.constraint_w, c_features))
    print("return 2", r)
    env.plot_history()
    plt.savefig("driver.pdf")
    plt.show()
