import math
from argparse import ArgumentParser

import numpy as np


class EPR(object):

    def __init__(
        self,
        transition_matrix,
        home_location,
        p_t_raw,
        time_slot=30,
        rho=0.6,
        gamma=0.41,
        alpha=1.86,
        n_w=6.1,
        beta1=3.67,
        beta2=10,
        simu_slot=48,
    ):

        super().__init__()
        self.time_slot = time_slot  # time resolution is half an hour
        self.rho = rho  # it controls the exploration probability for other regions
        self.gamma = (
            gamma  # it is the attenuation parameter for exploration probability
        )
        self.alpha = alpha  # it controls the exploration depth
        self.n_w = n_w  # it is the average number of tour based on home a week.
        self.beta1 = beta1  # dwell rate
        self.beta2 = beta2  # burst rate
        self.simu_slot = simu_slot
        self.pop_num = home_location.shape[0]

        self.transition_matrix = transition_matrix
        self.p_t = np.array(p_t_raw).reshape(-1, (time_slot // 30)).sum(axis=1)
        self.region_num = self.transition_matrix.shape[0]
        self.home_location = home_location

    def run(self):
        self.pop_info = self.trace_simulate()

    def distance(self, p1, p2):  # caculate distance
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_p_t(self, now_time):
        i = int(now_time / (self.time_slot * 60))
        # print(i)
        return self.p_t[i]

    def predict_next_place_time(self, p_t_value, current_location_type):
        p1 = 1 - self.n_w * p_t_value
        p2 = 1 - self.beta1 * self.n_w * p_t_value
        p3 = self.beta2 * self.n_w * p_t_value
        location_is_change = 0
        new_location_type = "undefined"
        if current_location_type == "home":
            if np.random.uniform(0, 1) <= p1:
                new_location_type = "home"
                location_is_change = 0
            else:
                new_location_type = "other"
                location_is_change = 1
        elif current_location_type == "other":
            p = np.random.uniform(0, 1)
            if p <= p2:
                new_location_type = "other"
                location_is_change = 0
            elif np.random.uniform(0, 1) <= p3:
                new_location_type = "other"
                location_is_change = 1
            else:
                new_location_type = "home"
                location_is_change = 1
        if new_location_type == "home":
            return 0, location_is_change
        else:
            return 2, location_is_change

    def negative_pow(self, k):
        p_k = {}
        for i, region in enumerate(k, 1):
            p_k[region[0]] = i ** (-self.alpha)
        temp = sum(p_k.values())
        for key in p_k:
            p_k[key] = p_k[key] / temp
        return p_k

    def predict_next_place_location_simplify(
        self, P_new, region_history, current_region, home_region
    ):
        rp = np.random.uniform(0, 1)
        prob_accum, next_region = 0, 0
        if np.random.uniform(0, 1) < P_new:
            # explore; choose the next region based on the transition matrix
            transition = self.transition_matrix[current_region].copy()
            transition[home_region] = 0
            for key in region_history:
                transition[key] = 0
            transition += 1e-6
            trans_prob = transition / sum(transition)
            next_region = np.random.choice(range(self.region_num), p=trans_prob)
            region_history[next_region] = 1
        else:
            region_history_sum = sum(list(region_history.values()))
            for key in region_history:
                prob_accum += region_history[key] / region_history_sum
                if rp < prob_accum:
                    next_region = key
                    region_history[key] += 1
                    break
        return next_region

    def predict_next_place_location(
        self, region_history, current_location, home_region
    ):
        s = len(region_history.values())
        p_new = self.rho * s ** (-self.gamma) if s != 0 else 1
        return self.predict_next_place_location_simplify(
            p_new, region_history, current_location, home_region
        )

    def individual_trace_simulate(self, info, start_time, simu_slot):
        current_location_type = "home"
        simu_trace = [[info["home"], start_time]]
        for i in range(simu_slot - 1):
            # pt is the currently move based probability
            now_time = (i + 1) * 60 * self.time_slot + start_time
            p_t_value = self.get_p_t(now_time)
            now_type, location_change = self.predict_next_place_time(
                p_t_value, current_location_type
            )
            if location_change == 1:
                current_location = simu_trace[-1][0]
                if now_type == 0:
                    next_location = info["home"]
                    current_location_type = "home"
                else:
                    next_location = self.predict_next_place_location(
                        info["region_history"], current_location, info["home"]
                    )
                    current_location_type = "other"
            else:
                next_location = simu_trace[-1][0]
            simu_trace.append([next_location, now_time])
        return simu_trace

    def trace_simulate(self):
        pop_info = []
        for i in range(self.pop_num):
            pop_info.append(
                {
                    "n_w": self.n_w,
                    "beta1": self.beta1,
                    "beta2": self.beta2,
                    "home": self.home_location[i],
                    "region_history": {},
                }
            )
            pop_info[i]["trace"] = np.array(
                self.individual_trace_simulate(pop_info[i], 0, self.simu_slot)
            )[:, 0]
        return pop_info


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--num-traj", type=int, default=1000)
    parser.add_argument("--save", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # transition matrix
    transition_matrix = np.load("path/to/transition_matrix.npy")
    # move probability
    move_prob = np.load("path/to/move_prob.npy")
    # population
    population = np.load("path/to/population.npy")
    traj_epr = []
    for _ in range(args.num_traj):
        # choose home based on population
        home_prob = population / population.sum()
        home = np.random.choice(range(len(home_prob)), p=home_prob)
        epr = EPR(transition_matrix, home, move_prob)
        epr.run()
        traj = np.array([info["trace"] for info in epr.pop_info])
        traj_epr.append(traj)
    traj_epr = np.concatenate(traj_epr, axis=0)
    np.save(args.save, traj_epr)
