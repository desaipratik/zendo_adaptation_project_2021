from scipy.optimize import minimize
import numpy as np
import math
import random
import numpy as np

def get_tau(tau, raw_probs, data):
    probs = []
    for a,b in zip(raw_probs, data):
        prob = math.e**(a/tau) / (math.e**(a/tau) + math.e**((1-a)/tau))
        if b == 0:
            prob = 1 - prob
        probs.append(prob)
    return -np.sum([np.log(i) for i in probs])

def fitted_probs(fitted_tau,raw_probs,data):
    probs = []
    for a,b in zip(raw_probs, data):
        prob = math.e**(a/fitted_tau) / (math.e**(a/fitted_tau) + math.e**((1-a)/fitted_tau))
        if b == 0:
            prob = 1 - prob
        probs.append(prob)
    return [probs, np.sum([np.log(i) for i in probs])]

def compare_BIC(ll_m, ll_b, n):
    return (-2 * ll_m + 1 * np.log(n)) < (-2 * ll_b)

def hard_max_selections(select_probs):
    return [np.round(i) for i in select_probs]

def compute_distance(rule_1,rule_2):
    if rule_1 == rule_2:
        return 0
    else:
        for a,b in zip(rule_1,rule_2):
            if a != b:
                step_1 = sum([sum(i) for i in rule_1[rule_1.index(a):]])
                print(step_1)
                step_2 = sum([sum(i) for i in rule_2[rule_2.index(b):]])
                print(step_2)
            return step_1 + step_2


def compute_acc(groud_truth, resp):
    return sum([a and b or not a and not b for a,b in zip(groud_truth, resp)]) / len(resp)

