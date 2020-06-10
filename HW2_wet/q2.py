import copy
import random

import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np


def run_VI(policy):
    V = {x: 0 for x in S}
    for iteration in range(n_VI_iterations):
        V = VI(policy, V)
    return V


def reward(s):
    return sum([C[job] for job in s])


def VI(pi, state_values):
    for s in S:
        if len(s) == 1:
            next_state = 0
        else:
            temp = list(s)
            temp.remove(pi[s])
            job_finished_state = tuple(temp)
            next_state = state_values[job_finished_state]

        state_values[s] = round(reward(s) + Mu[pi[s]] * next_state + (1 - Mu[pi[s]]) * state_values[s], 3)
    return state_values


def max_cost_pi():
    pi = {}
    for s in S:
        if s == (0,):
            pi[s] = 0
        else:
            max_cost_action = max([C[i] for i in s])
            pi[s] = C.index(max_cost_action)
    return pi


def c_mu_pi():
    pi = {}
    for s in S:
        if s == (0,):
            pi[s] = 0
        else:
            c_mu_actions = [(C[i] * Mu[i], i) for i in s]
            max_c_mu_action = max(c_mu_actions)
            pi[s] = max_c_mu_action[1]
    return pi


def PI(initial_policy):
    current_policy = initial_policy
    policy_value = run_VI(current_policy)
    initial_states_per_iteration = [policy_value[(1, 2, 3, 4, 5)]]
    new_policy = {}
    while new_policy != current_policy:
        current_policy = copy.copy(new_policy)
        for s in S:
            if len(s) == 1:
                new_policy[s] = list(s)[0]
            else:
                min_action_val = float("inf")
                for nxt in combinations(s, len(s) - 1):
                    action = list(set(s) - set(nxt))[0]
                    action_val = (1 - Mu[action]) * policy_value[s] + Mu[action] * policy_value[nxt]
                    if action_val < min_action_val:
                        min_action_val = action_val
                        min_action = action
                new_policy[s] = min_action
        policy_value = run_VI(new_policy)
        initial_states_per_iteration += [policy_value[(1, 2, 3, 4, 5)]]
    return new_policy, initial_states_per_iteration


def simulate(s, a):
    cost = reward(s)
    success = 1 if random.random() < Mu[a] else 0
    if success:
        temp = list(s)
        temp.remove(a)
        next_state = tuple(temp)
    else:
        next_state = s
    return cost, next_state


Mu = [0, 0.6, 0.5, 0.3, 0.7, 0.1]
C = [0, 1, 4, 6, 2, 9]
S = [(0,)]
for i in range(5):
    S += list(combinations([1, 2, 3, 4, 5], i + 1))
n_VI_iterations = 250
if __name__ == '__main__':
    c_mu_policy_value = run_VI(c_mu_pi())
    max_cost_policy_value = run_VI(max_cost_pi())

    simulate((1, 2, 3, 4), 2)

    """PI"""
    optimal_policy, initial_state_vals = PI(max_cost_policy_value)
    optimal_policy_value = run_VI(optimal_policy)
    # plt.plot(initial_state_vals, '-o', linewidth=2,
    #          label='Initial state value')
    # plt.title('Initial state value per Policy Iteration')
    # plt.ylabel('Initial state value')
    # plt.xlabel('Iteration')
    # plt.legend()
    # plt.show()

    """VI"""
    x = np.arange(len(optimal_policy_value))
    width = 0.35
    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width / 2, list(c_mu_policy_value.values()), 0.35, label='C Mu Policy')
    bar2 = ax.bar(x + width / 2, list(max_cost_policy_value.values()), 0.35, label='Max Cost Policy')
    ax.set_ylabel('Value')
    ax.set_title('State Values')
    ax.set_xticks(x)
    ax.set_xticklabels(list(max_cost_policy_value.keys()), rotation='vertical')
    ax.legend()
    fig.tight_layout()
    plt.show()