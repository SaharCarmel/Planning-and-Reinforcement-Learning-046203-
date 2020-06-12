import copy
import random

import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np


def run_VI(policy):
    V = {x: 0 for x in S}
    for iteration in range(VI_iterations):
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


def TD_0(pi, V, real_V, step_size):
    visits = {x: 0 for x in S}
    max_errs = []
    initial_state_errs = []
    for i in range(TD_0_iterations):
        """sample random state"""
        state = S[random.randint(1, len(S) - 1)]
        visits[state] += 1
        """use policy action for this state"""
        action = pi[state]
        cost, next_state = simulate(state, action)
        """step size"""
        if not step_size:
            alpha = 1 / visits[state]
        elif step_size == 1:
            alpha = 0.01
        else:
            alpha = 10 / (100 + visits[state])
        """TD 0 update"""
        V[state] = V[state] + alpha * (cost + V[next_state] - V[state])
        """error per iteration calculation"""
        max_errs.append(max([real_V[s] - V[s] for s in S]))
        initial_state_errs.append(real_V[(1, 2, 3, 4, 5)] - V[(1, 2, 3, 4, 5)])

    return V, max_errs, initial_state_errs


def TD_lambda(pi, V, real_V, step_size, lmbda):
    visits = {x: 0 for x in S}
    max_errs = []
    initial_state_errs = []
    for i in range(TD_0_iterations):
        """sample random state"""
        state = S[random.randint(1, len(S) - 1)]
        visits[state] += 1
        """use policy action for this state"""
        action = pi[state]
        cost, next_state = simulate(state, action)
        """step size"""
        if not step_size:
            alpha = 1 / visits[state]
        elif step_size == 1:
            alpha = 0.01
        else:
            alpha = 10 / (100 + visits[state])
        """TD lambda update"""
        current_state = state
        d, i = 0, 0
        while next_state != S[0]:
            step_cost = lmbda ** i * (cost + V[next_state] - V[current_state])
            d += step_cost
            if step_cost < epsilon:
                break
            current_state = next_state
            action = pi[current_state]
            cost, next_state = simulate(current_state, action)
            i += 1

        V[state] = V[state] + alpha * d

        """error per iteration calculation"""
        max_errs.append(max([real_V[s] - V[s] for s in S]))
        initial_state_errs.append(real_V[(1, 2, 3, 4, 5)] - V[(1, 2, 3, 4, 5)])

    return V, max_errs, initial_state_errs


def simulate(s, a):
    cost = reward(s)
    success = 1 if random.random() < Mu[a] else 0
    next_state = tuple(set(s).difference([a]) or [0]) if success else s
    return cost, next_state


def pt_c():
    values = run_VI(max_cost_pi())
    x = np.arange(len(values))
    fig, ax = plt.subplots()
    ax.bar(x, list(values.values()), label='Max Cost Policy')
    ax.set_ylabel('Value')
    ax.set_title('State Values')
    ax.set_xticks(x)
    ax.set_xticklabels(list(values.keys()), rotation='vertical')
    ax.legend()
    fig.tight_layout()
    plt.show()


def pt_d():
    optimal_policy, initial_state_vals = PI(max_cost_policy_value)
    plt.plot(initial_state_vals, '-o', linewidth=2,
             label='Initial state value')
    plt.title('Initial state value per Policy Iteration')
    plt.ylabel('Initial state value')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()


def pt_e():
    x = np.arange(len(optimal_policy_value))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, list(c_mu_policy_value.values()), 0.35, label='C Mu Policy')
    ax.bar(x + width / 2, list(max_cost_policy_value.values()), 0.35, label='Max Cost Policy')
    ax.set_ylabel('Value')
    ax.set_title('State Values')
    ax.set_xticks(x)
    ax.set_xticklabels(list(max_cost_policy_value.keys()), rotation='vertical')
    ax.legend()
    fig.tight_layout()
    plt.show()


def pt_g():
    policy = max_cost_pi()
    real_vals = run_VI(policy)
    V_approxs, max_errs_lists, initial_state_errs_lists = [], [], []
    for i in range(3):
        initial_V = {x: 0 for x in S}
        V_approx_i, max_errs_i, initial_state_errs_i = TD_0(policy, initial_V, real_vals, i)
        V_approxs.append(V_approx_i)
        max_errs_lists.append(max_errs_i)
        initial_state_errs_lists.append(initial_state_errs_i)

    fig1, ax1 = plt.subplots()
    x = np.arange(len(max_errs_lists[0]))
    ax1.plot(x, max_errs_lists[0], label='alpha = 1/n_visitst')
    ax1.plot(x, max_errs_lists[1], label='alpha = 0.01')
    ax1.plot(x, max_errs_lists[2], label='alpha = 10/(100+n_visits)')
    ax1.set_ylabel('Max State Value Error')
    ax1.set_title('TD_0 Max State Value Error by Iteration')
    ax1.legend()
    fig1.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    x = np.arange(len(max_errs_lists[0]))
    ax2.plot(x, initial_state_errs_lists[0], label='alpha = 1/n_visitst')
    ax2.plot(x, initial_state_errs_lists[1], label='alpha = 0.01')
    ax2.plot(x, initial_state_errs_lists[2], label='alpha = 10/(100+n_visits)')
    ax2.set_ylabel('Initial State Value Error')
    ax2.set_title('TD_0 Initial State Value Error by Iteration')
    ax2.legend()
    fig2.tight_layout()
    plt.show()


def pt_h():
    policy = max_cost_pi()
    real_vals = run_VI(policy)
    max_errs_lists, initial_state_errs_lists = [], []
    lambdas = [0.1, 0.5, 0.9, 1]
    for l in lambdas:
        max_err, initial_state_err = np.zeros(TD_0_iterations), np.zeros(TD_0_iterations)
        for i in range(20):
            initial_V = {x: 0 for x in S}
            V_approx_i, max_err_i, initial_state_err_i = TD_lambda(policy, initial_V, real_vals, 2, l)
            max_err += np.array(max_err_i)
            initial_state_err += np.array(initial_state_err_i)
        max_errs_lists.append(max_err / 20)
        initial_state_errs_lists.append(initial_state_err / 20)

    fig1, ax1 = plt.subplots()
    x = np.arange(len(max_errs_lists[0]))
    ax1.plot(x, max_errs_lists[0], label='lambda = ' + str(lambdas[0]))
    ax1.plot(x, max_errs_lists[1], label='lambda = ' + str(lambdas[1]))
    ax1.plot(x, max_errs_lists[2], label='lambda = ' + str(lambdas[2]))
    ax1.plot(x, max_errs_lists[3], label='lambda = ' + str(lambdas[3]))
    ax1.set_ylabel('Max State Value Error')
    ax1.set_title('TD_lambda Max State Value Error by Iteration')
    ax1.legend()
    fig1.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    x = np.arange(len(max_errs_lists[0]))
    ax2.plot(x, initial_state_errs_lists[0], label='lambda = ' + str(lambdas[0]))
    ax2.plot(x, initial_state_errs_lists[1], label='lambda = ' + str(lambdas[1]))
    ax2.plot(x, initial_state_errs_lists[2], label='lambda = ' + str(lambdas[2]))
    ax2.plot(x, initial_state_errs_lists[3], label='lambda = ' + str(lambdas[3]))
    ax2.set_ylabel('Initial State Value Error')
    ax2.set_title('TD_lambda Initial State Value Error by Iteration')
    ax2.legend()
    fig2.tight_layout()
    plt.show()


def greedy_exploration(Q_vals, state, e):
    if random.random() <= Q_learning_epsilon[e]:
        action = random.choice(list(state))
    else:  # greedy
        sorted_Qs = sorted([[Q_vals[(state, a)], a] for a in state])
        action = sorted_Qs[0][1]

    return state, action


def get_Q_greedy_policy_values(Q_vals):
    V = {x:float('inf') for x in S}
    for q in Q_vals:
        if V[q[0]] > Q_vals[q]:
            V[q[0]] = Q_vals[q]
    return V


def Q_learning(Q, optimal_V, step_size, eps_n):
    max_errs = []
    visits = copy.copy(Q)
    initial_state_errs = []
    for i in range(TD_0_iterations):
        """sample random state"""
        state = S[random.randint(1, len(S) - 1)]
        """random action for this state"""
        action = random.choice(list(state))
        visits[state, action] += 1
        cost, next_state = simulate(state, action)
        """step size"""
        if not step_size:
            alpha = 1 / visits[state, action]
        elif step_size == 1:
            alpha = 0.01
        else:
            alpha = 10 / (100 + visits[state, action])
        """get next state Q"""
        next_Q = greedy_exploration(Q, next_state, eps_n)
        """Q update"""
        Q[state, action] = Q[state, action] + alpha * (cost + Q[next_Q] - Q[state, action])

        """error per iteration calculation"""
        greedy_vals = get_Q_greedy_policy_values(Q)
        max_errs.append(max([optimal_V[s] - greedy_vals[s] for s in S]))
        initial_state_errs.append(optimal_V[(1, 2, 3, 4, 5)] - greedy_vals[(1, 2, 3, 4, 5)])

    return Q, max_errs, initial_state_errs


def pt_i():
    initial_Q = {}
    for s in S:
        for a in s:
            initial_Q[s, a] = 0
    optimal_policy = c_mu_pi()
    optimal_policy_vals = run_VI(optimal_policy)
    Q_learned, max_errs_lists, initial_state_errs_lists = [], [], []
    for i in range(3):
        empty_Q = copy.copy(initial_Q)
        Q_i, max_errs_i, initial_state_errs_i = Q_learning(empty_Q, optimal_policy_vals, i, eps_n=0)

        Q_learned.append(Q_i)
        max_errs_lists.append(max_errs_i)
        initial_state_errs_lists.append(initial_state_errs_i)

    fig1, ax1 = plt.subplots()
    x = np.arange(len(max_errs_lists[0]))
    ax1.plot(x, max_errs_lists[0], label='alpha = 1/n_visitst')
    ax1.plot(x, max_errs_lists[1], label='alpha = 0.01')
    ax1.plot(x, max_errs_lists[2], label='alpha = 10/(100+n_visits)')
    ax1.set_ylabel('Max State Value Error')
    ax1.set_title('Q-Learning Max State Value Error by Iteration')
    ax1.legend()
    fig1.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    x = np.arange(len(max_errs_lists[0]))
    ax2.plot(x, initial_state_errs_lists[0], label='alpha = 1/n_visit')
    ax2.plot(x, initial_state_errs_lists[1], label='alpha = 0.01')
    ax2.plot(x, initial_state_errs_lists[2], label='alpha = 10/(100+n_visits)')
    ax2.set_ylabel('Initial State Value Error')
    ax2.set_title('Q-Learning Initial State Value Error by Iteration')
    ax2.legend()
    fig2.tight_layout()
    plt.show()


def pt_j():
    initial_Q = {}
    for s in S:
        for a in s:
            initial_Q[s, a] = 0
    optimal_policy = c_mu_pi()
    optimal_policy_vals = run_VI(optimal_policy)
    Q_learned, max_errs_lists, initial_state_errs_lists = [], [], []
    for i in range(2):
        empty_Q = copy.copy(initial_Q)
        Q_i, max_errs_i, initial_state_errs_i = Q_learning(empty_Q, optimal_policy_vals, step_size=2, eps_n=i)
        Q_learned.append(Q_i)
        max_errs_lists.append(max_errs_i)
        initial_state_errs_lists.append(initial_state_errs_i)

    fig1, ax1 = plt.subplots()
    x = np.arange(len(max_errs_lists[0]))
    ax1.plot(x, max_errs_lists[0], label='epsilon = 0.1')
    ax1.plot(x, max_errs_lists[1], label='epsilon = 0.01')
    ax1.set_ylabel('Max State Value Error')
    ax1.set_title('Q-Learning Max State Value Error by Iteration with alpha = 10/(100+n_visits)')
    ax1.legend()
    fig1.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots()
    x = np.arange(len(max_errs_lists[0]))
    ax2.plot(x, initial_state_errs_lists[0], label='epsilon = 0.1')
    ax2.plot(x, initial_state_errs_lists[1], label='epsilon = 0.01')
    ax2.set_ylabel('Initial State Value Error')
    ax2.set_title('Q-Learning Initial State Value Error by Iteration with alpha = 10/(100+n_visits)')
    ax2.legend()
    fig2.tight_layout()
    plt.show()


"""Globals"""
Mu = [0, 0.6, 0.5, 0.3, 0.7, 0.1]
C = [0, 1, 4, 6, 2, 9]
S = [(0,)]
for i in range(5):
    S += list(combinations([1, 2, 3, 4, 5], i + 1))
VI_iterations = 250
TD_0_iterations = 10000
epsilon = 1e-6
Q_learning_epsilon = [0.1, 0.01]

if __name__ == '__main__':
    c_mu_policy_value = run_VI(c_mu_pi())
    max_cost_policy_value = run_VI(max_cost_pi())
    optimal_policy, initial_state_vals = PI(max_cost_pi())
    optimal_policy_value = run_VI(optimal_policy)

    """PART 1"""
    # pt_c()
    # pt_d()
    # pt_e()
    """PART 2"""
    # pt_g()
    # pt_h()
    # pt_i()
    pt_j()

    # c_mu_policy_value = run_VI(c_mu_pi())
    # max_cost_policy_value = run_VI(max_cost_pi())
    #
    # simulate((1, 2, 3, 4), 2)
