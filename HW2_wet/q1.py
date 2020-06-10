import numpy as np


def reward(V, x, y, action):
    if action == 'stick':
        if x < 17:
            return d_p[y][22] - sum(d_p[y][17:22])
        else:  # x>=17
            return d_p[y][22] + sum(d_p[y][:x]) - sum(d_p[y][x + 1:22])


    else:  # "hit
        sm = 0
        for i in range(2, 12):
            state_value = V[x + i][y] if x + i < 22 else -1
            sm += c_p[i] * state_value

        return sm


def subset_sum(target, sm, i=0, partial=[]):
    s = sum(partial)

    # check if the partial sum is equals to target
    if s == target:
        sm += [partial]
    if s >= target:
        return  # if we reach the number why bother to continue

    for i in range(i, len(cards)):
        n = cards[i]
        subset_sum(target, sm, i + 1, partial + [n])


def get_dp(sum_val):
    card_combinations = []
    subset_sum(sum_val, card_combinations)
    overall_probability = 0
    for combination in card_combinations:
        combination_probability = 1
        for card in combination:
            combination_probability *= c_p[card]
        overall_probability += combination_probability

    return overall_probability


def dealer_probabilities():
    DP = np.zeros([12, 23])
    for current_val in range(2, 12):
        for goal_val in range(17, 22):
            DP[current_val][goal_val] = get_dp(goal_val - current_val)
        DP[current_val][22] = np.prod(1 - DP[current_val][17:22])
    return DP


def value_iteration(n_iterations):
    state_values = np.zeros([23, 23])
    state_values[22][:] = -1
    for v in state_values:
        v[22] = 1

    for iteration in range(n_iterations):

        for player in range(4, 23):
            for dealer in range(2, 12):
                hit = reward(state_values, player, dealer, 'hit')
                stick = reward(state_values, player, dealer, 'stick')
                state_values[player][dealer] = max(hit, stick)
        print()
    return state_values


"""RUN"""
cards = list(range(2, 12))
c_p = np.ones(12) / 13
c_p[0] = 0
c_p[1] = 0
c_p[10] *= 4
d_p = dealer_probabilities()
value_iteration(10)
