transition_cost = [[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2],
                   [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]]
action_dict = {'B': 0, 'K': 1, 'O': 2, '-': 3, }

def best_ending():
    max_value_ending = {'B': None, 'K': None, 'O': None}
    for state in ['B', 'K', 'O']:
        max = 0
        for action in ['B', 'K', 'O']:
            trans_1 = transition_cost[action_dict[state]][action_dict[action]]
            trans_2 = transition_cost[action_dict[action]][action_dict['-']]
            value = trans_1 * trans_2
            if value > max:
                max = value
                max_value_ending[state] = action
    return max_value_ending

def best_2_steps():
    max_value_2_step = {'B': None, 'K': None, 'O': None}
    max_value_step = {'B': None, 'K': None, 'O': None}
    transition_dicts = [max_value_step, max_value_2_step]
    transition_1, transition_2 = 1, 1
    for i in range(2):
        update_dict = transition_dicts[i]
        for state in ['B', 'K', 'O']:
            max = 0
            for action in ['B', 'K', 'O']:
                transition_1 = transition_cost[action_dict[state]][
                    action_dict[action]]
                if i == 1:
                    transition_2 = transition_cost[action_dict[action]][
                        action_dict[max_value_step[action]]]
                value = transition_1 * transition_2
                if value > max:
                    max = value
                    update_dict[state] = action
    return max_value_2_step

"""calculate"""
best_ending_dict = best_ending()
double_step_dict = best_2_steps()
"""run"""
k = 5
current_state = 'B'
print(current_state, end='')
for x in range(k - 2):
    current_state = double_step_dict[current_state]
    print(current_state, end='')
print(best_ending_dict[current_state])

