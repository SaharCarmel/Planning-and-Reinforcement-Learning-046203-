transition_cost = [[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2],
                   [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]]
action_dict = {'B': 0, 'K': 1, 'O': 2, '-': 3, }

def get_best_sequences():
    max_value_2_step = {'B': None, 'K': None, 'O': None}
    max_value_step = {'B': None, 'K': None, 'O': None}
    max_value_last_step = {'B': None, 'K': None, 'O': None}
    transition_dicts = [max_value_step, max_value_2_step, max_value_last_step]
    transition_1, transition_2 = 1, 1
    for i in range(3):
        update_dict = transition_dicts[i]
        for state in ['B', 'K', 'O']:
            max = 0
            for action in ['B', 'K', 'O']:
                transition_1 = transition_cost[action_dict[state]][action_dict[action]]
                if i == 1:
                    transition_2 = transition_cost[action_dict[action]][action_dict[max_value_step[action]]]
                elif i == 2:
                    transition_2 = transition_cost[action_dict[action]][action_dict['-']]
                value = transition_1 * transition_2
                if value > max:
                    max = value
                    update_dict[state] = action
    return max_value_2_step, max_value_last_step

"""calculate"""
double_step_dict, last_step_dict = get_best_sequences()
"""run"""
k = 5
current_state = 'B'
print(current_state, end='')
for x in range(k - 2):
    current_state = double_step_dict[current_state]
    print(current_state, end='')
print(last_step_dict[current_state])

