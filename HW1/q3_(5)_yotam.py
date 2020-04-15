transition_cost = [[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]]
actions = {
    'B': 0,
    'K': 1,
    'O': 2,
    '-': 3,
}


def dp_sol(k):
    word = 'B'
    current_state = 'B'

    for i in range(k):
        

    get_best_action()

    return word


print(dp_sol(5))
