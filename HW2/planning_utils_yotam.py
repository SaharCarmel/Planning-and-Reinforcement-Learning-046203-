def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    prev_state, prev_action = prev[goal_state.to_string()]
    while prev[prev_state.to_string()]:
        result.insert(0, (prev_state, prev_action))
        prev_state, prev_action = prev[prev_state.to_string()]
    result.insert(0, (prev_state, prev_action))

    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan) - 1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))

# plan[0][0].to_string() ==plan[1][0].to_string()
