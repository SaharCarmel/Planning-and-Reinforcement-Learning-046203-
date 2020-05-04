def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    current_state = goal_state
    while prev[current_state.to_string()]:
        prev_state = prev[current_state.to_string()]
        to_cord = current_state._get_empty_location()
        from_cord = prev_state._get_empty_location()
        if to_cord[1] - from_cord[1]:
            action = 'l' if (to_cord[1] - from_cord[1]) < 0 else 'r'
        else:
            action = 'u' if (to_cord[0] - from_cord[0]) < 0 else 'd'
        result.insert(0, (prev_state, action))
        current_state = prev_state
    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan) - 1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))

# plan[0][0].to_string() ==plan[1][0].to_string()
