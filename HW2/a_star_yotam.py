from puzzle import *
from planning_utils_yotam import *
import heapq
import datetime


def a_star(puzzle):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # this is the heuristic function for of the start state
    initial_to_goal_heuristic = initial.get_manhattan_distance(goal)

    # the fringe is the queue to pop items from
    fringe = [(initial_to_goal_heuristic, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}
    expanded = 0
    concluded.add(initial.to_string())
    while len(fringe) > 0:
        # remove the following line and complete the algorithm
        current_priority, current_state = heapq.heappop(fringe)
        possible_actions = current_state.get_actions()
        for a in possible_actions:
            new_state = current_state.apply_action(a)
            new_heuristic = new_state.get_manhattan_distance(goal)
            new_priority = new_heuristic + distances[current_state.to_string()] + 1

            #TODO not working properly
            if new_state.to_string() in concluded:
                # if new_priority < distances[new_state.to_string()] + new_heuristic:
                #     prev[new_state.to_string()] = current_state
                #     expanded += 1
                # else:
                print("ignored {} with priority {}".format(new_state, new_priority))
            else:
                expanded += 1
                heapq.heappush(fringe, (new_priority, new_state))
                distances[new_state.to_string()] = distances[current_state.to_string()] + 1
                prev[new_state.to_string()] = current_state
                concluded.add(new_state.to_string())
                print("did not ignore {} with priority {}".format(new_state, new_priority))


            if new_state.to_string() == goal.to_string():
                fringe = []
                break
    print('expanded:',expanded)
    return prev

def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = a_star(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)

    """hard state for Q2_pt4"""
    # goal_state = State(s='8 6 7\n2 5 4\n3 0 1')

    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))
