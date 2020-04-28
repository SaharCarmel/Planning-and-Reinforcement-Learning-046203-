import numpy as np
from cartpole_cont import CartPoleContEnv


def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau
    I = np.identity(4)
    A_hat = np.array([[0, 1, 0, 0], [0, 0, (pole_mass / cart_mass) * g, 0], [0, 0, 0, 1],
                      [0, 0, (g / pole_length) * (1 + (pole_mass / cart_mass)), 0]])
    A = A_hat * dt + I
    return A


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau
    B_hat = np.array([[0], [1 / cart_mass], [0], [1 / (cart_mass * pole_length)]])
    B = B_hat * dt
    return B


def find_lqr_control_input(cart_pole_env):
    '''
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action of shape (1,4).
    '''
    assert isinstance(cart_pole_env, CartPoleContEnv)

    # TODO - you first need to compute A and B for LQR
    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)

    # TODO - Q and R should not be zero, find values that work, hint: all the values can be <= 1.0
    # the state cost - add larger values where you want there to be more punishment for errors
    # in this case we want to punish for errors in location and in angle (x[0],x[2])
    Q = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ])

    # the control cost
    R = np.array([1])

    # TODO - you need to compute these matrices in your solution, but these are not returned.
    Ps = []

    # TODO - these should be returned see documentation above
    us = []
    xs = [np.expand_dims(cart_pole_env.state, 1)]
    Ks = []

    """LQR"""
    A_T = A.transpose()
    B_T = B.transpose()
    Q_f = Q
    P_0 = Q_f
    Ps.append(P_0)

    for i in range(1, cart_pole_env.planning_steps + 1):
        # Previous P
        P_minus_1 = Ps[i - 1]

        # calculate K
        a = np.linalg.multi_dot([B_T, P_minus_1, B])
        b = np.add(R, a).transpose()
        K = - np.linalg.multi_dot([b, B_T, P_minus_1, A])
        # K = - (R + np.linalg.multi_dot([B_T, P_minus_1, B])).transpose() * B_T * P_minus_1 * A
        # K_T = K.transpose()
        Ks.append(K)

        # Calculate new P
        c = np.linalg.multi_dot([A_T, P_minus_1, A])
        d = np.linalg.multi_dot([A_T, P_minus_1, B, b, B_T, P_minus_1, A])
        P = np.add(Q, c, -d)

        # pt_3 = np.add(A, B.dot(K))
        # P = np.add(Q, np.linalg.multi_dot([K_T, R, K]), np.linalg.multi_dot([pt_3.transpose(), P_minus_1, pt_3]))
        # P = Q + K_T*R*K +(A + B*K).transpose()*P_minus_1*(A + B*K)
        Ps.append(P)

        # control step u
        u = K.dot(cart_pole_env.state)
        u = np.expand_dims(u, 1)
        us.append(u)

        # state step x
        x_plus_1 = np.add(A.dot(xs[i - 1]), B.dot(u))
        xs.append(x_plus_1)

    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


if __name__ == '__main__':
    env = CartPoleContEnv(initial_theta=np.pi * 0.1)
    # the following is an example to start at a different theta
    # env = CartPoleContEnv(initial_theta=np.pi * 0.25)

    # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))

    # start a new episode
    actual_state = env.reset()
    env.render()
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)
    # run the episode until termination, and print the difference between planned and actual
    is_done = False
    iteration = 0
    is_stable_all = []
    while not is_done:
        # print the differences between planning and execution time
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
        print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        # apply action according to actual state visited
        # make action in range
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
    valid_episode = np.all(is_stable_all[-100:])
    # print if LQR succeeded
    print('valid episode: {}'.format(valid_episode))
