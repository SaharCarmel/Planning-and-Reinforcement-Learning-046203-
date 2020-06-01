import numpy as np
from cartpole_cont import CartPoleContEnv
import matplotlib.pyplot as plt


def get_A(cart_pole_env, theta, d_theta, F):
    g = cart_pole_env.gravity
    m = cart_pole_env.masspole
    M = cart_pole_env.masscart
    l = cart_pole_env.length
    dt = cart_pole_env.tau

    """system dynamics"""
    a = - (m * (g - (2 * g * np.cos(theta) ** 2) + (l * np.cos(theta) * d_theta ** 2))) / (
            M + m - m * np.cos(theta) ** 2) - (2 * m * np.cos(theta) * np.sin(theta) * (
            - l * m * np.sin(theta) * d_theta ** 2 + F + g * m * np.cos(theta) * np.sin(theta))) / (
                - m * np.cos(theta) ** 2 + M + m) ** 2
    b = -(2 * d_theta * l * m * np.sin(theta)) / (m * np.sin(theta) ** 2 + M)
    c = (g * m * np.cos(theta) - F * np.sin(theta) + M * g * np.cos(theta) - d_theta ** 2 * l * m * np.cos(
        theta) ** 2 + d_theta ** 2 * l * m * np.sin(theta) ** 2) / (l * (M + m - m * np.cos(theta) ** 2)) - (
                2 * m * np.cos(theta) * np.sin(theta) * (
                - l * m * np.cos(theta) * np.sin(theta) * d_theta ** 2 + F * np.cos(theta) + g * m * np.sin(
            theta) + M * g * np.sin(theta))) / (l * (- m * np.cos(theta) ** 2 + M + m) ** 2)
    d = -(2 * d_theta * m * np.cos(theta) * np.sin(theta)) / (M + m - m * np.cos(theta) ** 2)
    A_hat = np.array([[0, 1, 0, 0], [0, 0, a, b], [0, 0, 0, 1], [0, 0, c, d]])
    A = A_hat * dt + np.identity(4)
    return A


def get_B(cart_pole_env, theta):
    m = cart_pole_env.masspole
    M = cart_pole_env.masscart
    l = cart_pole_env.length
    dt = cart_pole_env.tau
    B_hat = np.array([[0], [1 / (M + m - m * np.cos(theta) ** 2)],
                      [0], [np.cos(theta) / (l * (M + m - m * np.cos(theta) ** 2))]])
    B = B_hat * dt
    return B


def cost_function(x, u, final=False):
    w1 = 1
    w2 = 1
    w3 = 1
    R = np.array([w3])
    Q = np.array(
        [w1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, w2, 0],
        [0, 0, 0, 0])
    """additional parameters"""
    H = np.zeros([1, 4])
    q = np.zeros([1, 4]).T
    r = np.zeros([1, 4]).T

    if not final:
        """cost variables"""
        l_x = Q @ x + q
        l_u = R @ u + r
        l_xx = Q
        l_uu = R
        l_xu = H
        l_ux = H.T
        l = 0.5 * (x.T @ Q @ x + u.T @ R @ u + x.T @ H.T @ u + u @ H @ x)
    else:
        l_xx = Q
        l_x = Q @ x
        l = x.T @ Q @ x
        l_u = None
        l_uu = None
        l_xu = None
        l_ux = None

    return l, l_x, l_xx, l_u, l_uu, l_xu, l_ux


def forward_simulation(x, u, planning_steps):

    for i in range(planning_steps):



def find_ilqr_control_input(cart_pole_env, ilqr_iterations = 100):
    assert isinstance(cart_pole_env, CartPoleContEnv)
    actual_state = cart_pole_env.reset()
    cart_pole_env.render()
    steps = cart_pole_env.planning_steps
    u_current = [0] * steps
    x_init = (actual_state,)

    for i in range(ilqr_iterations):
        x_current, cost_current = forward_simulation(list(x_init), u_current, steps)

        for i in range(steps):
            A[i] = get_A(cart_pole_env, x_current[i].item(2), x_current[i].item(3), u_current[i])
            B[i] = get_B(cart_pole_env, x_current[i].item(2))
            l[i], lx[i], lu[i], lxx[i], luu[i], lux[i], lux[i] = cost_function(x_current[i], u_current[i])
            # cost for each step will be accumulated across all steps eventually:
            lx[i] *= env.tau
            lxx[i] *= env.tau
            lu[i] *= env.tau
            luu[i] *= env.tau
            lux[i] *= env.tau


    """Parameters"""
    # A = get_A(cart_pole_env, angle, delta_angle, force)
    # B = get_B(cart_pole_env, angle)
    # w1 = 1
    # w2 = 1
    # w3 = 1
    # R = np.array([w3])
    # Q = np.array([
    #     [w1, 0, 0, 0],
    #     [0, 0, 0, 0],
    #     [0, 0, w2, 0],
    #     [0, 0, 0, 0]
    # ])
    # Q_f = Q
    # P_0 = Q_f
    # Ps = [P_0]
    # us = []
    # xs = [np.expand_dims(cart_pole_env.state, 1)]
    # Ks = []

    """Run LQR"""
    for i in range(1, cart_pole_env.planning_steps + 1):
        P_plus_1 = Ps[i - 1]
        P = Q + A.T @ P_plus_1 @ A - \
            A.T @ P_plus_1 @ B @ (np.linalg.inv(R + B.T @ P_plus_1 @ B)) @ B.T @ P_plus_1 @ A
        K = - np.linalg.inv(R + B.T @ P_plus_1 @ B) @ B.T @ P_plus_1 @ A
        u = K @ xs[i - 1]
        x_plus_1 = A @ xs[i - 1] + B @ u
        # Append
        Ps.append(P)
        Ks.append(K)
        us.append(u)
        xs.append(x_plus_1)

    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    xs.reverse(), us.reverse(), Ks.reverse()  # TODO reversed or not?
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


def ilqr():
    env = CartPoleContEnv(initial_theta=np.pi)
    actual_state = env.reset()
    u_ref = [0] * env.planning_steps
    x_ref = [actual_state]
    """run forward simulation to obtain x_ref"""
    for actual_action in u_ref:
        actual_state, reward, is_done, _ = env.step(actual_action)
        x_ref.append(actual_state)

    """linearilize f around the ref"""
    A_ref = get_A(env, angle, delta_angle, force)
    B_ref = get_B(env)
    """quadritize c"""

    """LQR"""

    """simulate u*"""

    """get new reference"""


def main():
    env = CartPoleContEnv(initial_theta=np.pi)
    xs, us, Ks = find_ilqr_control_input(env)
    is_done = False
    iteration = 0
    is_stable_all = []
    while not is_done:
        predicted_theta = xs[iteration].item(2)
        actual_theta = actual_state[2]
        predicted_action = us[iteration].item(0)
        actual_action = (Ks[iteration] @ np.expand_dims(actual_state, 1)).item(0)
        print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
        actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
        actual_action = np.array([actual_action])
        actual_state, reward, is_done, _ = env.step(actual_action)
        is_stable = reward == 1.0
        is_stable_all.append(is_stable)
        env.render()
        iteration += 1
    env.close()
    valid_episode = np.all(is_stable_all[-100:])
    print('valid episode: {}'.format(valid_episode))


if __name__ == '__main__':
    main()
