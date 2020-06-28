import numpy as np
import time
import matplotlib.pyplot as plt
from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor


class Solver:
    def __init__(self, number_of_kernels_per_dim, number_of_actions, gamma, learning_rate):
        # Set max value for normalization of inputs
        self._max_normal = 1
        # get state \action information
        self.data_transformer = DataTransformer()
        state_mean = [-3.00283763e-01, 5.61618575e-05]
        state_std = [0.51981243, 0.04024895]
        self.data_transformer.set(state_mean, state_std)
        self._actions = number_of_actions
        # create RBF features:
        self.feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        self.number_of_features = self.feature_extractor.get_number_of_features()
        # the weights of the q learner
        self.theta = np.random.uniform(-0.001, 0, size=number_of_actions * self.number_of_features)
        # discount factor for the solver
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _normalize_state(self, s):
        return self.data_transformer.transform_states(np.array([s]))[0]

    def get_features(self, state):
        normalized_state = self._normalize_state(state)
        features = self.feature_extractor.encode_states_with_radial_basis_functions([normalized_state])[0]
        return features

    def get_q_val(self, features, action):
        theta_ = self.theta[action * self.number_of_features: (1 + action) * self.number_of_features]
        return np.dot(features, theta_)

    def get_all_q_vals(self, features):
        all_vals = np.zeros(self._actions)
        for a in range(self._actions):
            all_vals[a] = solver.get_q_val(features, a)
        return all_vals

    def get_max_action(self, state):
        sparse_features = solver.get_features(state)
        q_vals = solver.get_all_q_vals(sparse_features)
        return np.argmax(q_vals)

    def get_state_action_features(self, state, action):
        state_features = self.get_features(state)
        all_features = np.zeros(len(state_features) * self._actions)
        all_features[action * len(state_features): (1 + action) * len(state_features)] = state_features
        return all_features

    def update_theta(self, state, action, reward, next_state, done):
        # compute the new weights and set in self.theta. also return the bellman error (for tracking).
        if done:
            return 0
        else:
            features = self.get_state_action_features(state, action)
            state_features = self.get_features(state)
            q_current_state = self.get_q_val(state_features, action)
            max_action = self.get_max_action(next_state)
            q_next_state = self.get_q_val(state_features, max_action)
            bellman_error = reward + gamma * q_next_state - q_current_state
            self.theta = self.theta + self.learning_rate * bellman_error * features
        return bellman_error


def modify_reward(reward):
    reward -= 1
    if reward == 0.:
        reward = 100.
    return reward


def run_episode(env, solver, is_train=True, epsilon=None, max_steps=200, render=False):
    episode_gain = 0
    deltas = []
    if is_train:
        start_position = -0.5
        start_velocity = np.random.uniform(-env.max_speed, env.max_speed)
    else:
        start_position = -0.5
        start_velocity = np.random.uniform(-env.max_speed / 100., env.max_speed / 100.)
    state = env.reset_specific(start_position, start_velocity)
    step = 0
    if render:
        env.render()
        time.sleep(0.1)
    while True:
        if epsilon is not None and np.random.uniform() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = solver.get_max_action(state)
        if render:
            env.render()
            time.sleep(0.1)
        next_state, reward, done, _ = env.step(action)
        reward = modify_reward(reward)
        step += 1
        episode_gain += reward
        if is_train:
            deltas.append(solver.update_theta(state, action, reward, next_state, done))
        if done or step == max_steps:
            return episode_gain, np.mean(deltas)
        state = next_state


if __name__ == "__main__":

    """seeds"""
    # seeds = [123, 234, 345]
    # epsilons = [0.1]
    """epsilons"""
    seeds = [123]
    # epsilons = [1.0, 0.75, 0.5, 0.3, 0.01]
    epsilons = [0.1]

    gamma = 0.999
    learning_rate = 0.05
    epsilon_current = 0.1
    epsilon_decrease = 1.
    epsilon_min = 0.01

    max_episodes = 10000

    seed_rewards, seed_performance, seed_bottom_val, seed_bellman_err_avg = [], [], [], []
    for seed in seeds:
        env = MountainCarWithResetEnv()
        np.random.seed(seed)
        env.seed(seed)

        solver = Solver(
            # learning parameters
            gamma=gamma, learning_rate=learning_rate,
            # feature extraction parameters
            number_of_kernels_per_dim=[5, 7],
            # env dependencies (DO NOT CHANGE):
            number_of_actions=env.action_space.n,
        )

        for epsilon_current in epsilons:
            rewards, performance, bottom_val, bellman_err_avg, bellman_err = [], [], [], [], []
            for episode_index in range(0, max_episodes):
                episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon_current)
                rewards.append(episode_gain)
                bellman_err.append(mean_delta)
                if episode_index < 100:
                    bellman_err_avg.append(np.mean(bellman_err))
                else:
                    bellman_err_avg.append(np.mean(bellman_err[episode_index-99:episode_index+1]))
                bottom_state = [0, 0]
                bottom_val.append(solver.get_q_val(solver.get_features(bottom_state), solver.get_max_action(bottom_state)))

                # reduce epsilon if required
                epsilon_current *= epsilon_decrease
                epsilon_current = max(epsilon_current, epsilon_min)

                print(
                    f'Episode {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}, average error {mean_delta}')

                # termination condition:
                if episode_index % 10 == 9:
                    test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
                    mean_test_gain = np.mean(test_gains)
                    successes = [x>=-75 for x in test_gains]
                    performance.append(np.mean(successes))
                    print(f'tested 10 episodes: mean gain is {mean_test_gain}')
                    if mean_test_gain >= -75.:
                        print(f'solved in {episode_index} episodes')
                        break

            seed_rewards.append(rewards)
            seed_performance.append(performance)
            seed_bottom_val.append(bottom_val)
            seed_bellman_err_avg.append(bellman_err_avg)
        # run_episode(env, solver, is_train=False, render=True)

            if len(epsilons) > 1:
                fig = plt.figure()
                ax1 = fig.add_subplot()
                x = list(range(1, len(rewards) + 1))
                ax1.plot(x, rewards, label='epsilon =' + str(epsilon_current))
                ax1.set_xlabel('Episodes')
                ax1.set_ylabel('Total Rewards')
                ax1.set_title('Total Rewards by Episodes')
                ax1.legend()
                fig.tight_layout()
                plt.show()


    """plots"""
    if len(epsilons) > 1:
        fig = plt.figure()
        ax1 = fig.add_subplot()
        for i in range(len(epsilons)):
            x = list(range(1, len(seed_rewards[i]) + 1))
            x = [i*10 for i in x]
            ax1.plot(x, seed_rewards[i], label='epsilon =' + str(epsilons[i]))
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Total Rewards')
        ax1.set_title('Total Rewards by Episodes')
        ax1.legend()
        fig.tight_layout()
        plt.show()

    else:
        """plots"""
        fig = plt.figure()
        ax1 = fig.add_subplot()
        for i in range(len(seeds)):
            x = list(range(1, len(seed_rewards[i]) + 1))
            ax1.plot(x, seed_rewards[i], label='seed =' + str(seeds[i]))
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Total Rewards')
        ax1.set_title('Total Rewards by Episodes')
        ax1.legend()
        fig.tight_layout()
        plt.show()

        fig2, ax2 = plt.subplots()
        for i in range(len(seeds)):
            x = list(range(1, len(seed_performance[i]) + 1))
            ax2.plot(x, seed_performance[i], label='seed =' + str(seeds[i]))
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Performance')
        ax2.set_title('Performance by Episodes')
        ax2.legend()
        fig2.tight_layout()
        plt.show()

        fig3, ax3 = plt.subplots()
        for i in range(len(seeds)):
            x = list(range(1, len(seed_bottom_val[i]) + 1))
            ax3.plot(x, seed_bottom_val[i], label='seed =' + str(seeds[i]))
        ax3.set_xlabel('Episodes')
        ax3.set_ylabel('Bottom Hill Approx Value')
        ax3.set_title('Bottom Hill Approx Value by Episodes')
        ax3.legend()
        fig3.tight_layout()
        plt.show()

        fig4, ax4 = plt.subplots()
        for i in range(len(seeds)):
            x = list(range(1, len(seed_bellman_err_avg[i]) + 1))
            ax4.plot(x, seed_bellman_err_avg[i], label='seed =' + str(seeds[i]))
        ax4.set_xlabel('Episodes')
        ax4.set_ylabel('Bellman Error')
        ax4.set_title('Bellman Error by Episodes')
        ax4.legend()
        fig4.tight_layout()
        plt.show()
