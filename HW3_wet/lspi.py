import numpy as np
import matplotlib.pyplot as plt
import json
from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer


def compute_lspi_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma):
    # compute the next w given the data.
    b = np.zeros(linear_policy.w.shape)
    A = np.zeros([linear_policy.w.shape[0], linear_policy.w.shape[0]])
    n_samples = len(encoded_states)
    current_state_values = linear_policy.get_q_features(encoded_states, actions)
    max_actions = linear_policy.get_max_action(encoded_next_states)
    next_state_values = linear_policy.get_q_features(encoded_next_states, max_actions)
    for i in range(n_samples):
        b += rewards[i] * np.expand_dims(current_state_values[i], 1)
        if not done_flags[i]:
            A += np.expand_dims(current_state_values[i], 1) * (
                    gamma * np.expand_dims(next_state_values[i], 0)
                    - np.expand_dims(current_state_values[i], 0))
    next_w = np.linalg.inv(A / n_samples) @ b / n_samples
    return next_w


if __name__ == '__main__':

    # seeds = [123, 111, 345]
    # samples = [100000]

    seeds = [123]
    samples = [150000, 100000, 50000]

    number_of_kernels_per_dim = [12, 10]
    gamma = 0.99
    w_updates = 20
    evaluation_number_of_games = 50
    evaluation_max_steps_per_game = 200

    seed_performance = []
    for seed in seeds:
        for samples_to_collect in samples:
            np.random.seed(seed)

            env = MountainCarWithResetEnv()
            # collect data
            states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
            # get data success rate
            data_success_rate = np.sum(rewards) / len(rewards)
            print(f'success rate {data_success_rate}')
            # standardize data
            data_transformer = DataTransformer()
            data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
            states = data_transformer.transform_states(states)
            next_states = data_transformer.transform_states(next_states)
            # process with radial basis functions
            feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
            # encode all states    encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
            encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
            encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
            # set a new linear policy
            linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
            # but set the weights as random
            linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
            # start an object that evaluates the success rate over time
            evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
            performance = []
            for lspi_iteration in range(w_updates):
                print(f'starting lspi iteration {lspi_iteration}')

                new_w = compute_lspi_iteration(
                    encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
                )
                norm_diff = linear_policy.set_w(new_w)
                performance.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))

                if norm_diff < 0.00001:
                    break
            print('done lspi')
            seed_performance.append(performance)


    """plot seeds"""
    # fig = plt.figure()
    # ax1 = fig.add_subplot()
    # for i in range(len(seeds)):
    #     x = list(range(1, len(seed_performance[i]) + 1))
    #     ax1.plot(x, seed_performance[i], label='seed =' + str(seeds[i]))
    # ax1.set_ylabel('Performance')
    # ax1.set_xlabel('w-updates')
    # ax1.legend()
    # fig.tight_layout()
    # plt.show()

    """plot samples"""
    fig = plt.figure()
    ax1 = fig.add_subplot()
    for i in range(len(samples)):
        x = list(range(1, len(seed_performance[i]) + 1))
        ax1.plot(x, seed_performance[i], label='# samples =' + str(samples[i]))
    ax1.set_ylabel('Performance')
    ax1.set_xlabel('w-updates')
    ax1.legend()
    fig.tight_layout()
    plt.show()

