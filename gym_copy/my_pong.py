import gym
import numpy as np

def downsample(observation):
    return observation[::2, ::2, :]
def remove_color(observation):
    return observation[:,:,0]
def remove_background(observation):
    observation[observation == 144] = 0
    observation[observation == 109] = 0
    return observation
def relu(matrix):
    matrix[matrix<0]=0
    return matrix
def sigmoid(matrix):
    return 1.0/(1.0+np.exp(-matrix))

def preprocess_observation(input_observation, prev_processed_observation, input_dimensions):
    processed_observation = input_observation[35:195]
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation!=0] = 1
    processed_observation = processed_observation.astype(np.float).ravel()

    if prev_processed_observation is not None:
        output_observation = processed_observation - prev_processed_observation
    else:
        output_observation = processed_observation - np.zeros(input_dimensions)

    prev_processed_observation = processed_observation

    return output_observation, prev_processed_observation

def apply_neural_network(observation, weigths):
    hidden_layer_values = np.dot(weigths['1'], observation)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(weigths['2'],hidden_layer_values)
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        return 2
    else:
        return 3

def discount_rewards(episode_rewards, gamma):
    discounted_rewards = np.zeros_like(episode_rewards)
    running_add = 0
    for t in reversed(xrange(0, episode_rewards.size)):
        if episode_rewards[t] !=0 :
            running_add = 0
        running_add = running_add * gamma + episode_rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] +=(learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name])
    return weights, expectation_g_squared, g_dict

def main():
    env = gym.make("Pong-v0")
    observation = env.reset()
    batch_size = 10
    gamma = 0.99
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80*80
    learning_rate = 1e-4


    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observation = None

    weigths = {
        '1':np.random.randn(num_hidden_layer_neurons,input_dimensions) / np.sqrt(input_dimensions),
        '2':np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    expectation_g_squared = {}

    g_dict = {}
    for layer_name in weigths.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weigths[layer_name])
        g_dict[layer_name] = np.zeros_like(weigths[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    while True:
        env.render()
        processed_observation, prev_processed_observation = preprocess_observation(observation, prev_processed_observation, input_dimensions)
        hidden_layer_values, up_probability = apply_neural_network(processed_observation, weigths)

        episode_observations.append(processed_observation)
        episode_hidden_layer_values.append(hidden_layer_values)

        action = choose_action(up_probability)

        observation, reward, done, info = env.step(action)

        reward_sum += reward

        episode_rewards.append(reward)

        fake_label = 1 if action == 2 else 0

        loss_function_gradient = fake_label - up_probability

        episode_gradient_log_ps.append(loss_function_gradient)

        if done :
            episode_number+=1

            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            episode_gradient_log_ps_discounted =  discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            gradient = compute_gradient(episode_gradient_log_ps_discounted, episode_hidden_layer_values, episode_observations, weigths)

            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            if episode_number % batch_size == 0:
                weights, expectation_g_squared, g_dict = update_weights(weigths, expectation_g_squared, g_dict, decay_rate, learning_rate)

            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
            observation = env.reset()
            running_reward = reward_sum if running_reward is None else running_reward*0.99 + reward_sum*0.01
            print 'resetting env. episode reward total was {}. Running mean : {}.'.format(reward_sum,running_reward)
            reward_sum = 0
            prev_processed_observation = None
main()
