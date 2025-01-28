import tensorflow as tf
import numpy as np
import google.protobuf
import sys
import matplotlib.pyplot as plt

from collections import deque

plt.ion()


def one_hot_encode_input(state, env_state_size):
    one_hot_state = np.zeros(env_state_size)
    one_hot_state[state] = 1
    return one_hot_state


class FrozenLake:
    replay_buffer = deque(maxlen=2000)
    batch_size = 32
    discount_factor = 0.95
    optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-4, clipnorm=1.0)
    loss_function = tf.keras.losses.MeanSquaredError()

    def __init__(self, env_state_size=16, n_actions=4):
        self.epsilon = 1
        self.batch_size = 100
        self.steps = 0
        self.state_array = [0] * 16
        self.state_array = [0] * 16
        self.reward_array = []
        self.epsilon_array = []
        self.prediction_array = [-1] * 16
        self.loss_array = []
        self.input_shape = env_state_size
        self.output_shape = n_actions
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(env_state_size,)),
            tf.keras.layers.Dense(units=32, activation="elu"),
            tf.keras.layers.Dense(units=32, activation="elu"),
            tf.keras.layers.Dense(self.output_shape)
        ])
        print("Initializing frozen lake")
        print("TensorFlow version:", tf.__version__)
        print("Protobuf version: ", google.protobuf.__version__)
        print("python version: ", sys.version)
        print("model summary: ", self.model.summary())

    def initialize_testing(self):
        self.model = tf.keras.models.load_model('output/RF_model.keras')
        print("model loaded")
        self.model.summary()

    def predict_using_model(self, state):
        # encode input
        encoded_input = np.array([one_hot_encode_input(state, self.input_shape)])  #  bring to (none, 16) shape
        # print('encoded input: ', encoded_input)
        next_q_values = self.model.predict(encoded_input, verbose=0)
        return next_q_values

    def play_trained_agent(self, env, state):
        q_values = self.model.predict(self.predict_using_model(state), verbose=0)[0]
        print("Playing by model: ", q_values, ', state: ', state)
        action = np.argmax(q_values)
        next_state, reward, done, truncated, info = env.step(action)
        return next_state, reward, done, truncated, info

    def epsilon_greedy_policy(self, state):
        if np.random.rand() < self.epsilon:
            # print("Playing random")
            return np.random.randint(self.output_shape)  # random action
        else:
            q_values = self.model.predict(np.array([one_hot_encode_input(state, self.input_shape)]), verbose=0)[0]
            print("Playing by model: ", q_values, ', state: ', state, ', epsilon: ', self.epsilon)
            # self.prediction_array.append({np.argmax(q_values),np.argmax(q_values)})
            return np.argmax(q_values)  # find max index (action value)

    def sample_experience(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        # print('Indices: ', indices)
        batch = [self.replay_buffer[index] for index in indices]
        return [
            np.array([batch_item[field_index] for batch_item in batch]) for field_index in range(5)
        ]  # [ state, actions, rewards, next_states, truncates, dones]

    def prioritize_sample_experience(self):
        batch = []
        indices = []
        for i in range(16):
            filtered_data = list(filter(lambda item: item[3] == i, self.replay_buffer))
            if len(filtered_data) > 2:
                # batch.append(random.sample(filtered_data, 2))
                tmp = np.random.randint(len(filtered_data), size=2)  # get two items random from the results
                # tmp = np.random.choice(len(filtered_data), size=2, replace=False)
                indices.extend([self.replay_buffer.index(filtered_data[tmp[0]]),
                                self.replay_buffer.index(filtered_data[tmp[1]])])  # get the index
            elif len(filtered_data) == 2:
                indices.append(self.replay_buffer.index(filtered_data[0]))
                indices.append(self.replay_buffer.index(filtered_data[1]))
            elif len(filtered_data) == 1:
                indices.append(self.replay_buffer.index(filtered_data[0]))

        # print('Indices: ', indices)
        batch = [self.replay_buffer[index] for index in indices]

        return [
            np.array([batch_item[field_index] for batch_item in batch]) for field_index in range(5)
        ]

    def play_one_step(self, env, state):
        self.state_array[state] += 1
        self.steps += 1
        action = self.epsilon_greedy_policy(state)
        next_state, reward, done, truncated, info = env.step(action)
        if done:  # could be done because agent went to the target or fll to the hole
            if reward == 0:  # done because falling to the hole
                reward = -1
            if reward == 1:  # done because of completion
                reward = 10
        else:
            # if next state belongs to last two rows, give increased reward
            if next_state > 11:  # for reaching last row
                reward = 2
            elif next_state > 7:  # for reaching one before last row
                reward = 1
            else:
                reward = 0.10  # minor encouragement for staying in the board

        # check if agent tres to move out of boundary
        if state == next_state:
            print('Agent tries to go out of the boundary')
            reward = -0.50  # to encourage exploration I didn't put same punishment as falling to lake

        print("Reward: ", reward, " state: ", state)
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.reward_array.append(reward)
        self.epsilon_array.append(self.epsilon)
        # plot reward
        if reward > 0 or (self.steps % 10 == 0):
            plt.figure("Rewards Graph plot")
            plt.plot(range(self.steps), np.array(self.reward_array))
            plt.plot(range(self.steps), np.array(self.epsilon_array))
            plt.title('Rewards Graph')
            # Display the plot
            plt.show(block=False)
        return next_state, reward, done, truncated, info

    def training_step(self, batch_size):
        """ Calculating q values for the next states (next_q_value) and calculate target_q_values using bellman equation
        and  then calculate the loss between them and train the model """
        experiences = self.prioritize_sample_experience()
        # results are separate arrays
        states, actions, rewards, next_states, dones = experiences
        # print('actions: ', actions)
        # encode all input states before feeding to the model
        encoded_next_states = []
        encoded_current_states = []
        # encode next states
        for state in next_states:
            encoded_next_states.append(one_hot_encode_input(state, self.input_shape))
        # encode current states
        for current_state in states:
            encoded_current_states.append(one_hot_encode_input(current_state, self.input_shape))
        # print('encoded next_states:', np.array(encoded_next_states))
        next_q_values = self.model.predict(np.array(encoded_next_states), verbose=0)
        # print("selecting max Q from: ", next_q_values)
        # self.print_state_values(next_states, next_q_values, batch_size)
        max_next_q_value = next_q_values.max(axis=1)
        # print("max Q: ", max_next_q_value)

        runs = 1.0 - dones
        # using bellman equation
        target_q_values = rewards + runs * self.discount_factor * max_next_q_value
        target_q_values = target_q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.output_shape)
        """ [[0, 0, 0, 1],  # Action 3
         [1, 0, 0, 0],  # Action 0
         [0, 0, 0, 1],  # Action 3
         [0, 0, 1, 0]]  # Action 2 """

        with tf.GradientTape() as tape:
            # get the q values generated without using bellman equation
            all_q_values = self.model(np.array(encoded_current_states))
            # plot currently learnt states
            self.plot_q_values(states, all_q_values)
            q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
            # calculate loss
            loss = tf.reduce_mean(self.loss_function(target_q_values, q_values))
            self.loss_array.append(loss)
            plt.figure("Loss graph")
            plt.plot(range(len(self.loss_array)), np.array(self.loss_array))
            plt.grid(False)
            plt.show(block=False)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def get_action(self, state):
        if state == 0:
            return 'Left'
        elif state == 1:
            return 'Down'
        elif state == 2:
            return 'Right'
        else:
            return 'Up'

    def print_state_values(self, lake_states, predicted_values, batch_size):
        results = [-1] * batch_size
        print('lake_states: ', lake_states)
        for index, state in enumerate(lake_states):
            max_q_action = np.argmax(predicted_values[index][0])
            print('state: ', state, ', ', self.get_action(max_q_action))

    def end_of_training(self):
        # Save model
        self.model.save("output/RF_model.keras")
        # save graphs
        plt.savefig('output/graph.png')
        print('Model and graphs saved')

    def plot_q_values(self, states, q_values):
        # process q vales
        print('plot states: ', states, ', q_values: ', q_values)
        fig, axes = plt.subplots(4, 4, figsize=(16, 10))

        for state_id in range(16):
            # since st
            state_array_index = -1
            state_indexes = np.where(states == state_id)[0]
            if state_indexes is not None and len(state_indexes) > 0:
                state_array_index = np.where(states == state_id)[0][0]
            else:
                print('No states found for : ', state_id)
                break
            row, col = divmod(state_id, 4)
            ax = axes[row, col]
            # print('q_values: ', q_values)
            ax.bar(['Left', 'Down', 'Right', 'Up'], q_values[state_array_index], color='green')
            ax.set_title(f"State {state_id}")
            ax.set_ylim([0, np.max(q_values)])  # Keep y-axis consistent

        for ax in axes.flat[len(states):]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig("output/q_values_grid.png", dpi=300)  # Save the entire grid as a file
        # plt.show()
        plt.close()
