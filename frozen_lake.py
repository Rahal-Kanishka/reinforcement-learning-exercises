import tensorflow as tf
import numpy as np
import google.protobuf
import sys
import matplotlib.pyplot as plt
import random

from collections import deque


plt.ion()

class FrozenLake:
    input_shape = [1]
    n_outputs = 4
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=32, activation="elu", input_shape=input_shape),
        tf.keras.layers.Dense(units=32, activation="elu"),
        tf.keras.layers.Dense(n_outputs)
    ])
    replay_buffer = deque(maxlen=2000)
    batch_size = 32
    discount_factor = 0.95
    optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-3)
    loss_function = tf.keras.losses.MeanSquaredError()

    def __init__(self):
        print("Initializing frozen lake")
        print("TensorFlow version:", tf.__version__)
        print("Protobuf version: ", google.protobuf.__version__)
        print("python version: ", sys.version)
        self.epsilon = 0.99
        self.batch_size = 100
        self.steps = 0
        self.state_array = [0] * 16
        self.state_array = [0] * 16
        self.reward_array = []
        self.epsilon_array = []
        self.prediction_array = [-1] * 16
        self.loss_array = []

    def initialize_testing(self):
        self.model = tf.keras.models.load_model('output/RF_model.h5')
        print("model loaded")
        self.model.summary()

    def play_trained_agent(self, env, state):
        q_values = self.model.predict(np.array([state], dtype=np.float32), verbose=0)[0]
        print("Playing by model: ", q_values, ', state: ', state)
        action = np.argmax(q_values)
        next_state, reward, done, truncated, info = env.step(action)
        return next_state, reward, done, truncated, info


    def epsilon_greedy_policy(self, state, epsilon=0.0):
        if np.random.rand() < epsilon:
            # print("Playing random")
            return np.random.randint(self.n_outputs)  # random action
        else:
            q_values = self.model.predict(np.array([state], dtype=np.float32), verbose=0)[0]
            print("Playing by model: ", q_values, ', state: ', state, ', epsilon: ', epsilon)
            # self.prediction_array.append({np.argmax(q_values),np.argmax(q_values)})
            return np.argmax(q_values)  # find max index (action value)

    def sample_experience(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        print('Indices: ', indices)
        batch = [self.replay_buffer[index] for index in indices]
        return [
            np.array([batch_item[field_index] for batch_item in batch]) for field_index in range(5)
        ]  # [ state, actions, rewards, next_states, truncates, dones]

    def prioritize_sample_experience(self, batch_size):
        batch = []
        indices = []
        for i in range(16):
            filtered_data = list(filter(lambda item: item[3] == i, self.replay_buffer))
            if len(filtered_data) > 2:
                # batch.append(random.sample(filtered_data, 2))
                tmp = np.random.randint(len(filtered_data), size=2)  # get two items random from the results
                indices.extend([self.replay_buffer.index(filtered_data[tmp[1]]), self.replay_buffer.index(filtered_data[tmp[1]])])  # get the index
            elif len(filtered_data) == 2:
                indices.append(self.replay_buffer.index(filtered_data[0]))
                indices.append(self.replay_buffer.index(filtered_data[1]))
            elif len(filtered_data) == 1:
                indices.append(self.replay_buffer.index(filtered_data[0]))

        print('Indices: ', indices)
        batch = [self.replay_buffer[index] for index in indices]

        return [
            np.array([batch_item[field_index] for batch_item in batch]) for field_index in range(5)
        ]

    def play_one_step(self, env, state, epsilon):
        self.state_array[state] += 1
        self.steps += 1
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, truncated, info = env.step(action)
        if done:  # could be done because agent went to the target or fll to the hole
            if reward == 0:  # done because falling to the hole
                reward = -1
            if reward == 1:  # done because of completion
                reward = 100

        else:
            reward = 0.01  # minor encouragement for staying in the board
        # check if agent tres to move out of boundary
        if state == next_state:
            print('Agent tries to go out of the boundary')
            reward = -0.5  # to encourage exploration I didn't put same punishment as falling to lake

        print("Reward: ", reward, " steps: ", self.steps)
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
        experiences = self.prioritize_sample_experience(batch_size)
        # results are separate arrays
        states, actions, rewards, next_states, dones = experiences
        next_q_values = self.model.predict(next_states, verbose=0)
        print("selecting max Q from: ", next_q_values)
        self.print_state_values(next_states, next_q_values, batch_size)
        max_next_q_value = next_q_values.max(axis=1)
        print("max Q: ", max_next_q_value)
        runs = 1.0 - dones
        target_q_values = rewards + runs * self.discount_factor * max_next_q_value
        target_q_values = target_q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_q_values = self.model(states)
            q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
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
            max_q_action = np.argmax(predicted_values[index])
            print('state: ', state, ', ', self.get_action(max_q_action))

    def end_of_training(self):
        # Save model
        self.model.save("output/RF_model.h5")
        # save graphs
        plt.savefig('output/graph.png')
        print('Model and graphs saved')

    def plot_q_values(self, states, q_values):

        # process q vales

        fig, axes = plt.subplots(4, 4, figsize=(16, 10))

        for state_id in range(16):
            # since st
            state_array_index = -1
            state_indexes = np.where(states == state_id)[0]
            if state_indexes is not None and len(state_indexes) > 0:
                state_array_index = np.where(states == state_id)[0][0]
            else:
                break
            row, col = divmod(state_id, 4)
            ax = axes[row, col]
            ax.bar(['Left', 'Down', 'Right', 'Up'], q_values[state_array_index], color='green')
            ax.set_title(f"State {state_id}")
            ax.set_ylim([0, np.max(q_values)])  # Keep y-axis consistent

        for ax in axes.flat[len(states):]:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig("output/q_values_grid.png", dpi=300)  # Save the entire grid as a file
        # plt.show()
        plt.close()
