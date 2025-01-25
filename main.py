from frozen_lake import FrozenLake
import gym
import tensorflow as tf


def train_model():
    global step
    # execute only if run as the entry point into the program
    frozenLake = FrozenLake()
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
    state_array = [0] * 16
    for episode in range(300):
        print('Episode: ', episode, ', epsilon: ', frozenLake.epsilon, 'state steps: ', state_array)
        obs = env.reset()[0]
        for step in range(500):
            epsilon = max(1 - episode / 500, 0.01)
            obs, reward, done, truncated, info = frozenLake.play_one_step(env, obs, epsilon)
            # print('step: ', step, ', Epsilon: ', epsilon, ', Reward: ', reward, ', obs: ', obs)
            state_array[obs] += 1
            env.render()
            if done or truncated:
                break

        if episode > 50:
            frozenLake.training_step(batch_size=32)
    frozenLake.end_of_training()


def test_model():
    frozen_lake_test = FrozenLake()
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")
    total_steps = 0
    frozen_lake_test.initialize_testing()
    for episode in range(50):
        obs = env.reset()[0]
        for test_step in range(500):
            obs, reward, done, truncated, info = frozen_lake_test.play_trained_agent(env, obs)
            env.render()
            total_steps += 1
            print('Episode: ', episode, ', steps: ', total_steps, ', reward: ', reward)
            if done or truncated:
                break


if __name__ == '__main__':
    test_model()

# if __name__ == '__main__':
#     print("TensorFlow version:", tf.__version__)
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     print('gpus', gpus)
#     cifar = tf.keras.datasets.cifar100
#     (x_train, y_train), (x_test, y_test) = cifar.load_data()
#     model = tf.keras.applications.ResNet50(
#         include_top=True,
#         weights=None,
#         input_shape=(32, 32, 3),
#         classes=100, )
#
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#     model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
#     model.fit(x_train, y_train, epochs=5, batch_size=64)
