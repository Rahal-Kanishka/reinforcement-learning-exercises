from frozen_lake import FrozenLake
import gym
import tensorflow as tf

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    frozeLake = FrozenLake()
    env = gym.make('FrozenLake-v1', render_mode= "human")

    for episode in range(600):
        print('Episode: ', episode, ', epsilon: ', frozeLake.epsilon)
        obs = env.reset()[0]
        for step in range(200):
            epsilon = max(1 - episode / 50, 0.01)
            obs, reward, done, truncated, info = frozeLake.play_one_step(env, obs, epsilon)
            # print('step: ', step, ', Epsilon: ', epsilon, ', Reward: ', reward, ', obs: ', obs)
            env.render()
            if done or truncated:
                break

        if episode > 25:
            frozeLake.training_step(batch_size=32)


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