# Object-oriented principles: 1)Abstraction 2)Inheritance 3)Encapsulation 4)Polymorphism
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

class TransferLearning:

    def __init__(self, gravity_change, gamma, alpha):
        self.gravity = 10.0
        self.gravity_change = gravity_change
        self.gamma = gamma # Discount factor of reward func
        self.alpha = alpha # Learning rate of DDPG
        self.epochs = 5
        self.steps_train = 1000
        self.steps_eval = 100
        self.timesteps = 0
        self.seeds = [2, 6, 7, 1, 9]

    def train(self, model):# steps_train times new env train
        steps_train = self.steps_train
        model.learn(total_timesteps=steps_train, log_interval=10)
        self.timesteps += steps_train
        return model

    def eval(self, model, gravity, seed): # one time env eval
        steps_eval = self.steps_eval
        if gravity is not None:
            env_eval = gym.make("Pendulum-v1", g=gravity)
        else:
            env_eval = gym.make("Pendulum-v1")
        env_eval.seed(seed)
        obs = env_eval.reset()
        sum_rewards = 0
        for step in range(steps_eval):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env_eval.step(action)
            sum_rewards += rewards
        sum_rewards /= steps_eval
        return sum_rewards

    def learn(self):
        epochs = self.epochs
        alpha = self.alpha
        gamma = self.gamma
        gravity_change = self.gravity_change
        seeds = self.seeds
        list_no = [[] for i in range(len(seeds))]
        list_g = [[] for j in range(len(seeds))]
        for k in range(len(seeds)):
            print(f"seeds:{k}")
            sum_steps = []
            self.timesteps = 0
            env = gym.make("Pendulum-v1")
            env.seed(seeds[k])
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
            model = DDPG("MlpPolicy", env, learning_rate=alpha, gamma=gamma, action_noise=action_noise, verbose=1)
            for epoch in range(epochs):
                print(f"epoch:{epoch}")
                model = self.train(model)
                reward_no = self.eval(model,gravity=None, seed=seeds[k])
                reward_g = self.eval(model,gravity=gravity_change, seed=seeds[k])
                list_no[k].append(reward_no)
                list_g[k].append(reward_g)
                sum_steps.append(self.timesteps)
        fig, ax = plt.subplots()
        L1, = ax.plot(sum_steps, np.mean(list_no, axis=0))
        L2, = ax.plot(sum_steps, np.mean(list_g, axis=0))
        plt.fill_between(sum_steps, np.max(list_no, axis=0), np.min(list_no, axis=0),
                         facecolor='green', edgecolor='black', alpha=0.3)
        plt.fill_between(sum_steps, np.max(list_g, axis=0), np.min(list_g, axis=0),
                         facecolor='red', edgecolor='black', alpha=0.3)
        plt.legend([L1, L2], ['Evaluate without change', 'Evaluate with gravity change'], loc=4)
        ax.set(xlabel='Timesteps', ylabel='Mean rewards',
                title='Evalute Rewards Over Timesteps')
        ax.grid()
        fig.savefig("test.png")
        plt.show()

if __name__ == "__main__":
    test = TransferLearning(gravity_change=20.0,gamma=0.99,alpha=1e-3)
    TransferLearning.learn(test)

