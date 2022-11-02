# Object-oriented principles: 1)Abstraction 2)Inheritance 3)Encapsulation 4)Polymorphism
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

class TransferLearning:

    def __init__(self, gravity_change, length_change, gamma, alpha):
        self.gravity = 10.0
        self.gravity_change = gravity_change
        self.length = 1.0
        self.length_change = length_change
        self.gamma = gamma # Discount factor of reward func
        self.alpha = alpha # Learning rate of DDPG
        self.epochs = 100
        self.steps_train = 1000
        self.steps_eval = 100
        self.timesteps = 0
        self.seeds = [2, 6, 7, 1, 9]

    def train(self, env, model):# steps_train times new env train
        steps_train = self.steps_train
        # The noise objects for DDPG
        model.learn(total_timesteps=steps_train, log_interval=10)
        self.timesteps += steps_train
        return model

    def eval(self, model, gravity, seeds): # one time env eval
        if gravity is not None:
            env_eval = gym.make("Pendulum-v1", g=gravity)
        else:
            env_eval = gym.make("Pendulum-v1")
        steps_eval = self.steps_eval
        list_rewards = []
        for seed in seeds:
            env_eval.seed(seed)
            obs = env_eval.reset()
            sum_rewards = 0
            for step in range(steps_eval):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env_eval.step(action)
                sum_rewards += rewards
            list_rewards.append(sum_rewards/steps_eval)
        return list_rewards

    def learn(self):
        epochs = self.epochs
        alpha = self.alpha
        gamma = self.gamma
        gravity_change = self.gravity_change
        length_change = self.length_change
        seeds = self.seeds
        env = gym.make("Pendulum-v1")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG("MlpPolicy", env, learning_rate=alpha, gamma=gamma, action_noise=action_noise, verbose=1)
        sum_steps = []
        maxlist_no = []
        meanlist_no = []
        minlist_no = []
        maxlist_g = []
        meanlist_g = []
        minlist_g = []
        for epoch in range(epochs):
            model = self.train(env, model)
            reward_no = self.eval(model,gravity=None, seeds=seeds)
            reward_g = self.eval(model,gravity=gravity_change, seeds=seeds)
            max_no = max(reward_no)
            mean_no = sum(reward_no)/len(reward_no)
            min_no = min(reward_no)
            max_g = max(reward_g)
            mean_g = sum(reward_g)/len(reward_g)
            min_g = min(reward_g)
            maxlist_no.append(max_no)
            meanlist_no.append(mean_no)
            minlist_no.append(min_no)
            maxlist_g.append(max_g)
            meanlist_g.append(mean_g)
            minlist_g.append(min_g)
            sum_steps.append(self.timesteps)
            print(f"epoch:{epoch}, rewards no change:{mean_no}, rewards g change:{mean_g}")
        fig, ax = plt.subplots()
        ax.plot(sum_steps, maxlist_no)
        ax.plot(sum_steps, meanlist_no)
        ax.plot(sum_steps, minlist_no)
        ax.plot(sum_steps, maxlist_g)
        ax.plot(sum_steps, meanlist_g)
        ax.plot(sum_steps, minlist_g)
        ax.set(xlabel='timesteps', ylabel='rewards',
               title='Evalute rewards over timesteps')
        ax.grid()
        fig.savefig("test.png")
        plt.show()

if __name__ == "__main__":
    test = TransferLearning(gravity_change=100.0,length_change=0.01,gamma=0.99,alpha=1e-3)
    TransferLearning.learn(test)

