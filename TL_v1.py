# Object-oriented principles: 1)Abstraction 2)Inheritance 3)Encapsulation 4)Polymorphism
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


class TL:
    def __init__(self, gravity_change, gamma, alpha):
        self.gravity = 10.0
        self.gravity_change = gravity_change
        self.gamma = gamma  # Discount factor of reward func
        self.alpha = alpha  # Learning rate of DDPG
        self.epochs = 5
        self.steps_train = 1000
        self.steps_eval = 100
        self.timesteps = 0
        self.seeds = [2, 6, 7, 1, 9]

    def train(self, model):  # steps_train times new env train
        model.learn(total_timesteps=self.steps_train, log_interval=10)
        self.timesteps += self.steps_train
        return model

    def eval(self, model, gravity, seed):  # one time env eval
        if gravity is not None:
            self.env_eval = gym.make("Pendulum-v1", g=gravity)
        else:
            self.env_eval = self.env_train
        self.env_eval.seed(seed)
        obs = self.env_eval.reset()
        sum_rewards = 0
        for step in range(self.steps_eval):
            action, _states = self.model.predict(obs)
            obs, rewards, dones, info = self.env_eval.step(action)
            sum_rewards += rewards
        sum_rewards /= self.steps_eval
        return sum_rewards

    def learn(self):
        list_no = [[] for i in range(len(self.seeds))]
        list_g = [[] for j in range(len(self.seeds))]
        sum_steps = []
        self.env_train = gym.make("Pendulum-v1")
        n_actions = self.env_train.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        for k in range(len(self.seeds)):
            print(f"seeds:{k}")
            sum_steps = []
            self.timesteps = 0
            self.env_train.seed(self.seeds[k])
            self.model = DDPG(
                "MlpPolicy",
                self.env_train,
                learning_rate=self.alpha,
                gamma=self.gamma,
                action_noise=action_noise,
                verbose=1,
            )
            for epoch in range(self.epochs):
                print(f"epoch:{epoch}")
                self.model = self.train(self.model)
                reward_no = self.eval(self.model, gravity=None, seed=self.seeds[k])
                reward_g = self.eval(
                    self.model, gravity=self.gravity_change, seed=self.seeds[k]
                )
                list_no[k].append(reward_no)
                list_g[k].append(reward_g)
                sum_steps.append(self.timesteps)
        self.sum_steps = sum_steps
        self.list_no = list_no
        self.list_g = list_g


if __name__ == "__main__":
    test = TL(gravity_change=20.0, gamma=0.99, alpha=1e-3)
    TL.learn(test)

    fig, ax = plt.subplots()
    (L1,) = ax.plot(test.sum_steps, np.mean(test.list_no, axis=0))
    (L2,) = ax.plot(test.sum_steps, np.mean(test.list_g, axis=0))
    plt.fill_between(
        test.sum_steps,
        np.max(test.list_no, axis=0),
        np.min(test.list_no, axis=0),
        facecolor="green",
        edgecolor="black",
        alpha=0.3,
    )
    plt.fill_between(
        test.sum_steps,
        np.max(test.list_g, axis=0),
        np.min(test.list_g, axis=0),
        facecolor="red",
        edgecolor="black",
        alpha=0.3,
    )
    plt.legend(
        [L1, L2], ["Evaluate without change", "Evaluate with gravity change"], loc=4
    )
    ax.set(
        xlabel="Timesteps",
        ylabel="Mean rewards",
        title="Evalute Rewards Over Timesteps",
    )
    ax.grid()
    fig.savefig("test.png")
    plt.show()
