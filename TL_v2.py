import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback


class TL:
    """
    A Transferlearning class by implementing RL on gym environment.

    Here we use DDPG and Pendulum-v1 as RL model and gym environment. The gravity will be changed in evaluation process
    to test the performance of Transferlearning. For each gravity change, we train the model with different seeds. For
    every some steps of training, evaluate it on an environment with a changed value of gravity. We run a trained policy
    for a certain number of episodes. At the end of these evaluation episodes, we get rewards from the agent over time
    steps and take the average rewards, which tell us how the policy performs. Then the results should be performed on a
    plot, which show the curves of the trained model on different evaluation environment. These also contain the maximum
    and minimum intervals.

    Parameters
    ----------
    gravity_list: the gravities for evaluation process
    seeds: different environment initial situation
    gamma: discount factor of rewards function
    alpha: learning rate of DDPG
    sum_episodes: the training episodes for the total training process of each seed and gravity change
    eval_episodes: the evaluation episodes for one evaluation process
    eval_freq: after several training episodes start evaluation process

    Outputs
    ----------
    Files in gravity/seeds/ folders contains:
    1) best_model.zip: contain the best model of each total training process
    2) evaluations.npz: contain the results of each total training process, ep_lengths, results(rewards), timesteps
    A plot:
    which show the curves of the trained model on different evaluation environment
    """

    def __init__(self, gravity_list, seeds):
        self.gravity_list = gravity_list
        self.seeds = seeds
        self.gamma = 0.99  # Discount factor of reward func
        self.alpha = 1e-3  # Learning rate of DDPG
        self.sum_episodes = 10000
        self.eval_episodes = 100
        self.eval_freq = 1000

    def _setup_trainenv(self):
        """
        Set up the training environment using the default gravity(10.0).
        """
        self.env_train = gym.make("Pendulum-v1")

    def _setup_evalenv(self, gravity):
        """
        Set up the evaluation environment using the given gravity to perform transfer learning.

        Parameters
        ----------
        gravity: the gravity for evaluation process
        """
        self.env_eval = gym.make("Pendulum-v1", g=gravity)

    def train_eval(self):
        """
        Do the training-evaluation loops for each seeds and gravity changes.
        """
        self._setup_trainenv()
        self.results_list = [[] for i in range(len(self.gravity_list))]
        for j in range(len(self.gravity_list)):
            gravity = self.gravity_list[j]
            self._setup_evalenv(gravity)
            self.results_timesteps = []
            for k in range(len(self.seeds)):
                seed = self.seeds[k]
                self.env_train.seed(self.seeds[k])
                self.env_eval.seed(self.seeds[k])
                eval_callback = EvalCallback(
                    self.env_eval,
                    best_model_save_path=f"./gravity_{gravity}/seed_{seed}/",
                    log_path=f"./gravity_{gravity}/seed_{seed}/",
                    n_eval_episodes=self.eval_episodes,
                    eval_freq=self.eval_freq,
                    deterministic=True,
                    render=False,
                )

                model = DDPG("MlpPolicy", self.env_train)
                model.learn(self.sum_episodes, callback=eval_callback)
                results = np.load(f"./gravity_{gravity}/seed_{seed}/evaluations.npz")
                self.results_list[j].append(np.mean(results["results"], axis=1))
                self.results_timesteps = results["timesteps"]

    def plot(self):
        """
        Plot the mean evaluate rewards over training time steps on different evaluation environment with different
        seeds.
        """
        createVar = locals()
        fig, ax = plt.subplots()
        color_list = ["red", "blue", "green", "yellow", "grey"]
        plt_list = []
        legend_list = []
        for i in range(len(self.gravity_list)):
            (L,) = ax.plot(
                self.results_timesteps, np.mean(self.results_list[i], axis=0)
            )
            plt_list.append(L)
            legend_list.append(f"Evaluate with gravity {self.gravity_list[i]}")
            i_color = i % len(color_list)
            plt.fill_between(
                self.results_timesteps,
                np.max(self.results_list[i], axis=0),
                np.min(self.results_list[i], axis=0),
                facecolor=color_list[i_color],
                edgecolor="black",
                alpha=0.3,
            )
        plt.legend(plt_list, legend_list, loc=0, prop = {'size':5})
        ax.set(
            xlabel="Timesteps",
            ylabel="Mean rewards",
            title="Evalute Rewards Over Timesteps",
        )
        ax.grid()
        fig.savefig("process.png")
        plt.show()


if __name__ == "__main__":
    process = TL(gravity_list=[10.0, 20.0, 30.0], seeds=[2, 6, 1, 7, 8])
    TL.train_eval(process)
    TL.plot(process)
