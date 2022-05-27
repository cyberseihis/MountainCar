from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
from torch.distributions import Normal, Uniform
import torch as t
import torch.nn as nn
import gym

# configurations
env = gym.make("MountainCarContinuous-v0")
observe_dim = 2
max_episodes = 1000
max_steps = 999
solved_repeat = 3


# model definition
class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, state, action=None):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        probs = t.sigmoid(self.fc3(a))
        mu = probs[0][0] * 2 - 1
        sig = probs[0][1]
        dist = Normal(mu, sig)
        act = action if action is not None else dist.sample()
        act_entropy = dist.entropy()
        act_log_prob = dist.log_prob(act.flatten())
        return act, act_log_prob, act_entropy


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state):
        v = t.relu(self.fc1(state))
        v = t.relu(self.fc2(v))
        v = self.fc3(v)
        return v


def train():
    actor = Actor(observe_dim)
    critic = Critic(observe_dim)

    ppo = PPO(
        actor, critic,
        t.optim.Adam, nn.MSELoss(reduction="sum"),
        actor_update_times=5, critic_update_times=10,
        actor_learning_rate=1e-4, critic_learning_rate=1e-4,
        discount=0.95)

    episode, step = 0, 0
    smoothed_total_reward = 0
    found = False
    Scores = []
    Dist = []

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)
        changeRate = 5
        unif = Uniform(-1, 1)
        maxDist = -10
        tmp_observations = []
        while not terminal and step <= max_steps:
            # if episode % 100 == 0:
            # env.render()
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                if step % changeRate == 1:
                    if episode > 10:
                        action = ppo.act({"state": old_state})[0]
                    else:
                        action = unif.sample()
                state, reward, terminal, _ = env.step([action])
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                # reward = max(reward * 0.05, reward)
                total_reward += reward
                maxDist = max(maxDist, state[0][0])

                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": t.tensor([[action]])},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

        # update
        ppo.store_episode(tmp_observations)
        if total_reward > 0:
            found = True
            logger.info("On Flag")
        if found:
            pass
        elif episode > 20:
            return episode
        ppo.update()

        # show reward
        smoothed_total_reward *= 0.9
        smoothed_total_reward += total_reward * 0.1
        Info = f"Ep {episode} >={maxDist:.2f} $={smoothed_total_reward:.2f}"
        logger.info(Info)
        Scores.append(total_reward)
        Dist.append(maxDist)


if __name__ == "__main__":
    while train() < 30:
        pass
