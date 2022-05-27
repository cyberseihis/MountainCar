from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym

# configurations
env = gym.make("MountainCarContinuous-v0")
observe_dim = 2
action_num = 1
max_episodes = 1000
max_steps = 200
solved_reward = 190
solved_repeat = 5


# model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, some_state):
        a = t.relu(self.fc1(some_state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


if __name__ == "__main__":
    critic = QNet(observe_dim, 1)
    actor = QNet(observe_dim, action_num)
    dqn = PPO(actor, critic,
              t.optim.Adam,
              nn.MSELoss(reduction='sum'))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    terminal = False

    while episode < max_episodes:
        EpTransisions = []
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = t.tensor(env.reset(), dtype=t.float32).view(1, observe_dim)

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act(
                    {"some_state": old_state}
                )
                acted = action[0].tolist()
                state, reward, terminal, _ = env.step(acted[0])
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                EpTransisions.append({
                    "state": {"some_state": old_state},
                    "action": {"action": action[0]},
                    "next_state": {"some_state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps
                })

        # update, update more if episode is longer, else less
        dqn.store_episode(EpTransisions)
        if episode > 10:
            for _ in range(step):
                dqn.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0
