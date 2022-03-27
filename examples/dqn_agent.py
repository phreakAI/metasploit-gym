"""Example 
"""
import random
import numpy as np
import metasploit_gym
import virtualbox
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    def __init__(self, capacity, s_dims, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [
            self.s_buf[sample_idxs],
            self.a_buf[sample_idxs],
            self.next_s_buf[sample_idxs],
            self.r_buf[sample_idxs],
            self.done_buf[sample_idxs],
        ]
        return [torch.from_numpy(buf).to(self.device) for buf in batch]


class DQN(nn.Module):
    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim[0] * input_dim[1], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l - 1], layers[l]))
        self.out = nn.Linear(layers[-1], num_actions)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def get_action(self, x):
        x = x.unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            return self.forward(x).max(1)[1]


class DQNAgent:
    """A simple Deep Q-Network Agent"""

    def __init__(
        self,
        env,
        seed=42,
        lr=0.0001,
        training_steps=100,
        batch_size=32,
        replay_size=1000,
        final_epsilon=0.05,
        exploration_steps=50,
        gamma=0.99,
        hidden_sizes=[64, 64],
        target_update_freq=10,
        verbose=True,
        **kwargs,
    ):
        self.verbose = True
        self.seed = seed
        np.random.seed(self.seed)

        self.env = env

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        self.logger = SummaryWriter()

        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(
            1.0, self.final_epsilon, self.exploration_steps
        )

        self.batch_size = batch_size
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0

        # Neural networks related attributes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(self.obs_dim, hidden_sizes, self.num_actions).to(self.device)

        self.target_dqn = DQN(self.obs_dim, hidden_sizes, self.num_actions).to(
            self.device
        )

        self.target_update_freq = target_update_freq

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # replay setup
        self.replay = ReplayMemory(replay_size, self.obs_dim, self.device)

    def save(self, file_path):
        self.dqn.save_model(file_path)

    def load(self, file_path):
        self.dqn.load_model(file_path)

    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            o = torch.from_numpy(o).float().to(self.device)
            return self.dqn.get_action(o).cpu().item()
        return random.randint(0, self.num_actions - 1)

    def optimize(self):
        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch
        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch)

        with torch.no_grad():
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_val = target_q_val_raw.max(1)[0]
            target = r_batch + self.discount * (1 - d_batch) * target_q_val

        # calculate loss
        q_vals = q_vals.view(-1)

        loss = self.loss_fn(q_vals, target)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        return loss.item(), mean_v

    def train(self):
        num_episodes = 0
        training_steps_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(10)
            ep_return, ep_steps, goal = ep_results
            num_episodes += 1
            training_steps_remaining -= ep_steps

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar("epsilon", self.get_epsilon(), self.steps_done)
            self.logger.add_scalar("episode_return", ep_return, self.steps_done)
            self.logger.add_scalar("episode_goal_reached", int(goal), self.steps_done)

            if num_episodes % 10 == 0 and self.verbose:
                print(f"\nEpisode {num_episodes}:")
                print(f"\tsteps done = {self.steps_done} /" f"{self.training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")
                self.dqn.save_model(
                    f"models/ScriptKiddie_low_lr_Episode_{num_episodes}.pt"
                )
        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")

    def run_train_episode(self, step_limit):
        o = self.env.reset()
        done = False

        steps = 0
        episode_return = 0
        print("STARTING EPISODE")
        print(f"Step limit = {step_limit}")

        while not done and steps < step_limit:
            a = self.get_egreedy_action(o, self.get_epsilon())
            action = env.action_space.get_action(a)
            next_o, r, done, _ = self.env.step(action)
            print(f"ACTION REWARD {r}")
            self.replay.store(o, a, next_o, r, done)
            self.steps_done += 1
            loss, mean_v = self.optimize()
            self.logger.add_scalar("loss", loss, self.steps_done)
            self.logger.add_scalar("mean_v", mean_v, self.steps_done)

            o = next_o
            episode_return += r
            steps += 1
        print("ENDING EPISODE")
        return episode_return, steps, self.env.goal_reached()

    def run_eval_episode(
        self, env=None, render=False, eval_epsilon=0.05, render_mode="readable"
    ):
        if env is None:
            env = self.env
        o = env.reset()
        done = False

        steps = 0
        episode_return = 0

        line_break = "=" * 60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render(render_mode)
            input("Initial state. Press enter to continue..")

        while not done:
            a = self.get_egreedy_action(o, eval_epsilon)
            action = env.action_space.get_action(a)
            next_o, r, done, _ = env.step(action)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Steps {steps}")
                print(line_break)
                print(f"Action performed = {env.action_space.get_action(a)}")
                env.render(render_mode)
                print(f"Reward = {r}")
                print(f"Done = {done}")
                input("Press enter to continue.")

                if done:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")
        return episode_return, steps, env.goal_reached()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render_eval", action="store_true", help="Renders final policy"
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="*",
        default=[64, 64],
        help="(default=[64. 64])",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default=0.001)"
    )
    parser.add_argument(
        "-t",
        "--training_steps",
        type=int,
        default=200,
        help="training steps (default=20000)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="(default=32)")
    parser.add_argument(
        "--target_update_freq", type=int, default=1000, help="(default=1000)"
    )
    parser.add_argument("--seed", type=int, default=0, help="(default=0)")
    parser.add_argument("--replay_size", type=int, default=1000, help="(default=1000)")
    parser.add_argument(
        "--final_epsilon", type=float, default=0.05, help="(default=0.05)"
    )
    parser.add_argument("--init_epsilon", type=float, default=1.0, help="(default=1.0)")
    parser.add_argument(
        "--exploration_steps", type=int, default=10000, help="(default=10000)"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="(default=0.99)")
    parser.add_argument("--quite", action="store_false", help="Run in Quite mode")
    args = parser.parse_args()

    def environment_reset():
        start = time.time()
        name = "gym_episode_start"
        vb = virtualbox.VirtualBox()
        session = virtualbox.Session()
        try:
            vm = vb.find_machine("metasploitable")
            snap = vm.find_snapshot(name)
            vm.create_session(session=session)
        except virtualbox.library.VBoxError as e:
            print(e.msg)
            return True
        except Exception as e:
            print(str(e))
            return True
        shutting_down = session.console.power_down()
        while shutting_down.operation_percent < 100:
            time.sleep(0.5)
        restoring = session.machine.restore_snapshot(snap)
        while restoring.operation_percent < 100:
            time.sleep(0.5)
        vm = vb.find_machine("metasploitable")
        session = virtualbox.Session()
        proc = vm.launch_vm_process(session, "gui", [])
        proc.wait_for_completion(timeout=-1)
        return True

    env = metasploit_gym.metasploit_env.MetasploitNetworkEnv(
        reset_function=environment_reset
    )
    dqn_agent = DQNAgent(env, verbose=args.quite, **vars(args))
    dqn_agent.train()
    dqn_agent.run_eval_episode(render=args.render_eval)
