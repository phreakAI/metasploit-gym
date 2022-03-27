import numpy as np
import metasploit_gym
import time
import time

LINE_BREAK = "-" * 60


def run_random_agent(env, step_limit=10, verbose=True):
    if verbose:
        print(LINE_BREAK)
        print("STARTING EPISODE")
        print(LINE_BREAK)
    # reset environment from last episode
    env.reset()
    total_reward = 0
    done = False
    t = 0
    start = time.time()
    action_count = 0
    while not done and t < step_limit:
        a = env.action_space.sample()
        a = env.action_space.get_action(a)
        print(a)
        _, r, done, _ = env.step(a)
        if done:
            print("DONE AT")
            print(t)
            return t, total_reward, env.goal_reached()
        if action_count == 256:
            stop = time.time()
            print("To collect 256 time steps requires: ")
            print(stop - start)
        total_reward += r
        if (t + 1) % 20 == 0 and verbose:
            print(f"t: {t}: reward:{total_reward}")
        t += 1

    if done and verbose:
        print(LINE_BREAK)
        print("EPISODE COMPLETE")
        print(LINE_BREAK)
        print(f"Total steps = {t}")
        print(f"Total reward = {total_reward}")
    elif verbose:
        print(LINE_BREAK)
        print("STEP LIMIT REACHED")
        print(LINE_BREAK)
    return t, total_reward, env.goal_reached()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=20,
        help="Number of random runs to perform (default=1",
    )
    args = parser.parse_args()
    run_steps = []
    run_rewards = []
    run_goals = 0

    def environment_reset():
        raise NotImplementedError("Reset your environment with this function")

    env = metasploit_gym.metasploit_env.MetasploitNetworkEnv(
        reset_function=environment_reset
    )
    for i in range(args.runs):
        steps, reward, done = run_random_agent(env, step_limit=5, verbose=True)
        run_steps.append(steps)
        run_rewards.append(reward)
        if done:
            run_goals += 1
    run_steps = np.array(run_steps)
    run_rewards = np.array(run_rewards)

    print(f"Mean steps = {run_steps.mean():.2f} +/- {run_steps.std():.2f}")
    print(f"Mean rewards = {run_rewards.mean():.2f} " f"+/- {run_rewards.std():.2f}")
    print(run_goals)
    print("out of")
    print(args.runs)
